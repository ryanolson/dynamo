// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's operators as Dynamo operators and as native gdl operators: the embeddings
//! preprocessor, Migration, and the encoder and prefill routers. Backend and the chat and
//! completions preprocessors are gdl around nodes instead (`BackendState`, `PreprocessorNode`):
//! their forward and backward halves run on either side of the hop below them.
//!
//! Each operator has one body, generic over [`Downstream`]: how it calls the stage below it.
//! Its Dynamo `Operator` impl passes the `Arc<dyn AsyncEngine>` it is given, and its gdl
//! `Operator` impl passes a [`GdlDownstream`] over gdl's `cx` and static `next`. So a native
//! stage builds no `Cx` of its own and no `Arc<dyn AsyncEngine>`.
//!
//! The request stays Dynamo's `SingleIn<T>` on both paths: its context carries the id, the
//! metadata and registry (session affinity, LoRA name, request metrics) and the stop and kill
//! that the stages below read. On gdl the chat, completion and preprocessed requests travel
//! boxed between the nodes ([`Boxed`](dynamo_gdl_adapters::Boxed)), so gdl's chain moves a pointer, not the request. A
//! native operator takes the request out of its box to run its body, and the body's first send
//! puts the request it sends in that box again ([`GdlDownstream::with_spare`]). Items stay `Annotated<T>`; an operator that edits its items in
//! place reads and answers them as [`Downstream::Item`], which on gdl leaves each item in its
//! gdl `Result`, so it is not taken out and wrapped again at every stage.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_gdl_adapters::{AnnotatedItems, Carried, CheckedItems, CxBridge, enter};
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{FutureExt, Stream};
use tracing::Instrument;

#[cfg(test)]
pub(crate) mod testing;

/// The stage below an operator, as the operator's body calls it.
///
/// Transition: remove when the gdl pipeline is the only local pipeline.
pub(crate) trait Downstream<Req: Data, T: Data>: Send + Sync {
    /// The response stream of one call. It gives the request's Dynamo context, which the body
    /// stops to stop the stages below.
    type Stream: Stream<Item = Annotated<T>> + AsyncEngineContextProvider + Send + Unpin + 'static;

    /// Send `request` on.
    ///
    /// Not named `generate`: on an `Arc<dyn AsyncEngine>` that name would shadow the engine's
    /// own wherever this trait is in scope.
    fn send(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send;

    /// Send `request` on and answer with the response of the stage below as it is, for an
    /// operator that adds nothing to it. On gdl this reads no item, where [`send`](Self::send)
    /// and [`respond`](Self::respond) would add a layer.
    fn forward(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Response<T>>> + Send;

    /// Send `request` on in a call that a kill of the request does not end: the leg that
    /// releases what a worker below holds for the request, such as the prefill router's decode
    /// leg after a KV handoff. The stage below still sees the kill through the request's
    /// context. On gdl the call runs under a root `Cx` of its own, since a cancel of the
    /// request's `cx` would refuse it.
    fn send_cleanup(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send;

    /// The operator's response, as its caller takes it.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    type Response<U: Data>;
}

/// The `Annotated` value of an item read through [`CheckedItems`], which is always `Ok`. An
/// operator that edits its items in place reads them in their gdl `Result`, so an item is never
/// taken out of its `Result` and wrapped again, which would move it (about 600 bytes) twice per
/// stage.
#[inline]
pub(crate) fn checked<U>(item: &gdl::Result<Annotated<U>>) -> &Annotated<U> {
    match item {
        Ok(item) => item,
        Err(_) => unreachable!("checked items carry no `Err`"),
    }
}

/// As [`checked`], to edit in place.
#[inline]
pub(crate) fn checked_mut<U>(item: &mut gdl::Result<Annotated<U>>) -> &mut Annotated<U> {
    match item {
        Ok(item) => item,
        Err(_) => unreachable!("checked items carry no `Err`"),
    }
}

/// As [`checked`], by value, for a path that takes the item out of its `Result`.
#[inline]
pub(crate) fn into_checked<U>(item: gdl::Result<Annotated<U>>) -> Annotated<U> {
    match item {
        Ok(item) => item,
        Err(_) => unreachable!("checked items carry no `Err`"),
    }
}

/// gdl's `cx` and `next`, as a native operator's body calls them. `next` takes requests in
/// the carrier `R` (see [`Carried`]): boxed for Migration and the routers, `SingleIn` itself for
/// the embeddings preprocessor.
pub(crate) struct GdlDownstream<N, R: Carried<Req>, Req: Data> {
    cx: gdl::Cx,
    next: N,
    /// The box of the request the operator was given, for the first request its body sends.
    spare: R::Spare,
}

impl<N, R: Carried<Req>, Req: Data> GdlDownstream<N, R, Req> {
    pub(crate) fn new(cx: gdl::Cx, next: N) -> Self {
        Self::with_spare(cx, next, R::Spare::default())
    }

    /// As [`new`](Self::new), and the first request the body sends goes in `spare`: the memory
    /// of the request the operator took apart ([`Carried::take`]), so the operator adds no
    /// allocation.
    pub(crate) fn with_spare(cx: gdl::Cx, next: N, spare: R::Spare) -> Self {
        Self { cx, next, spare }
    }

    /// Send `request` on in its carrier, as it came, and answer with the response of the stage
    /// below as it is: [`Downstream::forward`] for a request that does not go through the body.
    pub(crate) fn forward_carried<T>(
        &self,
        request: R,
    ) -> impl Future<Output = anyhow::Result<gdl::Items<Annotated<T>>>> + Send
    where
        N: gdl::Next<R, Annotated<T>>,
        T: Data,
    {
        // Carried as itself, so `native` gives back the same `gdl::Error`.
        self.next
            .generate(self.cx.clone(), request)
            .map(|result| result.map_err(anyhow::Error::new))
    }
}

impl<N, R, Req, T> Downstream<Req, T> for GdlDownstream<N, R, Req>
where
    N: gdl::Next<R, Annotated<T>>,
    R: Carried<Req>,
    Req: Data,
    T: Data,
{
    type Stream = AnnotatedItems<T>;

    fn send(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send {
        self.send_under(self.cx.clone(), request)
    }

    fn forward(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Response<T>>> + Send {
        self.forward_carried(R::carry_in(request, &self.spare))
    }

    fn send_cleanup(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send {
        let cx = CxBridge::cx_for_cleanup(request.context().as_ref());
        self.send_under(cx, request)
    }

    type Response<U: Data> = gdl::Items<Annotated<U>>;
}

impl<N, R: Carried<Req>, Req: Data> GdlDownstream<N, R, Req> {
    /// Send `request` on, as [`Downstream::send`] does, and read the response's items in their
    /// gdl `Result`, for an operator that edits them in place (see [`checked_mut`]). gdl's
    /// `Err` rule is already applied to the stream, so every item is `Ok`.
    pub(crate) fn send_items<T>(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<CheckedItems<T>>> + Send
    where
        N: gdl::Next<R, Annotated<T>>,
        T: Data,
    {
        let context = request.context();
        let root = self.cx.clone();
        self.next
            .generate(self.cx.clone(), R::carry_in(request, &self.spare))
            .map(move |result| match result {
                Ok(items) => Ok(CheckedItems::new(items, root, context)),
                // Carried as itself, so `native` gives back the same `gdl::Error`.
                Err(error) => Err(anyhow::Error::new(error)),
            })
    }

    /// Send `request` on under `cx` and read the response as a Dynamo stream.
    fn send_under<T>(
        &self,
        cx: gdl::Cx,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<AnnotatedItems<T>>> + Send
    where
        N: gdl::Next<R, Annotated<T>>,
        T: Data,
    {
        // The stream gives the request's own context: the stages below are Dynamo's, or keep
        // its stop and kill, so a stop sent to it reaches them, where a stop of `cx` would not.
        let context = request.context();
        let root = cx.clone();
        self.next
            .generate(cx, R::carry_in(request, &self.spare))
            .map(move |result| match result {
                Ok(items) => Ok(AnnotatedItems::new(items, root, context)),
                // Carried as itself, so `native` gives back the same `gdl::Error`.
                Err(error) => Err(anyhow::Error::new(error)),
            })
    }
}

/// A Dynamo operator `O` as a native gdl `Operator`, keyed by `K`.
///
/// gdl's `Operator` names its request and item types, so an operator that serves several of
/// them is one gdl operator per type, and `K` names the type that tells them apart: the
/// request for Backend and the preprocessor, the item for Migration. An operator with one impl
/// takes its request type. Build it once, at composition.
pub struct GdlOperator<O, K> {
    pub(crate) op: Arc<O>,
    key: PhantomData<fn(K)>,
}

impl<O, K> GdlOperator<O, K> {
    /// `op` as a gdl operator for `K`, for an `OperatorStage`.
    pub fn new(op: Arc<O>) -> Self {
        Self {
            op,
            key: PhantomData,
        }
    }
}

/// Run `work` as a task that a drop of the caller's future detaches rather than cancels, and
/// wait for its result: for a native operator whose work must outlive a gdl cancel, which
/// drops the operator's future. A panic in `work` resumes in the caller.
///
/// Not an `async fn`: the task starts here, so the future the caller holds is the task's
/// handle alone, not `work` as well. `work` is boxed, one allocation per call: a router's work
/// holds the stages below it, and the task's future type would otherwise nest all of them
/// inside tokio's task wrappers, deeper than the compiler's recursion limit.
pub(crate) fn detached<T: Send + 'static>(
    work: impl Future<Output = anyhow::Result<T>> + Send + 'static,
) -> impl Future<Output = anyhow::Result<T>> + Send {
    let work: Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send>> = Box::pin(work);
    let task = tokio::spawn(work.instrument(tracing::Span::current()));
    async move {
        match task.await {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => Err(anyhow::Error::new(error).context("detached operator task ended")),
        }
    }
}

/// A body's error as a native operator's: an error that came from below leaves as the same
/// `gdl::Error`, and one the body raised enters gdl with its `anyhow::Error` as the source.
pub(crate) fn native<T>(result: anyhow::Result<gdl::Items<T>>) -> gdl::Result<gdl::Items<T>> {
    result.map_err(gdl_error)
}

/// A body's error as gdl's: the same `gdl::Error` if it came from below, else one that carries
/// it as its source.
pub(crate) fn gdl_error(error: anyhow::Error) -> gdl::Error {
    error.downcast::<gdl::Error>().unwrap_or_else(enter)
}
