// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's operators as native gdl operators: the embeddings preprocessor, Migration, and the
//! encoder and prefill routers. Backend and the chat and completions preprocessors are gdl
//! around nodes instead (`BackendState`, `PreprocessorNode`): their forward and backward halves
//! run on either side of the hop below them.
//!
//! Each native operator's body calls the stage below it through a [`GdlDownstream`] over gdl's
//! `cx` and static `next`, so a native stage builds no `Cx` of its own and no
//! `Arc<dyn AsyncEngine>`.
//!
//! The request stays Dynamo's `SingleIn<T>`: its context carries the id, the metadata and
//! registry (session affinity, LoRA name, request metrics) and the stop and kill that the stages
//! below read. The chat, completion and preprocessed requests travel boxed between the nodes
//! ([`Boxed`](dynamo_gdl_adapters::Boxed)), so gdl's chain moves a pointer, not the request. A
//! native operator takes the request out of its box to run its body, and the body's first send
//! puts the request it sends in that box again ([`GdlDownstream::with_spare`]). Items stay
//! `Annotated<T>`, in gdl's ended form: they cross each stage as they are, and an ending of the
//! stages below reaches an operator as a last item ([`EndedItems`]).

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_gdl_adapters::{Carried, CxBridge, EndedItems, enter};
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::FutureExt;

#[cfg(test)]
pub(crate) mod testing;

/// The stage below a native operator, as the operator's body calls it: gdl's `cx` and `next`.
/// `next` takes requests in the carrier `R` (see [`Carried`]): boxed for Migration and the
/// routers, `SingleIn` itself for the embeddings preprocessor.
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

    /// Send `request` on, and read the response as a Dynamo stream. The stream gives the
    /// request's Dynamo context, which the body stops to stop the stages below.
    pub(crate) fn send<T>(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<EndedItems<T>>> + Send
    where
        N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
        T: Data,
    {
        self.send_under(self.cx.clone(), request)
    }

    /// Send `request` on and answer with the response of the stage below as it is, for an
    /// operator that adds nothing to it. This reads no item, where [`send`](Self::send) would
    /// add a layer.
    pub(crate) fn forward<T>(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<gdl::Ended<Annotated<T>>>> + Send
    where
        N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
        T: Data,
    {
        self.forward_carried(R::carry_in(request, &self.spare))
    }

    /// Send `request` on in its carrier, as it came, and answer with the response of the stage
    /// below as it is: [`forward`](Self::forward) for a request that does not go through the
    /// body.
    pub(crate) fn forward_carried<T>(
        &self,
        request: R,
    ) -> impl Future<Output = anyhow::Result<gdl::Ended<Annotated<T>>>> + Send
    where
        N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
        T: Data,
    {
        // Carried as itself, so `native` gives back the same `gdl::Error`.
        self.next
            .generate(self.cx.clone(), request)
            .map(|result| result.map_err(anyhow::Error::new))
    }

    /// Send `request` on in a call that a kill of the request does not end: the leg that
    /// releases what a worker below holds for the request, such as the prefill router's decode
    /// leg after a KV handoff. The stage below still sees the kill through the request's
    /// context. The call runs under a root `Cx` of its own, since a cancel of the request's
    /// `cx` would refuse it.
    pub(crate) fn send_cleanup<T>(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<EndedItems<T>>> + Send
    where
        N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
        T: Data,
    {
        let cx = CxBridge::cx_for_cleanup(request.context().as_ref());
        self.send_under(cx, request)
    }

    /// Send `request` on under `cx` and read the response as a Dynamo stream.
    fn send_under<T>(
        &self,
        cx: gdl::Cx,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<EndedItems<T>>> + Send
    where
        N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
        T: Data,
    {
        // The stream gives the request's own context: the stages below are Dynamo's, or keep
        // its stop and kill, so a stop sent to it reaches them, where a stop of `cx` would not.
        let context = request.context();
        let root = cx.clone();
        self.next
            .generate(cx, R::carry_in(request, &self.spare))
            .map(move |result| match result {
                Ok(items) => Ok(EndedItems::new(items, root, context)),
                // Carried as itself, so `native` gives back the same `gdl::Error`.
                Err(error) => Err(anyhow::Error::new(error)),
            })
    }
}

/// One of Dynamo's operators `O` (Backend, the preprocessor, Migration or a router) as a gdl
/// operator, keyed by `K`.
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

/// `work` on the heap, run in the caller's future: a drop of that future cancels it, as a drop
/// of Dynamo's call cancels the route inside it. One allocation per call: a router's work holds
/// the stages below it, and held inline its future type nests all of them, deeper than the
/// compiler's recursion limit.
pub(crate) fn boxed_work<T: Send + 'static>(
    work: impl Future<Output = anyhow::Result<T>> + Send + 'static,
) -> Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send>> {
    Box::pin(work)
}

/// A body's error as a native operator's: an error that came from below leaves as the same
/// `gdl::Error`, and one the body raised enters gdl with its `anyhow::Error` as the source.
pub(crate) fn native<T>(result: anyhow::Result<gdl::Ended<T>>) -> gdl::Result<gdl::Ended<T>> {
    result.map_err(gdl_error)
}

/// A body's error as gdl's: the same `gdl::Error` if it came from below, else one that carries
/// it as its source.
pub(crate) fn gdl_error(error: anyhow::Error) -> gdl::Error {
    error.downcast::<gdl::Error>().unwrap_or_else(enter)
}
