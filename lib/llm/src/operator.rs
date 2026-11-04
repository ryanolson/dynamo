// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's operators as Dynamo operators and as native gdl operators: the preprocessor,
//! Migration, and the encoder and prefill routers. Backend is a gdl around node instead (see
//! `backend.rs`): its forward and backward halves run on either side of the hop below it.
//!
//! Each operator has one body, generic over [`Downstream`]: how it calls the stage below it.
//! Its Dynamo `Operator` impl passes the `Arc<dyn AsyncEngine>` it is given, and its gdl
//! `Operator` impl passes a [`GdlDownstream`] over gdl's `cx` and static `next`. So a native
//! stage builds no `Cx` of its own and no `Arc<dyn AsyncEngine>`.
//!
//! The request stays Dynamo's `SingleIn<T>` on both paths: its context carries the id, the
//! metadata and registry (session affinity, LoRA name, request metrics) and the stop and kill
//! that the stages below read. Items stay `Annotated<T>`; an operator that edits its items in
//! place reads and answers them as [`Downstream::Item`], which on gdl leaves each item in its
//! gdl `Result`, so it is not taken out and wrapped again at every stage.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_gdl_adapters::{AnnotatedItems, CheckedItems, CxBridge, PipelineEngine, enter};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{FutureExt, Stream, StreamExt};
use tracing::Instrument;

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

    /// The stage below as a Dynamo engine, for a request the operator sends on its own, after
    /// the one it was called for has ended (speculative prefill).
    fn engine(&self) -> Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>;

    /// The operator's response, as its caller takes it.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    type Response<U: Data>;

    /// Box the operator's response stream once, for its caller. `context` is the request's.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    fn respond<U, S>(stream: S, context: Arc<dyn AsyncEngineContext>) -> Self::Response<U>
    where
        U: Data,
        S: Stream<Item = Annotated<U>> + Send + 'static;

    /// A stream from [`send`](Self::send) as the operator's response, unchanged.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    fn pass(stream: Self::Stream) -> Self::Response<T>;

    /// The response stream of a Dynamo engine that the operator calls itself, not through the
    /// stage below, as the operator's response.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    fn adopt<U: Data>(stream: ManyOut<Annotated<U>>) -> Self::Response<U>;

    /// The form of the items an operator that edits them in place reads and answers.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    type Form: ItemForm;

    /// The response stream of [`send_items`](Self::send_items). On gdl, gdl's `Err` rule is
    /// already applied to it, as [`send`](Self::send) applies it, so every item is `Ok`.
    type Items: Stream<Item = Item<Self::Form, T>>
        + AsyncEngineContextProvider
        + Send
        + Unpin
        + 'static;

    /// Send `request` on, as [`send`](Self::send) does, and read the response as items of
    /// [`Form`](Self::Form).
    fn send_items(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Items>> + Send;

    /// Box a stream of items of [`Form`](Self::Form) once, as the operator's response, as
    /// [`respond`](Self::respond) does for `Annotated` items.
    ///
    /// Transition: remove when the gdl pipeline is the only local pipeline.
    fn respond_items<U, S>(stream: S, context: Arc<dyn AsyncEngineContext>) -> Self::Response<U>
    where
        U: Data,
        S: Stream<Item = Item<Self::Form, U>> + Send + 'static;
}

/// A stream item as the stage below sends it, for an operator that edits its items in place:
/// `Annotated<U>` on Dynamo ([`DynamoForm`]), and on gdl the gdl item,
/// `gdl::Result<Annotated<U>>` ([`GdlForm`]). Read and answered in this form, an item is never
/// taken out of its `Result` and wrapped again, which would move it (about 600 bytes) twice per
/// stage.
///
/// Transition: remove when the gdl pipeline is the only local pipeline.
pub(crate) trait ItemForm: Send + Sync + 'static {
    /// One item.
    type Item<U: Data>: Send + 'static;

    /// The `Annotated` value of `item`.
    fn item<U: Data>(item: &Self::Item<U>) -> &Annotated<U>;

    /// The `Annotated` value of `item`, to edit in place.
    fn item_mut<U: Data>(item: &mut Self::Item<U>) -> &mut Annotated<U>;

    /// `item` as an `Annotated` value, for a path that takes it by value.
    fn into_annotated<U: Data>(item: Self::Item<U>) -> Annotated<U>;

    /// `item` in this form.
    fn from_annotated<U: Data>(item: Annotated<U>) -> Self::Item<U>;

    /// `item` with its `Annotated` value converted by `convert`.
    fn map_item<U: Data, V: Data>(
        item: Self::Item<U>,
        convert: impl FnOnce(Annotated<U>) -> Annotated<V>,
    ) -> Self::Item<V>;
}

/// An item of form `F`.
pub(crate) type Item<F, U> = <F as ItemForm>::Item<U>;

/// Dynamo's items: `Annotated<U>` as it is.
///
/// Transition: remove when the gdl pipeline is the only local pipeline.
pub(crate) struct DynamoForm;

impl ItemForm for DynamoForm {
    type Item<U: Data> = Annotated<U>;

    #[inline]
    fn item<U: Data>(item: &Annotated<U>) -> &Annotated<U> {
        item
    }

    #[inline]
    fn item_mut<U: Data>(item: &mut Annotated<U>) -> &mut Annotated<U> {
        item
    }

    #[inline]
    fn into_annotated<U: Data>(item: Annotated<U>) -> Annotated<U> {
        item
    }

    #[inline]
    fn from_annotated<U: Data>(item: Annotated<U>) -> Annotated<U> {
        item
    }

    #[inline]
    fn map_item<U: Data, V: Data>(
        item: Annotated<U>,
        convert: impl FnOnce(Annotated<U>) -> Annotated<V>,
    ) -> Annotated<V> {
        convert(item)
    }
}

/// gdl's items, `gdl::Result<Annotated<U>>`, read through [`CheckedItems`], so always `Ok`.
pub(crate) struct GdlForm;

impl ItemForm for GdlForm {
    type Item<U: Data> = gdl::Result<Annotated<U>>;

    #[inline]
    fn item<U: Data>(item: &gdl::Result<Annotated<U>>) -> &Annotated<U> {
        match item {
            Ok(item) => item,
            Err(_) => unreachable!("checked items carry no `Err`"),
        }
    }

    #[inline]
    fn item_mut<U: Data>(item: &mut gdl::Result<Annotated<U>>) -> &mut Annotated<U> {
        match item {
            Ok(item) => item,
            Err(_) => unreachable!("checked items carry no `Err`"),
        }
    }

    #[inline]
    fn into_annotated<U: Data>(item: gdl::Result<Annotated<U>>) -> Annotated<U> {
        match item {
            Ok(item) => item,
            Err(_) => unreachable!("checked items carry no `Err`"),
        }
    }

    #[inline]
    fn from_annotated<U: Data>(item: Annotated<U>) -> gdl::Result<Annotated<U>> {
        Ok(item)
    }

    #[inline]
    fn map_item<U: Data, V: Data>(
        item: gdl::Result<Annotated<U>>,
        convert: impl FnOnce(Annotated<U>) -> Annotated<V>,
    ) -> gdl::Result<Annotated<V>> {
        item.map(convert)
    }
}
/// Dynamo's `next`, as a Dynamo operator's body calls it.
///
/// A newtype rather than an impl on `Arc<dyn AsyncEngine>`: inside an `async_trait` method the
/// compiler checks the body's future for `Send` with the trait object's lifetime erased, and an
/// impl for the `'static` trait object is then not general enough. The newtype fixes the
/// lifetime where it is defined.
///
/// Transition: remove when the gdl pipeline is the only local pipeline.
pub(crate) struct DynamoDownstream<Req: Data, T: Data>(
    pub(crate) Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>,
);

impl<Req: Data, T: Data> Downstream<Req, T> for DynamoDownstream<Req, T> {
    type Stream = ManyOut<Annotated<T>>;

    // The engine's own future, as the body awaited before it was shared.
    fn send(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send {
        self.0.generate(request)
    }

    fn forward(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Response<T>>> + Send {
        self.0.generate(request)
    }

    fn send_cleanup(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send {
        self.0.generate(request)
    }

    fn engine(&self) -> Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>> {
        Arc::clone(&self.0)
    }

    type Response<U: Data> = ManyOut<Annotated<U>>;

    fn respond<U, S>(stream: S, context: Arc<dyn AsyncEngineContext>) -> ManyOut<Annotated<U>>
    where
        U: Data,
        S: Stream<Item = Annotated<U>> + Send + 'static,
    {
        ResponseStream::new(Box::pin(stream), context)
    }

    fn pass(stream: ManyOut<Annotated<T>>) -> ManyOut<Annotated<T>> {
        stream
    }

    fn adopt<U: Data>(stream: ManyOut<Annotated<U>>) -> ManyOut<Annotated<U>> {
        stream
    }

    type Form = DynamoForm;

    type Items = ManyOut<Annotated<T>>;

    fn send_items(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Items>> + Send {
        self.0.generate(request)
    }

    fn respond_items<U, S>(stream: S, context: Arc<dyn AsyncEngineContext>) -> ManyOut<Annotated<U>>
    where
        U: Data,
        S: Stream<Item = Annotated<U>> + Send + 'static,
    {
        ResponseStream::new(Box::pin(stream), context)
    }
}

/// gdl's `cx` and `next`, as a native operator's body calls them.
pub(crate) struct GdlDownstream<N> {
    cx: gdl::Cx,
    next: N,
}

impl<N> GdlDownstream<N> {
    pub(crate) fn new(cx: gdl::Cx, next: N) -> Self {
        Self { cx, next }
    }
}

impl<N, Req, T> Downstream<Req, T> for GdlDownstream<N>
where
    N: gdl::Next<SingleIn<Req>, Annotated<T>>,
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
        // Carried as itself, so `native` gives back the same `gdl::Error`.
        self.next
            .generate(self.cx.clone(), request)
            .map(|result| result.map_err(anyhow::Error::new))
    }

    fn send_cleanup(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Stream>> + Send {
        let cx = CxBridge::cx_for_cleanup(request.context().as_ref());
        self.send_under(cx, request)
    }

    /// Transition: root with `PipelineEngine::new` when no Dynamo engine runs inside a gdl
    /// pipeline.
    fn engine(&self) -> Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>> {
        // Each call derives a fresh root `Cx`: the request's `cx` is cancelled once its
        // response ends. Below the preprocessor of pipeline A is a Dynamo engine, so a kill
        // reaches it through the request's context, not a cancel that would drop its call.
        Arc::new(PipelineEngine::for_dynamo_stages(self.next.clone()))
    }

    type Response<U: Data> = gdl::Items<Annotated<U>>;

    fn respond<U, S>(stream: S, _context: Arc<dyn AsyncEngineContext>) -> gdl::Items<Annotated<U>>
    where
        U: Data,
        S: Stream<Item = Annotated<U>> + Send + 'static,
    {
        gdl::items(stream.map(Ok))
    }

    fn pass(stream: AnnotatedItems<T>) -> gdl::Items<Annotated<T>> {
        gdl::items(stream.map(Ok))
    }

    fn adopt<U: Data>(stream: ManyOut<Annotated<U>>) -> gdl::Items<Annotated<U>> {
        gdl::items(stream.map(Ok))
    }

    type Form = GdlForm;

    type Items = CheckedItems<T>;

    fn send_items(
        &self,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<Self::Items>> + Send {
        let context = request.context();
        let root = self.cx.clone();
        self.next
            .generate(self.cx.clone(), request)
            .map(move |result| match result {
                Ok(items) => Ok(CheckedItems::new(items, root, context)),
                // Carried as itself, so `native` gives back the same `gdl::Error`.
                Err(error) => Err(anyhow::Error::new(error)),
            })
    }

    fn respond_items<U, S>(
        stream: S,
        _context: Arc<dyn AsyncEngineContext>,
    ) -> gdl::Items<Annotated<U>>
    where
        U: Data,
        S: Stream<Item = gdl::Result<Annotated<U>>> + Send + 'static,
    {
        gdl::items(stream)
    }
}

impl<N> GdlDownstream<N> {
    /// Send `request` on under `cx` and read the response as a Dynamo stream.
    fn send_under<Req, T>(
        &self,
        cx: gdl::Cx,
        request: SingleIn<Req>,
    ) -> impl Future<Output = anyhow::Result<AnnotatedItems<T>>> + Send
    where
        N: gdl::Next<SingleIn<Req>, Annotated<T>>,
        Req: Data,
        T: Data,
    {
        // The stream gives the request's own context: the stages below are Dynamo's, or keep
        // its stop and kill, so a stop sent to it reaches them, where a stop of `cx` would not.
        let context = request.context();
        let root = cx.clone();
        self.next
            .generate(cx, request)
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
