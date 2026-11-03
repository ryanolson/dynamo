// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend and the preprocessor as Dynamo operators and as native gdl operators.
//!
//! Each operator has one body, generic over [`Downstream`]: how it calls the stage below it.
//! Its Dynamo `Operator` impl passes the `Arc<dyn AsyncEngine>` it is given, and its gdl
//! `Operator` impl passes a [`GdlDownstream`] over gdl's `cx` and static `next`. So a native
//! stage builds no `Cx` of its own and no `Arc<dyn AsyncEngine>`.
//!
//! The request stays Dynamo's `SingleIn<T>` on both paths: its context carries the id, the
//! metadata and registry (session affinity, LoRA name, request metrics) and the stop and kill
//! that the stages below read. Items stay `Annotated<T>`.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use dynamo_gdl_adapters::{AnnotatedItems, PipelineEngine, enter};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{FutureExt, Stream, StreamExt};

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
        // The stream gives the request's own context: the stages below are Dynamo's, or keep
        // its stop and kill, so a stop sent to it reaches them, where a stop of `cx` would not.
        let context = request.context();
        let root = self.cx.clone();
        self.next
            .generate(self.cx.clone(), request)
            .map(move |result| match result {
                Ok(items) => Ok(AnnotatedItems::new(items, root, context)),
                // Carried as itself, so `native` gives back the same `gdl::Error`.
                Err(error) => Err(anyhow::Error::new(error)),
            })
    }

    /// Transition: root with `PipelineEngine::new` when Migration and the encoder and prefill
    /// routers run as native gdl stages.
    fn engine(&self) -> Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>> {
        // Each call derives a fresh root `Cx`: the request's `cx` is cancelled once its
        // response ends. The stages below are Dynamo's, so a kill reaches them through the
        // request's context, not a cancel that would drop a call one is still finishing.
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
}

/// A Dynamo operator `O` as a native gdl `Operator` for requests of type `Req`.
///
/// gdl's `Operator` names its request and item types, so an operator that serves several
/// request types, as Backend and the preprocessor do, is one gdl operator per type. Build it
/// once, at composition.
pub struct GdlOperator<O, Req> {
    pub(crate) op: Arc<O>,
    req: PhantomData<fn(Req)>,
}

impl<O, Req> GdlOperator<O, Req> {
    /// `op` as a gdl operator for `Req`, for an `OperatorStage`.
    pub fn new(op: Arc<O>) -> Self {
        Self {
            op,
            req: PhantomData,
        }
    }
}

/// A body's error as a native operator's: an error that came from below leaves as the same
/// `gdl::Error`, and one the body raised enters gdl with its `anyhow::Error` as the source.
pub(crate) fn native<T>(result: anyhow::Result<gdl::Items<T>>) -> gdl::Result<gdl::Items<T>> {
    result.map_err(|error| error.downcast::<gdl::Error>().unwrap_or_else(enter))
}
