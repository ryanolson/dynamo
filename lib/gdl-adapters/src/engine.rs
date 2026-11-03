// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl pipeline as a Dynamo engine.

use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, ManyOut, SingleIn, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::Stream;
use gdl::Items;

use crate::bridge::CxBridge;
use crate::error::{exit, exit_item};

/// A gdl [`Next`](gdl::Next), such as a [`Source`](gdl::Source), as a Dynamo
/// `AsyncEngine`, so that `ModelManager`, the HTTP service and Python store it
/// as they store any engine.
///
/// Each call derives a fresh `Cx` from the request's context
/// ([`CxBridge::cx`]) and passes the request on as it is: the gdl request type
/// is Dynamo's `SingleIn<Req>`. The response stream carries the request's
/// context, so a kill through the stream reaches the request.
///
/// An eager error that entered gdl through an adapter leaves as the
/// `anyhow::Error` it entered as; one raised by gdl leaves as a `DynamoError`.
/// A gdl `Err` item ends the stream: it leaves as a last error item, or with
/// no item when it is a cancel of a request that was killed, since a killed
/// Dynamo stream just ends.
pub struct PipelineEngine<N, Req, T> {
    next: N,
    types: PhantomData<fn(Req) -> T>,
}

impl<N, Req, T> PipelineEngine<N, Req, T> {
    /// Wrap a gdl pipeline.
    pub fn new(next: N) -> Self {
        Self {
            next,
            types: PhantomData,
        }
    }
}

#[async_trait]
impl<N, Req, T> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, anyhow::Error>
    for PipelineEngine<N, Req, T>
where
    N: gdl::Next<SingleIn<Req>, Annotated<T>>,
    Req: Data,
    T: Data,
{
    async fn generate(&self, request: SingleIn<Req>) -> anyhow::Result<ManyOut<Annotated<T>>> {
        let context = request.context();
        let cx = CxBridge::cx(request.controller());
        let root = cx.clone();
        match self.next.generate(cx, request).await {
            Ok(items) => Ok(ResponseStream::new(
                Box::pin(AnnotatedItems::new(items, root, Arc::clone(&context))),
                context,
            )),
            Err(error) => Err(exit(error)),
        }
    }
}

/// gdl items read as a Dynamo stream of `Annotated<T>`, by the rule
/// [`PipelineEngine`] gives: a gdl `Err` item ends the stream, as a last error
/// item, or with no item when it is a cancel of a request that was killed.
///
/// A native gdl operator over Dynamo items reads its downstream through it,
/// so it sees what a Dynamo operator sees from the same stages.
pub struct AnnotatedItems<T> {
    items: Option<Items<Annotated<T>>>,
    root: gdl::Cx,
    context: Arc<dyn AsyncEngineContext>,
}

impl<T> AnnotatedItems<T> {
    /// Read `items`. `root` is the `Cx` the request was sent with and
    /// `context` its Dynamo context: a cancel of either one is a kill.
    pub fn new(
        items: Items<Annotated<T>>,
        root: gdl::Cx,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        Self {
            items: Some(items),
            root,
            context,
        }
    }
}

impl<T> std::fmt::Debug for AnnotatedItems<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnotatedItems")
            .field("ended", &self.items.is_none())
            .field("root", &self.root)
            .field("context", &self.context.id())
            .finish()
    }
}

/// The request's Dynamo context, as a Dynamo response stream gives it.
impl<T: Data> AsyncEngineContextProvider for AnnotatedItems<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        Arc::clone(&self.context)
    }
}

impl<T: Data> Stream for AnnotatedItems<T> {
    type Item = Annotated<T>;

    fn poll_next(mut self: Pin<&mut Self>, task: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let Some(items) = self.items.as_mut() else {
            return Poll::Ready(None);
        };
        match items.as_mut().poll_next(task) {
            Poll::Ready(Some(Ok(item))) => Poll::Ready(Some(item)),
            Poll::Ready(Some(Err(error))) => {
                self.items = None;
                let killed = error.kind() == gdl::ErrorKind::Cancelled
                    && (self.root.is_cancelled() || self.context.is_killed());
                Poll::Ready((!killed).then(|| exit_item(error)))
            }
            Poll::Ready(None) => {
                self.items = None;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
