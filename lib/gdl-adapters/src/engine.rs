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
use crate::request::Carried;

/// A gdl [`Next`](gdl::Next), such as a [`Source`](gdl::Source), as a Dynamo
/// `AsyncEngine`, so that `ModelManager`, the HTTP service and Python store it
/// as they store any engine.
///
/// Each call derives a fresh `Cx` from the request's context and passes the
/// request on in its carrier `R`: Dynamo's `SingleIn<Req>` as it is, or
/// [`Boxed<Req>`](crate::Boxed) for a large request. The
/// `Cx` follows the context's stop and, unless a Dynamo engine runs in the
/// pipeline ([`for_dynamo_stages`](Self::for_dynamo_stages)), its kill (see
/// [`CxBridge`]). The response stream carries the request's
/// context, so a kill through the stream reaches the request.
///
/// An eager error that entered gdl through an adapter leaves as the
/// `anyhow::Error` it entered as; one raised by gdl leaves as a `DynamoError`.
/// A gdl `Err` item ends the stream: it leaves as a last error item, or with
/// no item when it is a cancel of a request that was killed, since a killed
/// Dynamo stream just ends.
pub struct PipelineEngine<N, Req, T, R = SingleIn<Req>> {
    next: N,
    /// Whether a kill of the request cancels the `Cx` (see [`CxBridge`]).
    ///
    /// Transition: remove when no Dynamo stage runs inside a gdl pipeline.
    cancel_on_kill: bool,
    types: PhantomData<fn(Req, R) -> T>,
}

impl<N, Req, T, R> PipelineEngine<N, Req, T, R> {
    /// Wrap a gdl pipeline of gdl-native nodes: a kill of the request cancels
    /// it ([`CxBridge::cx`]).
    pub fn new(next: N) -> Self {
        Self {
            next,
            cancel_on_kill: true,
            types: PhantomData,
        }
    }

    /// Wrap a gdl pipeline with a Dynamo engine in it: a kill reaches the
    /// engine through the request's context only, so gdl does not drop its
    /// call, and the request ends as the engine ends it
    /// ([`CxBridge::cx_for_dynamo_stages`]).
    ///
    /// Transition: remove when no Dynamo stage runs inside a gdl pipeline.
    pub fn for_dynamo_stages(next: N) -> Self {
        Self {
            next,
            cancel_on_kill: false,
            types: PhantomData,
        }
    }
}

#[async_trait]
impl<N, Req, T, R> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, anyhow::Error>
    for PipelineEngine<N, Req, T, R>
where
    N: gdl::Next<R, Annotated<T>>,
    Req: Data,
    T: Data,
    R: Carried<Req>,
{
    async fn generate(&self, request: SingleIn<Req>) -> anyhow::Result<ManyOut<Annotated<T>>> {
        let context = request.context();
        let cx = if self.cancel_on_kill {
            CxBridge::cx(request.controller())
        } else {
            CxBridge::cx_for_dynamo_stages(request.controller())
        };
        let root = cx.clone();
        match self.next.generate(cx, R::carry(request)).await {
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
/// so it sees the stages below as a Dynamo stream, as Dynamo's operators did
/// on the `link` chain. One that can work on the gdl item as it is reads through
/// [`CheckedItems`] instead, and saves taking each item out of its `Result`.
/// The two apply the same rule.
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
                Poll::Ready(last_item(error, &self.root, self.context.as_ref()))
            }
            Poll::Ready(None) => {
                self.items = None;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// gdl's `Err` rule, which [`AnnotatedItems`] and [`CheckedItems`] share: the
/// last item of a stream that ended in `error`, or `None` when it is a cancel
/// of a request that was killed, since a killed Dynamo stream just ends.
fn last_item<T>(
    error: gdl::Error,
    root: &gdl::Cx,
    context: &dyn AsyncEngineContext,
) -> Option<Annotated<T>> {
    let killed =
        error.kind() == gdl::ErrorKind::Cancelled && (root.is_cancelled() || context.is_killed());
    (!killed).then(|| exit_item(error))
}

/// gdl items with gdl's `Err` rule applied, as [`AnnotatedItems`] applies it,
/// but each item left in its `Result`: an `Err` item ends the stream, as a
/// last `Ok` error item, or with no item when it is a cancel of a request that
/// was killed. So every item it yields is `Ok`.
///
/// An `Ok` item is returned in the `Poll` the items below returned, so reading
/// through it moves no item. A native gdl operator that edits its items in
/// place and answers with gdl items reads its downstream through it.
pub struct CheckedItems<T> {
    items: Option<Items<Annotated<T>>>,
    root: gdl::Cx,
    context: Arc<dyn AsyncEngineContext>,
}

impl<T> CheckedItems<T> {
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

impl<T> std::fmt::Debug for CheckedItems<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedItems")
            .field("ended", &self.items.is_none())
            .field("root", &self.root)
            .field("context", &self.context.id())
            .finish()
    }
}

/// The request's Dynamo context, as a Dynamo response stream gives it.
impl<T: Data> AsyncEngineContextProvider for CheckedItems<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        Arc::clone(&self.context)
    }
}

impl<T: Data> Stream for CheckedItems<T> {
    type Item = gdl::Result<Annotated<T>>;

    fn poll_next(mut self: Pin<&mut Self>, task: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let Some(items) = self.items.as_mut() else {
            return Poll::Ready(None);
        };
        // Every path returns this one value, so the compiler can build it where the caller
        // reads it: an `Ok` item is not moved on its way through.
        let mut poll = items.as_mut().poll_next(task);
        match &mut poll {
            Poll::Ready(Some(Ok(_))) | Poll::Pending => {}
            Poll::Ready(Some(Err(_))) => {
                self.items = None;
                let Poll::Ready(Some(Err(error))) = std::mem::replace(&mut poll, Poll::Pending)
                else {
                    unreachable!("matched above");
                };
                poll = Poll::Ready(last_item(error, &self.root, self.context.as_ref()).map(Ok));
            }
            Poll::Ready(None) => self.items = None,
        }
        poll
    }
}
