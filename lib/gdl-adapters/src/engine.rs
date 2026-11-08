// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl pipeline as a Dynamo engine.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, ManyOut, PipelineError, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::Stream;

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
/// A gdl ending of the stream leaves as a last error item, or with no item
/// when it is a cancel of a request that was killed, since a killed Dynamo
/// stream just ends.
///
/// The pipeline is in gdl's ended form ([`gdl::EndedForm`]): its response leaves through
/// [`EndedItems`], which asks the pipeline why it ended once, at its end.
pub struct PipelineEngine<N, Req, T, R = SingleIn<Req>> {
    next: N,
    /// Whether a kill of the request cancels the `Cx` (see [`CxBridge`]).
    cancel_on_kill: bool,
    /// The ids of the requests still in their dispatch (see [`InFlightIds`]).
    in_flight: InFlightIds,
    types: PhantomData<fn(Req, R) -> T>,
}

impl<N, Req, T, R> PipelineEngine<N, Req, T, R> {
    /// Wrap a gdl pipeline of gdl-native nodes: a kill of the request cancels
    /// it ([`CxBridge::cx`]).
    pub fn new(next: N) -> Self {
        Self {
            next,
            cancel_on_kill: true,
            in_flight: InFlightIds::default(),
            types: PhantomData,
        }
    }

    /// Wrap a gdl pipeline with a Dynamo engine in it: a kill reaches the
    /// engine through the request's context only, so gdl does not drop its
    /// call, and the request ends as the engine ends it
    /// ([`CxBridge::cx_for_dynamo_stages`]). Every frontend pipeline is rooted
    /// with it, so a kill ends a request as Dynamo's operators end it.
    pub fn for_dynamo_stages(next: N) -> Self {
        Self {
            next,
            cancel_on_kill: false,
            in_flight: InFlightIds::default(),
            types: PhantomData,
        }
    }
}

#[async_trait]
impl<N, Req, T, R> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, anyhow::Error>
    for PipelineEngine<N, Req, T, R>
where
    N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
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
        let claim = self.in_flight.claim(context.id())?;
        let dispatched = self.next.generate(cx, R::carry(request)).await;
        drop(claim);
        match dispatched {
            Ok(items) => Ok(ResponseStream::new(
                Box::pin(EndedItems::new(items, root, Arc::clone(&context))),
                context,
            )),
            Err(error) => Err(exit(error)),
        }
    }
}

/// The ids of the requests a [`PipelineEngine`] is still dispatching. A second request with
/// one of them is refused with [`PipelineError::DuplicateRequestId`], as Dynamo's pipeline
/// source refused it: its response map routed a response back by request id. An id is held
/// until the dispatch returns, not for the whole stream.
#[derive(Default)]
struct InFlightIds(Mutex<HashSet<Arc<str>>>);

impl InFlightIds {
    fn claim(&self, id: &str) -> Result<InFlightClaim<'_>, PipelineError> {
        let mut ids = self.lock();
        if ids.contains(id) {
            return Err(PipelineError::DuplicateRequestId(id.to_string()));
        }
        // One allocation, shared by the set and the claim that removes it.
        let id: Arc<str> = Arc::from(id);
        ids.insert(Arc::clone(&id));
        Ok(InFlightClaim { owner: self, id })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashSet<Arc<str>>> {
        // A poisoned lock means another thread panicked holding it; the set is still
        // consistent, because every update is a single insert or remove.
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Frees its id when dropped: when the dispatch returns or fails, or when the call is dropped.
struct InFlightClaim<'a> {
    owner: &'a InFlightIds,
    id: Arc<str>,
}

impl Drop for InFlightClaim<'_> {
    fn drop(&mut self) {
        self.owner.lock().remove(&self.id);
    }
}

/// gdl items in the ended form read as a Dynamo stream of `Annotated<T>`: each item passes
/// through as it is, and at its end the stream asks its input once why it ended. An ending
/// leaves by the rule [`PipelineEngine`] gives: a last error item, or no item when it is a
/// cancel of a request that was killed ([`last_item`]).
///
/// It is the exit of a [`PipelineEngine`] in the ended form, and the input of a native node: it
/// gives the request's Dynamo context and [`CutShort`], with no conversion per item. A node
/// that reads its input through it has handled the input's ending, so as an ended stream it
/// ends cleanly.
pub struct EndedItems<T> {
    items: Option<gdl::Ended<Annotated<T>>>,
    root: gdl::Cx,
    context: Arc<dyn AsyncEngineContext>,
}

impl<T> EndedItems<T> {
    /// Read `items`. `root` is the `Cx` the request was sent with and
    /// `context` its Dynamo context: a cancel of either one is a kill.
    pub fn new(
        items: gdl::Ended<Annotated<T>>,
        root: gdl::Cx,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        Self {
            items: Some(items),
            root,
            context,
        }
    }

    /// The input ended: ask it why, once, and drop it. Out of line, as it runs once per stream.
    #[cold]
    #[inline(never)]
    fn end(&mut self) -> Option<Annotated<T>> {
        let mut items = self.items.take()?;
        let ending = items.as_mut().ending();
        drop(items);
        last_item(ending?, &self.root, self.context.as_ref())
    }
}

impl<T> std::fmt::Debug for EndedItems<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EndedItems")
            .field("ended", &self.items.is_none())
            .field("root", &self.root)
            .field("context", &self.context.id())
            .finish()
    }
}

/// The request's Dynamo context, as a Dynamo response stream gives it.
impl<T: Data> AsyncEngineContextProvider for EndedItems<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        Arc::clone(&self.context)
    }
}

impl<T: Data> Stream for EndedItems<T> {
    type Item = Annotated<T>;

    fn poll_next(mut self: Pin<&mut Self>, task: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let Some(items) = this.items.as_mut() else {
            return Poll::Ready(None);
        };
        // gdl's boundary shape: no item is read, and the end's poll is consumed before
        // anything is called, so the compiler builds each item where the caller reads it.
        // Matching the end with `if let` and returning the poll after it copies every item
        // (632 bytes for an `Annotated<LLMEngineOutput>`), measured.
        let polled = items.as_mut().poll_next(task);
        match polled {
            Poll::Ready(Some(_)) | Poll::Pending => {}
            Poll::Ready(None) => {
                drop(polled);
                return Poll::Ready(this.end());
            }
        }
        polled
    }
}

/// A reader of this stream has had the input's ending as a last item, if it had one: the
/// stream ends cleanly.
impl<T: Data> gdl::EndedStream for EndedItems<T> {
    fn ending(self: Pin<&mut Self>) -> Option<gdl::Error> {
        None
    }
}

impl<T> CutShort for EndedItems<T> {
    fn cut_short(&self) -> bool {
        cut_short(&self.root, self.context.as_ref())
    }
}

/// Whether the request that `cx` and `context` belong to was cut short: killed through its
/// Dynamo context, or cancelled or past its deadline through `cx` ([`gdl::Cx::check`]). A stop
/// does not cut a request short: the answer that ends after it is whole.
///
/// A nested call whose request is killed or passes its deadline ends quietly, as if it had
/// finished; only the request's edge reports why. So a node that writes closing output when
/// its input ends, such as a `finish_reason`, a usage chunk or a flush of held text, asks this
/// once, on that clean end, and writes nothing when it is `true`. It reads both sides: under a
/// pipeline rooted with [`PipelineEngine::for_dynamo_stages`], a kill does not cancel `cx`.
pub fn cut_short(cx: &gdl::Cx, context: &dyn AsyncEngineContext) -> bool {
    cx.check().is_err() || context.is_killed()
}

/// A node's input that can say, once it has ended, whether its request was cut short
/// ([`cut_short`]). A node asks once per stream, on the clean end, never per item.
pub trait CutShort {
    /// Whether the request this stream answers was cut short.
    fn cut_short(&self) -> bool;
}

/// The rule for a gdl ending ([`EndedItems`]): the last item of a stream that
/// ended in `error`, or `None` when it is a cancel of a request that was
/// killed, since a killed Dynamo stream just ends.
fn last_item<T>(
    error: gdl::Error,
    root: &gdl::Cx,
    context: &dyn AsyncEngineContext,
) -> Option<Annotated<T>> {
    let killed =
        error.kind() == gdl::ErrorKind::Cancelled && (root.is_cancelled() || context.is_killed());
    (!killed).then(|| exit_item(error))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_runtime::pipeline::Context;

    use super::*;
    use crate::bridge::CxBridge;

    fn request() -> Context<()> {
        Context::with_id_and_metadata((), "req-1".to_string(), Default::default())
    }

    /// Read as a node reads its input: under a `Cx` of a pipeline rooted with
    /// `for_dynamo_stages`, which a Dynamo kill does not cancel.
    fn input(request: &Context<()>, cx: gdl::Cx) -> EndedItems<()> {
        EndedItems::new(
            gdl::ended::clean(futures::stream::empty()),
            cx,
            request.context(),
        )
    }

    fn cut(request: &Context<()>, cx: gdl::Cx) -> [bool; 2] {
        let input = input(request, cx.clone());
        [cut_short(&cx, request.controller()), input.cut_short()]
    }

    #[test]
    fn a_live_request_is_not_cut_short() {
        let request = request();
        let cx = CxBridge::cx_for_dynamo_stages(request.controller());
        assert_eq!(cut(&request, cx), [false; 2]);
    }

    /// A Dynamo kill reaches the node through the request's context only.
    #[test]
    fn a_kill_cuts_the_request_short() {
        let request = request();
        let cx = CxBridge::cx_for_dynamo_stages(request.controller());
        request.controller().kill();
        assert!(cx.check().is_ok(), "the kill does not cancel this `Cx`");
        assert_eq!(cut(&request, cx), [true; 2]);
    }

    #[test]
    fn a_cancel_of_the_cx_cuts_the_request_short() {
        let request = request();
        let cx = CxBridge::cx_for_dynamo_stages(request.controller());
        cx.cancel();
        assert_eq!(cut(&request, cx), [true; 2]);
    }

    #[test]
    fn a_passed_deadline_cuts_the_request_short() {
        let request = request();
        let cx = CxBridge::cx_for_dynamo_stages(request.controller()).with_timeout(Duration::ZERO);
        assert_eq!(cut(&request, cx), [true; 2]);
    }

    /// A stop asks for a graceful end: the answer that ends after it is whole.
    #[test]
    fn a_stop_does_not_cut_the_request_short() {
        let request = request();
        let cx = CxBridge::cx(request.controller());
        request.controller().stop_generating();
        assert!(cx.is_stopping(), "the stop reaches the `Cx`");
        assert!(cx.check().is_ok(), "gdl's check does not report a stop");
        assert_eq!(cut(&request, cx), [false; 2]);
    }
}
