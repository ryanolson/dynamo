// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stop and kill between a Dynamo request context and a gdl [`Cx`].

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use dynamo_runtime::engine::{AsyncEngineContext, Data};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context, async_trait};
use gdl::Cx;
use tokio::sync::oneshot;

/// The `Cx` metadata key that holds the Dynamo request id, for logs.
pub const REQUEST_ID: &str = "dynamo.request_id";

/// Builds a [`Cx`] from a Dynamo request context, and a Dynamo [`Context`]
/// from a `Cx`.
///
/// The Dynamo context is the source of truth for the request: a `Cx` made
/// from it follows its stop and kill, and nothing is stored in its registry.
/// Stop maps to [`Cx::request_stop`] and kill to [`Cx::cancel`]. A `Cx` never
/// leaves the cancelled state, so a kill stays a kill whatever comes after it,
/// on either side.
///
/// Except above a Dynamo engine. gdl's `Hop` races each inner call against
/// the attempt's cancel and drops the call when the cancel wins, and refuses
/// to start a call for a cancelled `Cx`. A Dynamo engine handles a kill
/// itself, and ends a killed request with an empty stream, where a dropped
/// call fails it as `Cancelled`. So a pipeline with a Dynamo engine in it
/// derives its `Cx` with
/// [`cx_for_dynamo_stages`](Self::cx_for_dynamo_stages), which a kill does
/// not cancel; the engine sees the kill through its own context, as it did
/// on Dynamo's `link` chain. [`cx`](Self::cx) is for gdl-native nodes.
pub struct CxBridge;

impl CxBridge {
    /// A fresh root `Cx` that follows `context`: a stop of `context` stops it,
    /// and a kill cancels it. The Dynamo id is in its metadata under
    /// [`REQUEST_ID`].
    ///
    /// The `Cx` is linked as a child of `context`, so it lives as long as
    /// `context` does. Signals go one way only: nothing that happens to the
    /// `Cx` reaches `context`, because gdl cancels every attempt when it ends,
    /// also when it ends normally.
    pub fn cx(context: &dyn AsyncEngineContext) -> Cx {
        Self::linked(context, true)
    }

    /// A fresh root `Cx` that follows `context`'s stop but not its kill, for a
    /// pipeline with a Dynamo engine in it (see [`CxBridge`]). Otherwise as
    /// [`cx`](Self::cx). Every frontend pipeline is rooted with it, so a kill ends a request
    /// as Dynamo's operators end it.
    pub fn cx_for_dynamo_stages(context: &dyn AsyncEngineContext) -> Cx {
        Self::linked(context, false)
    }

    /// A fresh root `Cx` that nothing of `context` cancels or stops, for a call that must
    /// reach its stage after the request is killed: the leg that releases what a worker holds
    /// for the request, such as the prefill router's decode leg after a KV handoff. The stage
    /// sees the request's stop and kill through `context`, which the request carries. The
    /// Dynamo id is in its metadata under [`REQUEST_ID`].
    ///
    /// It is not linked to `context`: the call's own attempt ends when its response stream
    /// ends or is dropped.
    pub fn cx_for_cleanup(context: &dyn AsyncEngineContext) -> Cx {
        let mut cx = Cx::new();
        cx.insert_meta(REQUEST_ID, context.id());
        cx
    }

    fn linked(context: &dyn AsyncEngineContext, cancel_on_kill: bool) -> Cx {
        let mut cx = Cx::new();
        cx.insert_meta(REQUEST_ID, context.id());
        context.link_child(Arc::new(CxLink {
            cx: cx.clone(),
            cancel_on_kill,
            children: Mutex::default(),
        }));
        // `link_child` does not replay what already happened to `context`.
        // Checked after linking, so a signal that lands in between reaches
        // the `Cx` one way or the other; both are idempotent. A signal under
        // way is seen too: Dynamo's controller publishes its state before it
        // tells its children (`tests/races.rs`).
        if context.is_killed() {
            if cancel_on_kill {
                cx.cancel();
            }
        } else if context.is_stopped() {
            cx.request_stop();
        }
        cx
    }

    /// A Dynamo request context for `data` that follows `cx`: a stop of `cx`
    /// stops it, and a cancel kills it. Use it where a gdl request starts and
    /// must reach a Dynamo engine.
    ///
    /// Its id is the Dynamo id in `cx`'s metadata, else `cx`'s own id, and its
    /// metadata is `cx`'s. A task forwards the signals until the context is
    /// dropped, so call it inside a Tokio runtime.
    pub fn context<T: Data>(cx: &Cx, data: T) -> Context<T> {
        let id = match cx.meta().get(REQUEST_ID) {
            Some(id) => id.clone(),
            None => cx.id().to_string(),
        };
        let request = Context::with_id_and_metadata(data, id, cx.meta().clone());
        let controller: Arc<dyn AsyncEngineContext> = request.context();
        let (released, on_release) = oneshot::channel();
        // The controller drops this guard when it is dropped, which ends the
        // task: the task holds the controller only weakly.
        controller.retain(Arc::new(released));
        let latch = Arc::new(KillLatch::default());
        controller.link_child(latch.clone());
        tokio::spawn(forward(
            cx.clone(),
            Arc::downgrade(&controller),
            latch,
            on_release,
        ));
        request
    }
}

/// Forward `cx`'s stop and cancel to `target` until `released` resolves.
///
/// A kill can arrive from anywhere while the stop is being sent. Dynamo's
/// controller keeps a kill once set, so the stop cannot replace it
/// (`tests/races.rs`). `killed`, a child of `target`, still sends a kill it
/// saw again after the stop; Dynamo's controller keeps its state and tells
/// its children again.
async fn forward(
    cx: Cx,
    target: Weak<dyn AsyncEngineContext>,
    killed: Arc<KillLatch>,
    mut released: oneshot::Receiver<()>,
) {
    tokio::select! {
        biased;
        _ = &mut released => return,
        () = cx.stopping() => {}
    }
    if !cx.is_cancelled() {
        let Some(context) = target.upgrade() else {
            return;
        };
        if !context.is_stopped() {
            context.stop_generating();
            if killed.0.load(Ordering::Acquire) {
                context.kill();
            }
        }
        drop(context);
        tokio::select! {
            biased;
            _ = &mut released => return,
            () = cx.cancelled() => {}
        }
    }
    if let Some(context) = target.upgrade() {
        context.kill();
    }
}

/// Whether a Dynamo controller has been killed, as a child it tells of the
/// kill. A controller calls only a child's stop and kill, so the waits never
/// resolve.
#[derive(Debug, Default)]
struct KillLatch(AtomicBool);

#[async_trait]
impl AsyncEngineContext for KillLatch {
    fn id(&self) -> &str {
        ""
    }

    fn is_stopped(&self) -> bool {
        self.is_killed()
    }

    fn is_killed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    async fn stopped(&self) {
        std::future::pending().await
    }

    async fn killed(&self) {
        std::future::pending().await
    }

    fn stop_generating(&self) {}

    fn stop(&self) {}

    fn kill(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn link_child(&self, _child: Arc<dyn AsyncEngineContext>) {}
}

/// A `Cx` as a child of a Dynamo controller, which calls it when the request
/// is stopped or killed.
struct CxLink {
    cx: Cx,
    /// Whether a kill cancels `cx` (see [`CxBridge`]).
    cancel_on_kill: bool,
    children: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
}

impl CxLink {
    fn children(&self) -> Vec<Arc<dyn AsyncEngineContext>> {
        self.children
            .lock()
            .expect("child contexts lock")
            .iter()
            .cloned()
            .collect()
    }
}

impl fmt::Debug for CxLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CxLink").field("cx", &self.cx).finish()
    }
}

#[async_trait]
impl AsyncEngineContext for CxLink {
    fn id(&self) -> &str {
        self.cx
            .meta()
            .get(REQUEST_ID)
            .map(String::as_str)
            .unwrap_or_default()
    }

    fn is_stopped(&self) -> bool {
        self.cx.is_stopping()
    }

    fn is_killed(&self) -> bool {
        self.cx.is_cancelled()
    }

    async fn stopped(&self) {
        self.cx.stopping().await;
    }

    async fn killed(&self) {
        self.cx.cancelled().await;
    }

    fn stop_generating(&self) {
        for child in self.children() {
            child.stop_generating();
        }
        self.cx.request_stop();
    }

    fn stop(&self) {
        for child in self.children() {
            child.stop();
        }
        self.cx.request_stop();
    }

    fn kill(&self) {
        for child in self.children() {
            child.kill();
        }
        if self.cancel_on_kill {
            self.cx.cancel();
        }
    }

    fn link_child(&self, child: Arc<dyn AsyncEngineContext>) {
        self.children
            .lock()
            .expect("child contexts lock")
            .push(child);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn request() -> Context<()> {
        Context::with_id_and_metadata((), "req-1".to_string(), Default::default())
    }

    async fn eventually(what: &str, check: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !check() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    #[test]
    fn the_cx_carries_the_dynamo_id() {
        let request = request();
        let cx = CxBridge::cx(request.controller());
        assert_eq!(cx.meta().get(REQUEST_ID).map(String::as_str), Some("req-1"));
    }

    #[test]
    fn a_stop_stops_the_cx_and_a_kill_cancels_it() {
        let request = request();
        let cx = CxBridge::cx(request.controller());
        assert!(!cx.is_stopping());

        request.context().stop_generating();
        assert!(cx.is_stopping());
        assert!(!cx.is_cancelled());

        request.context().kill();
        assert!(cx.is_cancelled());
    }

    #[test]
    fn a_kill_after_a_stop_stays_a_kill() {
        let request = request();
        let cx = CxBridge::cx(request.controller());
        request.context().stop_generating();
        request.context().kill();
        assert!(cx.is_cancelled());
        assert!(request.context().is_killed());

        // A later stop leaves the kill in place, in the controller and the `Cx`.
        request.context().stop_generating();
        assert!(cx.is_cancelled());
        assert!(request.context().is_killed());
    }

    #[test]
    fn a_context_killed_before_bridging_gives_a_cancelled_cx() {
        let request = request();
        request.context().kill();
        assert!(CxBridge::cx(request.controller()).is_cancelled());

        let request = super::tests::request();
        request.context().stop_generating();
        let cx = CxBridge::cx(request.controller());
        assert!(cx.is_stopping());
        assert!(!cx.is_cancelled());
    }

    #[test]
    fn a_kill_reaches_a_cx_bridged_from_a_child() {
        // Migration links each attempt's context as a child of the request's.
        let parent = request();
        let attempt = request();
        parent.context().link_child(attempt.context());
        let cx = CxBridge::cx(attempt.controller());
        parent.context().kill();
        assert!(cx.is_cancelled());
    }

    #[tokio::test]
    async fn a_context_from_a_cx_takes_its_id_and_metadata() {
        let mut cx = Cx::new();
        cx.insert_meta("tenant", "a");
        let request = CxBridge::context(&cx, 7u32);
        assert_eq!(request.id(), cx.id().to_string());
        assert_eq!(
            request.metadata().get("tenant").map(String::as_str),
            Some("a")
        );
        assert_eq!(*request.content(), 7);

        cx.insert_meta(REQUEST_ID, "req-9");
        assert_eq!(CxBridge::context(&cx, ()).id(), "req-9");
    }

    #[tokio::test]
    async fn a_cx_stop_stops_the_context_and_a_cancel_kills_it() {
        let cx = Cx::new();
        let request = CxBridge::context(&cx, ());
        let context = request.context();

        cx.request_stop();
        eventually("the stop", || context.is_stopped()).await;
        assert!(!context.is_killed());

        cx.cancel();
        eventually("the kill", || context.is_killed()).await;
    }

    #[tokio::test]
    async fn a_cx_cancel_after_a_stop_stays_a_kill() {
        let cx = Cx::new();
        let request = CxBridge::context(&cx, ());
        let context = request.context();
        cx.cancel();
        eventually("the kill", || context.is_killed()).await;
        cx.request_stop();
        // Give a wrongly forwarded stop the time to land.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(context.is_killed());
    }

    #[tokio::test]
    async fn dropping_the_context_ends_the_forwarding_task() {
        let metrics = tokio::runtime::Handle::current().metrics();
        let before = metrics.num_alive_tasks();
        let cx = Cx::new();
        let request = CxBridge::context(&cx, ());
        assert_eq!(metrics.num_alive_tasks(), before + 1);
        let weak = Arc::downgrade(&request.context());
        drop(request);
        // The task holds the controller only weakly, so it is gone.
        assert!(weak.upgrade().is_none());
        eventually("the task to end", || metrics.num_alive_tasks() == before).await;
    }
}
