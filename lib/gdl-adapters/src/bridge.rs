// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stop and kill between a Dynamo request context and a gdl [`Cx`].

use std::fmt;
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
        let mut cx = Cx::new();
        cx.insert_meta(REQUEST_ID, context.id());
        context.link_child(Arc::new(CxLink {
            cx: cx.clone(),
            children: Mutex::default(),
        }));
        // `link_child` does not replay what already happened to `context`.
        // Checked after linking, so a signal that lands in between reaches
        // the `Cx` one way or the other; both are idempotent.
        if context.is_killed() {
            cx.cancel();
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
        tokio::spawn(forward(cx.clone(), Arc::downgrade(&controller), on_release));
        request
    }
}

/// Forward `cx`'s stop and cancel to `target` until `released` resolves. A
/// stop is never sent after a kill: Dynamo's controller would replace the
/// kill with it.
async fn forward(
    cx: Cx,
    target: Weak<dyn AsyncEngineContext>,
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

/// A `Cx` as a child of a Dynamo controller, which calls it when the request
/// is stopped or killed.
struct CxLink {
    cx: Cx,
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
        self.cx.cancel();
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

        // Dynamo's controller lets a later stop replace its kill; the `Cx`
        // keeps it.
        request.context().stop_generating();
        assert!(cx.is_cancelled());
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
