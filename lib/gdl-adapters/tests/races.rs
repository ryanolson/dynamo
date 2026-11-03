// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bridge made while a stop or a kill is under way.
//!
//! Dynamo's controller tells its children of a stop or a kill before it
//! publishes the new state. A `Cx` linked in between is not told, and the
//! state it reads after linking is still live, so it misses the signal. The
//! same window exists for every `link_child` followed by a state check,
//! Migration's included. Closing it means publishing the state before telling
//! the children, in `dynamo-runtime`.
//!
//! And a stop forwarded from a gdl `Cx` while a kill is under way.

use std::sync::{Arc, Mutex, Weak, mpsc};
use std::time::Duration;

use dynamo_gdl_adapters::CxBridge;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context, async_trait};
use gdl::Cx;
use tokio::sync::oneshot;

#[derive(Debug)]
struct PausedChild {
    entered: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl PausedChild {
    fn pause(&self) {
        self.entered.send(()).unwrap();
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
    }
}

#[async_trait]
impl AsyncEngineContext for PausedChild {
    fn id(&self) -> &str {
        "paused-child"
    }

    fn is_stopped(&self) -> bool {
        false
    }

    fn is_killed(&self) -> bool {
        false
    }

    async fn stopped(&self) {
        std::future::pending().await
    }

    async fn killed(&self) {
        std::future::pending().await
    }

    fn stop_generating(&self) {
        self.pause();
    }

    fn stop(&self) {
        self.pause();
    }

    fn kill(&self) {
        self.pause();
    }

    fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
}

fn signal_during_registration(kill: bool) {
    let request = Context::new(());
    let context = request.context();
    let (entered, on_enter) = mpsc::channel();
    let (release, on_release) = mpsc::channel();
    context.link_child(Arc::new(PausedChild {
        entered,
        release: Mutex::new(on_release),
    }));

    let sender = context.clone();
    let signal = std::thread::spawn(move || {
        if kill {
            sender.kill();
        } else {
            sender.stop_generating();
        }
    });
    on_enter.recv_timeout(Duration::from_secs(5)).unwrap();

    // The signal has taken its child snapshot, but has not published its state.
    assert!(!context.is_stopped());
    let cx = CxBridge::cx(request.controller());
    release.send(()).unwrap();
    signal.join().unwrap();

    assert!(context.is_stopped());
    assert_eq!(context.is_killed(), kill);
    if kill {
        assert!(cx.is_cancelled(), "the bridge missed the completed kill");
    } else {
        assert!(cx.is_stopping(), "the bridge missed the completed stop");
    }
}

#[test]
#[ignore = "bridge registration race: the controller tells its children before it publishes the state, so a Cx linked in between misses the signal; the fix is in dynamo-runtime's Controller"]
fn a_kill_in_progress_reaches_a_new_bridge() {
    signal_during_registration(true);
}

#[test]
#[ignore = "bridge registration race: the controller tells its children before it publishes the state, so a Cx linked in between misses the signal; the fix is in dynamo-runtime's Controller"]
fn a_stop_in_progress_reaches_a_new_bridge() {
    signal_during_registration(false);
}

#[derive(Debug)]
struct KillDuringStop {
    target: Weak<dyn AsyncEngineContext>,
    finished: Mutex<Option<oneshot::Sender<()>>>,
}

#[async_trait]
impl AsyncEngineContext for KillDuringStop {
    fn id(&self) -> &str {
        "kill-during-stop"
    }

    fn is_stopped(&self) -> bool {
        false
    }

    fn is_killed(&self) -> bool {
        false
    }

    async fn stopped(&self) {
        std::future::pending().await
    }

    async fn killed(&self) {
        std::future::pending().await
    }

    fn stop_generating(&self) {
        let target = self.target.upgrade().unwrap();
        // Complete a concurrent kill before the parent's stop publishes its state.
        std::thread::spawn(move || {
            target.kill();
            assert!(target.is_killed());
        })
        .join()
        .unwrap();
        self.finished
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
    }

    fn stop(&self) {
        self.stop_generating();
    }

    fn kill(&self) {}

    fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
}

#[tokio::test]
async fn forwarding_a_stop_does_not_replace_a_concurrent_kill() {
    let cx = Cx::new();
    let request = CxBridge::context(&cx, ());
    let context = request.context();
    let (finished, on_finish) = oneshot::channel();
    context.link_child(Arc::new(KillDuringStop {
        target: Arc::downgrade(&context),
        finished: Mutex::new(Some(finished)),
    }));

    cx.request_stop();
    tokio::time::timeout(Duration::from_secs(5), on_finish)
        .await
        .unwrap()
        .unwrap();
    assert!(cx.is_stopping());
    assert!(!cx.is_cancelled());
    assert!(context.is_stopped());
    assert!(context.is_killed(), "the forwarded stop replaced the kill");
}
