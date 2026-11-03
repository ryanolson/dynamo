// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use dynamo_gdl_adapters::CxBridge;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context, async_trait};

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
fn a_kill_in_progress_reaches_a_new_bridge() {
    signal_during_registration(true);
}

#[test]
fn a_stop_in_progress_reaches_a_new_bridge() {
    signal_during_registration(false);
}
