// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use dynamo_gdl_adapters::CxBridge;
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, async_trait};
use gdl::Cx;
use tokio::sync::oneshot;

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
