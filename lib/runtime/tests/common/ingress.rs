// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A stand-in frontend for driving `Ingress` through `handle_payload`.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::Notify;
use tokio_util::codec::FramedRead;

use dynamo_runtime::pipeline::network::{
    ConnectionInfo, ControlMessage, StreamType,
    codec::{TwoPartCodec, TwoPartMessage},
    tcp::TcpStreamConnectionInfo,
};

/// A frontend's response endpoint, running on its own thread.
pub struct ResponsePeer {
    /// Where a request's response stream connects.
    pub address: String,
    closed: Arc<Closed>,
}

/// Connections the peer has closed.
#[derive(Default)]
struct Closed {
    count: AtomicUsize,
    notify: Notify,
}

impl ResponsePeer {
    /// Waits until the peer has closed `n` connections in all. It reads no
    /// clock, so its cost under callgrind does not follow wall-clock time.
    pub async fn closed(&self, n: usize) {
        loop {
            // Created before the check, so a close between the two still wakes it.
            let notified = self.closed.notify.notified();
            if self.closed.count.load(Ordering::Acquire) >= n {
                return;
            }
            notified.await;
        }
    }
}

/// Starts a frontend's response endpoint: it accepts each worker connection,
/// reads its frames and closes it after the closing sentinel, as the frontend
/// does. It runs on its own thread, so a test that counts allocations on its
/// own thread does not count the peer's, and that thread turns callgrind's
/// counting off.
pub fn start_response_peer() -> ResponsePeer {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    listener.set_nonblocking(true).unwrap();
    let closed = Arc::new(Closed::default());
    let peer_closed = closed.clone();
    std::thread::spawn(move || {
        // A frontend's work, not the worker's: keep it out of callgrind's count.
        super::callgrind::stop_counting_thread();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let closed = peer_closed.clone();
                tokio::spawn(async move {
                    let sentinel = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
                    let mut frames = FramedRead::new(socket, TwoPartCodec::default());
                    while let Some(Ok(frame)) = frames.next().await {
                        if let (Some(header), None) = frame.optional_parts()
                            && header.as_ref() == sentinel.as_slice()
                        {
                            break;
                        }
                    }
                    drop(frames);
                    closed.count.fetch_add(1, Ordering::Release);
                    closed.notify.notify_waiters();
                });
            }
        });
    });
    ResponsePeer { address, closed }
}

/// One single-in, many-out request-plane frame, as a frontend sends it. Its
/// response stream connects to `address`.
pub fn request_payload(address: &str, id: &str, body: &impl Serialize) -> Bytes {
    let connection: ConnectionInfo = TcpStreamConnectionInfo {
        address: address.to_string(),
        subject: id.to_string(),
        context: id.to_string(),
        stream_type: StreamType::Response,
    }
    .into();
    let header = serde_json::to_vec(&serde_json::json!({
        "id": id,
        "request_type": "single_in",
        "response_type": "many_out",
        "connection_info": connection,
    }))
    .unwrap();
    let body = serde_json::to_vec(body).unwrap();
    TwoPartCodec::default()
        .encode_message(TwoPartMessage::from_parts(header.into(), body.into()))
        .unwrap()
}
