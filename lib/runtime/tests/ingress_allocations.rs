// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations of one worker request through `Ingress::for_engine`.
//!
//! The request goes the whole worker path: `handle_payload` decodes the
//! request-plane frame, opens a TCP response stream back to a peer, calls the
//! engine and publishes its stream. The test counts every allocation made on
//! the worker's thread from `handle_payload` until every task the request
//! spawned has ended, so late work by the response-stream tasks is not counted
//! against the next request.
//!
//! The peer that stands in for the frontend runs on another thread and is not
//! counted. The test fails when a count rises above its ceiling.
//!
//! Run: `cargo test -p dynamo-runtime --test ingress_allocations`. For
//! instructions and `memcpy` calls per request, `lib/llm/benches/callgrind.sh`
//! runs this binary under callgrind as
//! `ingress_allocations row ingress <items> <requests>`. Callgrind counts only
//! the measured requests, between two client requests, and not the system
//! allocator or the peer's thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use anyhow::{Error, Result};
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{
        ManyOut, ResponseStream, SingleIn,
        network::{Ingress, PushWorkHandler},
    },
    protocols::annotated::Annotated,
};

// Not `mod common`: this binary has no test harness, so the `#[test]` modules in
// the other helpers would leave their imports unused.
#[path = "common/callgrind.rs"]
mod callgrind;
#[path = "common/ingress.rs"]
mod ingress;
use ingress::{ResponsePeer, request_payload, start_response_peer};

/// Allocations per request for a one-item and a 16-item response stream. These
/// are ceilings, set to the counts measured on this code; lower them when the
/// worker path gets cheaper.
const MAX_ALLOCATIONS_ONE_ITEM: u64 = 55;
const MAX_ALLOCATIONS_SIXTEEN_ITEMS: u64 = 115;

const WARMUP_REQUESTS: usize = 20;
const MEASURED_REQUESTS: usize = 200;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

fn record(size: usize) {
    // `try_with` because the allocator also runs while thread locals are torn down.
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
            let _ = BYTES.try_with(|n| n.set(n.get() + size as u64));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        let _pause = callgrind::pause();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        let _pause = callgrind::pause();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        let _pause = callgrind::pause();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _pause = callgrind::pause();
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocations and bytes counted on this thread while `COUNTING` was set.
fn take_counts() -> (u64, u64) {
    (ALLOCATIONS.replace(0), BYTES.replace(0))
}

/// Returns its request's text `items` times.
struct EchoEngine {
    items: usize,
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for EchoEngine {
    async fn generate(&self, request: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (text, context) = request.into_parts();
        let items = std::iter::repeat_n(text, self.items).map(Annotated::from_data);
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            context.context(),
        ))
    }
}

/// Runs the `n`th request (from 1) and returns its allocations and bytes,
/// including the response-stream tasks it spawned.
async fn measure_one(
    ingress: &dyn PushWorkHandler,
    peer: &ResponsePeer,
    n: usize,
    payload: Bytes,
    id: String,
) -> (u64, u64) {
    let metrics = tokio::runtime::Handle::current().metrics();
    let idle_tasks = metrics.num_alive_tasks();
    take_counts();
    COUNTING.set(true);
    ingress
        .handle_payload(payload, Some(id))
        .await
        .expect("worker request failed");
    // The reader, writer and monitor tasks end once the peer closes its side.
    // Wait for the close, then for the tasks to see it. Neither wait reads a
    // clock: under callgrind, a 1 ms sleep loop here made the count follow the
    // host's load (1.5% between runs of one binary).
    peer.closed(n).await;
    let mut yields = 0u32;
    while metrics.num_alive_tasks() > idle_tasks {
        yields += 1;
        assert!(yields < 1_000_000, "response-stream tasks did not end");
        tokio::task::yield_now().await;
    }
    COUNTING.set(false);
    take_counts()
}

/// Per-request allocations and bytes of `requests` requests, after warm-up.
/// Every measured request must cost the same, or the count is not a property
/// of the request. Callgrind counts the measured requests only.
async fn allocations_per_request(items: usize, requests: usize) -> (u64, u64) {
    let peer = start_response_peer();
    let ingress = Ingress::for_engine(Arc::new(EchoEngine { items })).unwrap();
    let mut counts = Vec::with_capacity(requests);
    for n in 0..WARMUP_REQUESTS + requests {
        if n == WARMUP_REQUESTS {
            callgrind::dump_stats_at(c"setup and warm-up");
        }
        let id = format!("request-{n:06}");
        let payload = request_payload(&peer.address, &id, &"hello");
        let count = measure_one(ingress.as_ref(), &peer, n + 1, payload, id).await;
        if n >= WARMUP_REQUESTS {
            counts.push(count);
        }
    }
    callgrind::dump_stats_at(c"measured requests");
    let allocations: Vec<u64> = counts.iter().map(|(n, _)| *n).collect();
    let min = *allocations.iter().min().unwrap();
    let max = *allocations.iter().max().unwrap();
    let bytes = counts.iter().map(|(_, b)| *b).sum::<u64>() / counts.len() as u64;
    eprintln!("{items} item(s): {min}..={max} allocations, {bytes} bytes per request");
    assert_eq!(min, max, "allocations per request vary: {allocations:?}");
    (min, bytes)
}

fn main() {
    let mut builder = tokio::runtime::Builder::new_current_thread();
    builder.enable_all();
    let runtime = callgrind::uncount_parked(&mut builder).build().unwrap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("row") {
        assert_eq!(
            args.get(1).map(String::as_str),
            Some("ingress"),
            "unknown row"
        );
        let items: usize = args.get(2).expect("items").parse().expect("items");
        let requests: usize = args.get(3).expect("requests").parse().expect("requests");
        let (allocs, bytes) = runtime.block_on(allocations_per_request(items, requests));
        println!("ingress items={items} allocs={allocs} bytes={bytes}");
        return;
    }
    let (one, _) = runtime.block_on(allocations_per_request(1, MEASURED_REQUESTS));
    let (sixteen, _) = runtime.block_on(allocations_per_request(16, MEASURED_REQUESTS));
    assert!(
        one <= MAX_ALLOCATIONS_ONE_ITEM,
        "one-item worker request: {one} allocations, ceiling {MAX_ALLOCATIONS_ONE_ITEM}"
    );
    assert!(
        sixteen <= MAX_ALLOCATIONS_SIXTEEN_ITEMS,
        "16-item worker request: {sixteen} allocations, ceiling {MAX_ALLOCATIONS_SIXTEEN_ITEMS}"
    );
    println!("worker_request_allocations: ok");
}
