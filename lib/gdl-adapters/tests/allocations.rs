// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocations per request through the adapters and native gdl operators
//! that read their downstream as Dynamo's native operators do
//! (`dynamo_llm::operator`), over an engine alone.
//!
//! Run with `--nocapture` to see the table. The counts are per request after
//! a warm-up, on a current-thread runtime, so every allocation is the
//! request's. The test fails only if the adapters allocate per stream item;
//! the per-request cost is reported, not judged.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_gdl_adapters::{AnnotatedItems, CxBridge, EngineSink, PipelineEngine};
use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use gdl::{Cx, Hop, Items, OperatorStage, SinkStage, Source};

struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call forwards to the system allocator with the caller's
// arguments; the counters do not touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

type Out = ManyOut<Annotated<u32>>;
type Engine = Arc<dyn AsyncEngine<SingleIn<u32>, Out, Error>>;

const WARM_UP: u64 = 100;
const REQUESTS: u64 = 2_000;
const MAX_STAGES: usize = 5;

/// Streams `items` copies of its request.
struct Repeat(usize);

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for Repeat {
    async fn generate(&self, request: SingleIn<u32>) -> Result<Out, Error> {
        let n = *request.content();
        let context = request.context();
        let items = stream::repeat_with(move || Annotated::from_data(n)).take(self.0);
        Ok(ResponseStream::new(Box::pin(items), context))
    }
}

/// The identity as a native gdl operator, shaped as Backend's and the
/// preprocessor's are: it keeps the request's Dynamo context, reads its
/// downstream through `AnnotatedItems`, and boxes its response once.
struct NativeIdentity;

impl gdl::Operator for NativeIdentity {
    type UpReq = SingleIn<u32>;
    type UpItem = Annotated<u32>;
    type DownReq = SingleIn<u32>;
    type DownItem = Annotated<u32>;

    async fn generate<N>(
        &self,
        cx: Cx,
        request: SingleIn<u32>,
        next: N,
    ) -> gdl::Result<Items<Annotated<u32>>>
    where
        N: gdl::Next<SingleIn<u32>, Annotated<u32>>,
    {
        let context = request.context();
        let root = cx.clone();
        let items = next.generate(cx, request).await?;
        Ok(gdl::items(
            AnnotatedItems::new(items, root, context).map(Ok),
        ))
    }
}

/// `stages` native gdl operators above an adapted engine.
fn native(stages: usize, engine: Engine) -> Engine {
    fn wrap<N: gdl::Next<SingleIn<u32>, Annotated<u32>>>(next: N) -> Engine {
        Arc::new(PipelineEngine::<_, u32, u32>::new(next))
    }
    fn stage<H: gdl::Next<SingleIn<u32>, Annotated<u32>>>(
        hop: H,
    ) -> Hop<(), OperatorStage<NativeIdentity, H>, ()> {
        Hop::new((), OperatorStage::new(Arc::new(NativeIdentity), hop), ())
    }
    let sink = Hop::new((), SinkStage::new(Arc::new(EngineSink::new(engine))), ());
    match stages {
        0 => wrap(Source::new(sink)),
        1 => wrap(Source::new(stage(sink))),
        2 => wrap(Source::new(stage(stage(sink)))),
        3 => wrap(Source::new(stage(stage(stage(sink))))),
        4 => wrap(Source::new(stage(stage(stage(stage(sink)))))),
        5 => wrap(Source::new(stage(stage(stage(stage(stage(sink))))))),
        _ => unreachable!("up to {MAX_STAGES} stages"),
    }
}

#[derive(Clone, Copy, Debug)]
struct Cost {
    allocations: f64,
    reallocations: f64,
    bytes: f64,
}

fn measure(runtime: &tokio::runtime::Runtime, engine: &Engine) -> Cost {
    let run = |count: u64| {
        runtime.block_on(async {
            for n in 0..count {
                let request = Context::with_id_and_metadata(
                    n as u32,
                    "request".to_string(),
                    Default::default(),
                );
                let mut stream = engine.generate(request).await.unwrap();
                while stream.next().await.is_some() {}
            }
        })
    };
    run(WARM_UP);
    let (allocations, reallocations, bytes) = (
        ALLOCATIONS.load(Ordering::Relaxed),
        REALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    run(REQUESTS);
    let per = |after: u64, before: u64| (after - before) as f64 / REQUESTS as f64;
    Cost {
        allocations: per(ALLOCATIONS.load(Ordering::Relaxed), allocations),
        reallocations: per(REALLOCATIONS.load(Ordering::Relaxed), reallocations),
        bytes: per(BYTES.load(Ordering::Relaxed), bytes),
    }
}

#[test]
fn allocations_per_request() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("items stages | engine | native: allocs reallocs bytes");
    let mut native_by_items = Vec::new();
    for items in [1, 16] {
        let engine: Engine = Arc::new(Repeat(items));
        let alone = measure(&runtime, &engine);
        let mut native_row = Vec::new();
        for stages in 0..=MAX_STAGES {
            let natively = measure(&runtime, &native(stages, engine.clone()));
            println!(
                "{items:>5} {stages:>6} | {:>6.1} | {:>6.1} {:>4.1} {:>7.0}",
                alone.allocations, natively.allocations, natively.reallocations, natively.bytes,
            );
            native_row.push(natively);
        }
        native_by_items.push(native_row);
    }

    // The bridge alone: one `Cx` derived from a fresh request context,
    // without the context's own allocations.
    let mut derived = 0;
    for n in 0..REQUESTS {
        let request =
            Context::with_id_and_metadata(n as u32, "request".to_string(), Default::default());
        let start = ALLOCATIONS.load(Ordering::Relaxed);
        let cx = CxBridge::cx(request.controller());
        derived += ALLOCATIONS.load(Ordering::Relaxed) - start;
        drop(cx);
    }
    println!(
        "CxBridge::cx: {:.1} allocations",
        derived as f64 / REQUESTS as f64
    );

    // Nothing in the adapters or a native stage allocates per item: 16 items cost what 1 does
    // on top of what the engine's own items cost, which `Repeat` keeps at none.
    for (one, sixteen) in native_by_items[0].iter().zip(&native_by_items[1]) {
        assert!(
            (one.allocations - sixteen.allocations).abs() < 0.5,
            "per-item allocation: {one:?} against {sixteen:?}"
        );
    }
}
