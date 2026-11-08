// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocations per request through the KV router's `RoutingHost`, as a Dynamo engine and as the
//! sink of a gdl local pipeline.
//!
//! The rows of `tests/allocations.rs` put a test engine where the frontend has its
//! `RoutingHost`, so no row there sees the host's own boundary. This test does: a process-local
//! runtime with one registered worker, round-robin routing, and a dispatch that stands in for
//! the request plane and streams N items. It fails when a request makes more allocations than
//! its record, and asks for the record to be lowered when it makes fewer.
//!
//! Run: `cargo test -p dynamo-llm --test routing_allocations`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use dynamo_gdl_adapters::{Boxed, PipelineEngine};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_llm::discovery::LoadThresholdHandle;
use dynamo_llm::kv_router::{
    KvRouter, LlmDispatch, RouterLoadSource, RoutingHost, RoutingLoadContext, SelectionPolicySource,
};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::types::Annotated;
use dynamo_runtime::distributed::{DistributedConfig, DistributedRuntime};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::network::egress::addressed_router::{
    AddressedRequest, StreamingDispatch,
};
use dynamo_runtime::pipeline::network::egress::route_span::wrap_route_span;
use dynamo_runtime::pipeline::{Context, Error, ManyIn, ManyOut, RouterMode, SingleIn};
use dynamo_runtime::{Runtime, component::Instance};
use futures::{StreamExt, stream};

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Recorded allocations of the typical request: the host as a Dynamo engine (`engine`), and as
/// the sink of a one-hop gdl pipeline behind `PipelineEngine` (`gdl`), at 1 and 16 items.
/// `gdl` includes the box `PipelineEngine` puts the request in, as the frontend's gdl pipelines
/// carry it (`dynamo_gdl_adapters::Boxed`): one allocation, accepted for the copies it saves.
/// `gdl` also includes the request id `PipelineEngine` holds while it dispatches, to refuse a
/// second in-flight request with that id: one allocation. The response leaves in one box, with
/// no `ResponseStream` around it: one allocation fewer.
const RECORDS: &[(&str, usize, u64)] = &[
    ("engine", 1, 26),
    ("engine", 16, 41),
    ("gdl", 1, 41),
    ("gdl", 16, 56),
];

const WARMUP: usize = 20;
const REQUESTS: usize = 100;

/// The request plane, as the host sees it: streams `items` engine outputs for any request.
struct ItemsDispatch {
    items: usize,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for ItemsDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        let items = (0..self.items as u32).map(|token| {
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![token],
                ..Default::default()
            })
        });
        Ok(ResponseStream::new(Box::pin(stream::iter(items)), context))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("the routing host dispatches unary requests")
    }
}

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("TinyLlama_v1.1".to_string())
        .token_ids(vec![1, 2, 3, 4])
        .stop_conditions(StopConditions {
            max_tokens: Some(64),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![2])
        .annotations(vec![])
        .build()
        .expect("preprocessed request")
}

type Engine =
    Arc<dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>>;

/// Allocations of the typical request through `engine`: the most common count over
/// `REQUESTS` requests, after `WARMUP` of warm-up.
async fn typical(engine: &Engine, items: usize, route_span: Option<&tracing::Span>) -> u64 {
    let mut counts = HashMap::<u64, usize>::new();
    for i in 0..WARMUP + REQUESTS {
        let request = Context::new(request());
        ALLOCS.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        let stream = engine.generate(request).await.expect("routed");
        let stream = match route_span {
            Some(span) => wrap_route_span(stream, span.clone()),
            None => stream,
        };
        let seen = stream.count().await;
        COUNTING.store(false, Ordering::Relaxed);
        assert_eq!(seen, items, "every item reaches the caller");
        if i >= WARMUP {
            *counts.entry(ALLOCS.load(Ordering::Relaxed)).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(allocs, n)| (*n, std::cmp::Reverse(*allocs)))
        .map(|(allocs, _)| allocs)
        .unwrap()
}

async fn host(runtime: &Runtime, items: usize, mode: RouterMode) -> Arc<RoutingHost> {
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .expect("process-local runtime");
    let endpoint = distributed
        .namespace(format!("routing-allocations-{items}"))
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let workers = client.wait_for_instances().await.unwrap();
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Aggregated,
        LoadThresholdHandle::new(Default::default()),
        &distributed.child_token(),
        None,
    )
    .await
    .expect("load context");
    let dispatch: LlmDispatch = Arc::new(ItemsDispatch { items });
    let mut builder = RoutingHost::builder(client.clone(), mode)
        .load_context(load_context)
        .dispatch(dispatch);
    if matches!(mode, RouterMode::KV) {
        let (_tx, workers) = tokio::sync::watch::channel(
            workers
                .into_iter()
                .map(|worker| (worker.id(), ModelRuntimeConfig::default()))
                .collect(),
        );
        let chooser = KvRouter::new(
            endpoint,
            client,
            workers,
            None,
            16,
            SelectionPolicySource::Registry,
            Some(KvRouterConfig {
                skip_initial_worker_wait: true,
                use_kv_events: false,
                router_track_active_blocks: false,
                ..Default::default()
            }),
            None,
            "decode",
            None,
            false,
            None,
            None,
        )
        .await
        .expect("KV router");
        builder = builder.kv_router(Arc::new(chooser));
    }
    Arc::new(builder.build().await.expect("routing host"))
}

/// Enable one span without allocating in the subscriber itself.
struct RouteSpanSubscriber(&'static str);

impl tracing::Subscriber for RouteSpanSubscriber {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.is_span() && metadata.name() == self.0
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, _: &tracing::Event<'_>) {}

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

async fn traced_kv_route_keeps_the_span_wrappers_allocation_budget(runtime: &Runtime) {
    let engine: Engine = host(runtime, 1, RouterMode::KV).await;
    let reference = {
        let _subscriber =
            tracing::subscriber::set_default(RouteSpanSubscriber("allocation_reference.route"));
        let span = tracing::info_span!("allocation_reference.route");
        assert!(!span.is_disabled());
        // Wrap the monitored response directly, without adding another stream boundary.
        typical(&engine, 1, Some(&span)).await
    };
    let traced = {
        let _subscriber =
            tracing::subscriber::set_default(RouteSpanSubscriber("kv_router.route_request"));
        typical(&engine, 1, None).await
    };
    println!("KV route: reference={reference}, traced={traced}");
    assert!(
        traced <= reference,
        "the traced KV route added {} allocations beyond wrap_route_span's existing boxes",
        traced.saturating_sub(reference),
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let failed = runtime.block_on(async {
        let dynamo_runtime = Runtime::from_current().unwrap();
        let mut failed = false;
        for items in [1, 16] {
            let host = host(&dynamo_runtime, items, RouterMode::RoundRobin).await;
            let engine: Engine = host.clone();
            let sink = host.clone();
            let gdl: Engine = Arc::new(PipelineEngine::<
                _,
                PreprocessedRequest,
                LLMEngineOutput,
                Boxed<PreprocessedRequest>,
            >::new(
                gdl::Source::new(gdl::Hop::new((), gdl::SinkStage::new(sink), ())),
            ));
            for (name, engine) in [("engine", &engine), ("gdl", &gdl)] {
                let allocs = typical(engine, items, None).await;
                let record = RECORDS
                    .iter()
                    .find(|(n, i, _)| *n == name && *i == items)
                    .map(|(_, _, allocs)| *allocs)
                    .unwrap();
                println!("{name:6} items={items:<3} allocs={allocs} record={record}");
                if allocs > record {
                    eprintln!("{name} items={items}: {allocs} allocations, above the record {record}");
                    failed = true;
                } else if allocs < record {
                    eprintln!("{name} items={items}: {allocs} allocations, below the record {record}; lower the record");
                }
            }
        }
        traced_kv_route_keeps_the_span_wrappers_allocation_budget(&dynamo_runtime).await;
        failed
    });
    if failed {
        std::process::exit(1);
    }
}
