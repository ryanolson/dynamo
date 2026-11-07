// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations per request through the frontend pipelines.
//!
//! A change to how a pipeline is composed must not add allocations per request. This binary
//! builds each pipeline with its real operators, puts a test engine at the terminal, and checks
//! that allocations per request stay at or below the recorded numbers in [`BASELINE`].
//!
//! | Row | Pipeline | Terminal |
//! |---|---|---|
//! | `engine` | the test LLM engine, called directly | itself |
//! | `a-gdl` | `build_pipeline`: preprocessor, backend | test LLM engine |
//! | `b-gdl` | `PreprocessedRouting::build_pipeline`: preprocessor, migration, backend, encoder router, prefill router | test LLM engine, in place of `RoutingHost`, through `EngineSink` |
//! | `c-gdl` | `PreprocessedRouting::build_preprocessed_pipeline`: migration, encoder router, prefill router | test LLM engine, in place of `RoutingHost`, through `EngineSink` |
//! | `embed-engine` | the test embeddings engine, called directly | itself |
//! | `embeddings-gdl` | `build_embeddings_pipeline`: preprocessor (an embeddings response needs no backend) | test embeddings engine, in place of `PushRouter` |
//!
//! The rows are gdl local pipelines, the only local pipeline the frontend composes. Until the
//! switch, `a`, `b`, `c` and `embed` measured the same pipelines composed as Dynamo's `link`
//! chains, and the gdl rows kept their `-gdl` names so their records carry on.
//!
//! The second number of a row is stream items: 1 and 16. For the embeddings rows it is
//! inputs per request instead, because an embeddings stream carries one item for any number of
//! inputs.
//!
//! The terminal stands in for everything behind it: in `b-gdl` and `c-gdl`, `RoutingHost` (worker
//! selection, `PushRouter`, the request plane and the worker). The `engine` rows give the
//! stand-in's own cost, so a pipeline's cost is its row minus the `engine` row.
//!
//! Setup, chosen to match the frontend's defaults: router mode round-robin, migration limit 0,
//! prefill and encoder routers disabled (aggregated serving), the TinyLlama sample model, a
//! one-message chat request. Each window starts where the HTTP handler calls the engine: it
//! creates the request `Context`, calls `generate`, drains the stream and drops it. HTTP parsing,
//! the handler's request metrics and SSE encoding are outside it.
//!
//! Counting: a global allocator counts calls to `alloc`, `alloc_zeroed` and `realloc`, and the
//! bytes they ask for, on every thread while a window is open. Tokenization runs on tokio's
//! blocking pool, so the rows count all threads. The runtime is current-thread, so the
//! preprocessor renders the template inline where the multi-thread frontend uses
//! `block_in_place`. It has one blocking thread, so a window can wait for the blocking work a
//! request leaves. No tracing subscriber is installed. `main` removes every `DYN_` variable and
//! `TOKENIZERS_PARALLELISM`, and sets `RAYON_NUM_THREADS=4`, so neither the shell nor the host's
//! core count changes the counts.
//!
//! Measured on a 20-core aarch64 Linux host, Rust 1.99.0, 200 requests after 50 of warm-up,
//! for the typical request. Debug and release builds give the same numbers:
//!
//! | Row | Allocations, 1 item | Bytes, 1 item | Allocations, 16 items | Bytes, 16 items |
//! |---|---:|---:|---:|---:|
//! | `engine` | 7 | 2,752 | 22 | 2,812 |
//! | `a-gdl` | 329 | 63,735 | 842 | 85,223 |
//! | `b-gdl` | 344 | 80,394 | 857 | 101,882 |
//! | `c-gdl` | 39 | 28,204 | 54 | 28,264 |
//! | `embed-engine` | 8 | 6,548 | 23 | 88,868 |
//! | `embeddings-gdl` | 178 | 41,735 | 2,262 | 527,183 |
//!
//! At the switch the Dynamo `link` chains made more allocations than these rows at both item
//! counts, the same number more at each, so nothing per item: `a` +17 (346 and 859), `b` +55
//! (399 and 912), `c` +32 (71 and 86), `embed` +21 (199 and 2,283).
//!
//! The rows carry the chat and preprocessed requests boxed (`dynamo_gdl_adapters::Boxed`),
//! accepted at one allocation per box: `a-gdl` and `b-gdl` +2 (the chat request where it enters
//! the pipeline, the preprocessed request where the preprocessor builds it; Migration sends its
//! first attempt in that box), `c-gdl` +1 (the preprocessed request where it enters). Unboxed,
//! gdl's chain copies a 2 KB request at every node and future it moves through. Boxed, per
//! request (aarch64, `hotpath.sh` with `HOTPATH_RUNS=3`): `a-gdl` 19 fewer `memcpy` calls and
//! 7,394 fewer instructions (4.6%), `b-gdl` 45 and 20,276 (10.6%), `c-gdl` 27 and 9,529 (23%),
//! with nothing per item. Callgrind leaves out the system allocator; a `malloc` and `free` pair
//! costs about 170 instructions.
//! `lib/gdl-adapters/tests/allocations.rs` measures the parts in isolation: 2 allocations for
//! a native gdl operator nested under the request's scope, and 21 against 19 at the boundary
//! (`PipelineEngine` above, `EngineSink` below); a `link` stage made 14. Every operator in the gdl rows is native, and
//! Backend and the chat preprocessor are around nodes of the hop below them; the test engine
//! at the terminal enters through `EngineSink`, where the frontend's `RoutingHost` is a gdl
//! sink of its own.
//! Each gdl row asked for fewer bytes than the `link` chain it replaced: the `link` chain boxed a
//! future per edge, the gdl composition one per stage, and `embeddings-gdl` has no backend stage.
//! The largest allocation of `b` and `c` was Migration's boxed future, 59,592 bytes per request;
//! the largest of `b-gdl` was 13,376 bytes and of `c-gdl` 12,256 (debug build, gdl 165b166c).
//!
//! Run: `cargo test -p dynamo-llm --test allocations`. For instructions and `memcpy` calls per
//! request, see `lib/llm/benches/callgrind.sh` and `hotpath.sh`, which run this binary under
//! callgrind as `allocations row <row> <items> <requests>`. Callgrind counts only the measured
//! requests: the binary asks it for one dump after warm-up and one after the last measured
//! request (see `lib/runtime/tests/common/callgrind.rs`). Callgrind does not count the system
//! allocator, whose cost follows its heap state, nor this thread while it waits at the end of a
//! window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::input::{
    PreprocessedRouting, build_embeddings_pipeline, build_pipeline,
};
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest};
use dynamo_llm::protocols::common::llm_backend::{
    EmbeddingsEngineOutput, FinishReason, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::types::Annotated;
use dynamo_llm::types::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_llm::types::openai::embeddings::NvCreateEmbeddingRequest;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Context, Error, ManyOut, RouterMode, ServiceEngine, SingleIn};

#[path = "../../runtime/tests/common/callgrind.rs"]
mod callgrind;

const MODEL_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const WARMUP: usize = 50;
const REQUESTS: usize = 200;

/// Recorded allocations and bytes of the typical request (see [`Measured::typical`]). Debug
/// and release builds give the same numbers.
const BASELINE: &[Row] = &[
    // Every gdl row: +1 for the request id `PipelineEngine` holds while it dispatches, to
    // refuse a second in-flight request with that id.
    row("engine", 1, 7, 2752),
    row("engine", 16, 22, 2812),
    // +2 for the boxed chat and preprocessed requests (see the module docs).
    row("a-gdl", 1, 329, 63735),
    row("a-gdl", 16, 842, 85223),
    // +2 for the boxed chat and preprocessed requests (see the module docs).
    row("b-gdl", 1, 344, 80394),
    row("b-gdl", 16, 857, 101882),
    // +1 for the boxed preprocessed request (see the module docs).
    row("c-gdl", 1, 39, 28204),
    row("c-gdl", 16, 54, 28264),
    row("embed-engine", 1, 8, 6548),
    row("embed-engine", 16, 23, 88868),
    row("embeddings-gdl", 1, 178, 41735),
    row("embeddings-gdl", 16, 2262, 527183),
];

#[global_allocator]
static ALLOCATOR: Counting = Counting;

static OPEN: AtomicBool = AtomicBool::new(false);
static ON_THREAD_ALLOCS: AtomicU64 = AtomicU64::new(0);
static ON_THREAD_BYTES: AtomicU64 = AtomicU64::new(0);
static OFF_THREAD_ALLOCS: AtomicU64 = AtomicU64::new(0);
static OFF_THREAD_BYTES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// Set while the window itself allocates on the measuring thread.
    static HARNESS: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

impl Counting {
    fn record(size: usize) {
        if !OPEN.load(Ordering::Relaxed) {
            return;
        }
        // `try_with` because the allocator also runs while thread-locals are torn down.
        if HARNESS.try_with(Cell::get).unwrap_or(false) {
            return;
        }
        let on_thread = MEASURING.try_with(Cell::get).unwrap_or(false);
        let (allocs, bytes) = if on_thread {
            (&ON_THREAD_ALLOCS, &ON_THREAD_BYTES)
        } else {
            (&OFF_THREAD_ALLOCS, &OFF_THREAD_BYTES)
        };
        allocs.fetch_add(1, Ordering::Relaxed);
        bytes.fetch_add(size as u64, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        let _pause = callgrind::pause();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        let _pause = callgrind::pause();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        let _pause = callgrind::pause();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _pause = callgrind::pause();
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Allocations and bytes in one window, split by thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    on_allocs: u64,
    on_bytes: u64,
    off_allocs: u64,
    off_bytes: u64,
}

impl Counts {
    fn allocs(&self) -> u64 {
        self.on_allocs + self.off_allocs
    }

    fn bytes(&self) -> u64 {
        self.on_bytes + self.off_bytes
    }
}

/// One recorded row: a pipeline at a number of stream items, and its typical request.
#[derive(Clone, Copy, Debug)]
struct Row {
    name: &'static str,
    items: usize,
    allocs: u64,
    /// Reported, not gated: it follows the size of every boxed future on the path, which
    /// changes with unrelated edits.
    bytes: u64,
}

const fn row(name: &'static str, items: usize, allocs: u64, bytes: u64) -> Row {
    Row {
        name,
        items,
        allocs,
        bytes,
    }
}

/// Run `body` with counting on, and keep counting until the tasks it spawned have finished.
///
/// A request's tasks can outlive its future, for example a task still waiting on I/O. Closing
/// the window when the future returns would move their allocations into the next window, or
/// out of every window.
///
/// The caller runs every window inside one `block_on`: a current-thread runtime allocates its
/// deferred-waker list on the first yield of each `block_on`, and a window per `block_on` would
/// count that list every time.
async fn window<F: std::future::Future>(body: F) -> (Counts, F::Output) {
    let runtime = tokio::runtime::Handle::current().metrics();
    let tasks_before = runtime.num_alive_tasks();
    for c in [
        &ON_THREAD_ALLOCS,
        &ON_THREAD_BYTES,
        &OFF_THREAD_ALLOCS,
        &OFF_THREAD_BYTES,
    ] {
        c.store(0, Ordering::SeqCst);
    }
    MEASURING.with(|m| m.set(true));
    OPEN.store(true, Ordering::SeqCst);
    let out = body.await;
    // The window reads no clock and sets no timer unless a request leaves tasks running: under
    // callgrind, the time driver's work follows wall-clock time, which made a row's
    // instructions per request differ by about 1% between runs of one binary.
    let mut settle_start = None;
    let total =
        || ON_THREAD_ALLOCS.load(Ordering::SeqCst) + OFF_THREAD_ALLOCS.load(Ordering::SeqCst);
    // Settle in rounds until one round allocates nothing. Work that spawns more work allocates
    // to spawn it, so a quiet round means nothing is left that could allocate.
    loop {
        let before = total();
        loop {
            // Each yield also polls the I/O driver, so tasks waiting on a socket make progress.
            tokio::task::yield_now().await;
            let alive = runtime.num_alive_tasks();
            if alive <= tasks_before {
                break;
            }
            let start = *settle_start.get_or_insert_with(std::time::Instant::now);
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "a request left {} tasks running",
                alive - tasks_before
            );
        }
        // `num_alive_tasks` does not see blocking jobs. The runtime has one blocking thread, so
        // a no-op job finishes only after every job queued before it. Spawning it allocates on
        // this thread; that is the harness's cost, not the request's.
        // There is no timeout on it, for the reason above: a request that leaves blocking work
        // running for ever hangs here, so run this binary under `timeout`.
        // Callgrind does not count this thread while it waits: no request task is left on it,
        // and how often it parks and wakes depends on when the blocking thread runs. The
        // blocking thread is still counted.
        let pause = callgrind::pause();
        HARNESS.with(|h| h.set(true));
        let barrier = tokio::task::spawn_blocking(|| ());
        HARNESS.with(|h| h.set(false));
        barrier.await.expect("barrier");
        drop(pause);
        if total() == before {
            break;
        }
    }
    OPEN.store(false, Ordering::SeqCst);
    MEASURING.with(|m| m.set(false));
    let counts = Counts {
        on_allocs: ON_THREAD_ALLOCS.load(Ordering::SeqCst),
        on_bytes: ON_THREAD_BYTES.load(Ordering::SeqCst),
        off_allocs: OFF_THREAD_ALLOCS.load(Ordering::SeqCst),
        off_bytes: OFF_THREAD_BYTES.load(Ordering::SeqCst),
    };
    (counts, out)
}

/// Drain a response stream; return its item count. Panics on an error item.
async fn drain<T>(mut stream: ManyOut<Annotated<T>>) -> usize
where
    T: dynamo_runtime::engine::Data + std::fmt::Debug,
{
    let mut items = 0;
    while let Some(item) = stream.next().await {
        assert!(!item.is_error(), "error item: {item:?}");
        items += 1;
    }
    items
}

/// A worker that answers every request with `items` one-token outputs, the last one finished.
struct LlmEngine {
    items: usize,
    token_ids: Arc<Vec<u32>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for LlmEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (_, context) = request.into_parts();
        let items = self.items;
        let token_ids = self.token_ids.clone();
        let stream = futures::stream::iter((0..items).map(move |i| {
            let mut output = LLMEngineOutput::stop();
            output.token_ids = vec![token_ids[i % token_ids.len()]];
            if i + 1 < items {
                output.finish_reason = None;
            } else {
                output.finish_reason = Some(FinishReason::Stop);
            }
            Annotated::from_data(output)
        }));
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

/// An embeddings worker that returns one output with one vector per input.
struct EmbeddingsEngine {
    vector: Arc<String>,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
        Error,
    > for EmbeddingsEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<EmbeddingsEngineOutput>>, Error> {
        let (request, context) = request.into_parts();
        let inputs = request.token_ids.len();
        let tokens: usize = request.token_ids.iter().map(Vec::len).sum();
        let output = EmbeddingsEngineOutput {
            embeddings: (0..inputs).map(|_| self.vector.as_ref().clone()).collect(),
            prompt_tokens: tokens as u32,
            total_tokens: tokens as u32,
        };
        let stream = futures::stream::iter(std::iter::once(Annotated::from_data(output)));
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

/// Shared inputs: the card, tokenizer and token ids for the test engine to emit.
struct Fixture {
    card: ModelDeploymentCard,
    tokenizer: dynamo_llm::tokenizers::Tokenizer,
    token_ids: Arc<Vec<u32>>,
}

impl Fixture {
    fn load() -> Self {
        let card = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).expect("load card");
        let tokenizer = card.tokenizer().expect("tokenizer");
        let token_ids = tokenizer
            .encode("The quick brown fox jumps over the lazy dog near the quiet river bank today")
            .expect("encode")
            .token_ids()
            .to_vec();
        Self {
            card,
            tokenizer,
            token_ids: Arc::new(token_ids),
        }
    }

    fn llm_engine(&self, items: usize) -> Arc<LlmEngine> {
        Arc::new(LlmEngine {
            items,
            token_ids: self.token_ids.clone(),
        })
    }

    fn routing(&self, items: usize) -> PreprocessedRouting {
        let backend_engine: ServiceEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        > = self.llm_engine(items);
        PreprocessedRouting::new(
            backend_engine,
            PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::RoundRobin, None),
            EncoderRouter::disabled(),
        )
    }
}

fn chat_request() -> NvCreateChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "messages": [{"role": "user", "content": "Write one sentence about a fox and a river."}],
        "max_tokens": 64,
        "stream": true,
    }))
    .expect("chat request")
}

fn preprocessed_request(fixture: &Fixture) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("TinyLlama_v1.1".to_string())
        .token_ids(fixture.token_ids.as_ref().clone())
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

fn embedding_request(inputs: usize) -> NvCreateEmbeddingRequest {
    let input: Vec<String> = (0..inputs)
        .map(|i| format!("A fox crossed the river, sentence {i}."))
        .collect();
    serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "input": input,
    }))
    .expect("embedding request")
}

/// Per-request results of one row.
struct Measured {
    name: &'static str,
    items: usize,
    counts: Vec<Counts>,
}

impl Measured {
    fn min_max(&self, f: impl Fn(&Counts) -> u64) -> (u64, u64) {
        let min = self.counts.iter().map(&f).min().unwrap_or(0);
        let max = self.counts.iter().map(&f).max().unwrap_or(0);
        (min, max)
    }

    /// The most common (allocations, bytes) of a request, and how many requests had it.
    ///
    /// A few requests can cost more than the rest. With 16 embeddings inputs, about 4 requests
    /// in 200 make one more allocation off the runtime thread, where rayon splits the batch
    /// encode; the cause is not traced further. A real regression moves every request, so the
    /// typical request is the number to guard.
    fn typical(&self) -> ((u64, u64), usize) {
        let mut seen: Vec<((u64, u64), usize)> = Vec::new();
        for c in &self.counts {
            let key = (c.allocs(), c.bytes());
            match seen.iter_mut().find(|(k, _)| *k == key) {
                Some((_, n)) => *n += 1,
                None => seen.push((key, 1)),
            }
        }
        seen.into_iter()
            .max_by_key(|&((allocs, _), n)| (n, std::cmp::Reverse(allocs)))
            .unwrap_or_default()
    }

    fn report(&self) {
        let ((allocs, bytes), n) = self.typical();
        let (amin, amax) = self.min_max(Counts::allocs);
        let (omin, omax) = self.min_max(|c| c.off_allocs);
        println!(
            "{:<14} items={:<3} allocs={allocs} bytes={bytes} share={n}/{} allocs_range={amin}..{amax} off_thread_allocs={omin}..{omax}",
            self.name,
            self.items,
            self.counts.len()
        );
    }
}

/// Measure `REQUESTS` requests after `WARMUP`; `make` builds each request's future outside the
/// window, and the future must yield the number of stream items it drained.
async fn measure<F, Fut>(name: &'static str, items: usize, requests: usize, mut make: F) -> Measured
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = usize>,
{
    let mut counts = Vec::with_capacity(requests);
    for i in 0..WARMUP + requests {
        if i == WARMUP {
            callgrind::dump_stats_at(c"setup and warm-up");
        }
        let fut = make();
        let (c, drained) = window(fut).await;
        assert!(drained > 0, "{name}: empty stream");
        if i >= WARMUP {
            counts.push(c);
        }
    }
    callgrind::dump_stats_at(c"measured requests");
    Measured {
        name,
        items,
        counts,
    }
}

async fn measure_engine(fixture: &Fixture, items: usize, requests: usize) -> Measured {
    let engine = fixture.llm_engine(items);
    measure("engine", items, requests, || {
        let engine = engine.clone();
        let request = preprocessed_request(fixture);
        async move { drain(engine.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

async fn measure_a(fixture: &Fixture, items: usize, requests: usize) -> Measured {
    let engine: ExecutionContext = fixture.llm_engine(items);
    let pipeline = build_pipeline::<
        NvCreateChatCompletionRequest,
        NvCreateChatCompletionStreamResponse,
    >(&fixture.card, engine, fixture.tokenizer.clone())
    .await
    .expect("pipeline a");
    measure("a-gdl", items, requests, || {
        let pipeline = pipeline.clone();
        let request = chat_request();
        async move { drain(pipeline.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

async fn measure_b(fixture: &Fixture, items: usize, requests: usize) -> Measured {
    let routing = fixture.routing(items);
    let cancel = tokio_util::sync::CancellationToken::new();
    let dynamo_renderer::PromptFormatter::OAI(formatter) =
        prompt_formatter_from_mdc(&fixture.card).expect("formatter");
    let preprocessor = OpenAIPreprocessor::new_with_parts_and_cancel(
        fixture.card.clone(),
        formatter,
        fixture.tokenizer.clone(),
        Some(cancel),
    )
    .expect("preprocessor");
    let pipeline = routing
        .build_pipeline::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>(
            &fixture.card,
            preprocessor,
            fixture.tokenizer.clone(),
            0,
            None,
            Arc::new(Metrics::new()),
        )
        .expect("pipeline b");
    measure("b-gdl", items, requests, || {
        let pipeline = pipeline.clone();
        let request = chat_request();
        async move { drain(pipeline.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

async fn measure_c(fixture: &Fixture, items: usize, requests: usize) -> Measured {
    let pipeline = fixture
        .routing(items)
        .build_preprocessed_pipeline(&fixture.card, 0, None, Arc::new(Metrics::new()))
        .expect("pipeline c");
    measure("c-gdl", items, requests, || {
        let pipeline = pipeline.clone();
        let request = preprocessed_request(fixture);
        async move { drain(pipeline.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

fn embeddings_engine() -> Arc<EmbeddingsEngine> {
    // A 1024-dimension vector, as the worker sends it: base64 of little-endian f32.
    let raw: Vec<u8> = (0..1024u32)
        .flat_map(|i| (i as f32 / 1024.0).to_le_bytes())
        .collect();
    Arc::new(EmbeddingsEngine {
        vector: Arc::new(base64::engine::general_purpose::STANDARD.encode(raw)),
    })
}

async fn measure_embed_engine(fixture: &Fixture, inputs: usize, requests: usize) -> Measured {
    let engine = embeddings_engine();
    let token_ids = fixture.token_ids.clone();
    measure("embed-engine", inputs, requests, || {
        let engine = engine.clone();
        let request = PreprocessedEmbeddingRequest::builder()
            .token_ids(vec![token_ids.as_ref().clone(); inputs])
            .model("TinyLlama_v1.1".to_string())
            .encoding_format(None)
            .dimensions(None)
            .build()
            .expect("embedding request");
        async move { drain(engine.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

async fn measure_embed(fixture: &Fixture, inputs: usize, requests: usize) -> Measured {
    let mut card = fixture.card.clone();
    card.model_type = ModelType::Embedding;
    let preprocessor =
        OpenAIPreprocessor::new_for_embeddings(card).expect("embeddings preprocessor");
    let pipeline = build_embeddings_pipeline(preprocessor, embeddings_engine());
    measure("embeddings-gdl", inputs, requests, || {
        let pipeline = pipeline.clone();
        let request = embedding_request(inputs);
        async move { drain(pipeline.generate(Context::new(request)).await.unwrap()).await }
    })
    .await
}

async fn measure_row(fixture: &Fixture, name: &str, items: usize, requests: usize) -> Measured {
    match name {
        "engine" => measure_engine(fixture, items, requests).await,
        "a-gdl" => measure_a(fixture, items, requests).await,
        "b-gdl" => measure_b(fixture, items, requests).await,
        "c-gdl" => measure_c(fixture, items, requests).await,
        "embed-engine" => measure_embed_engine(fixture, items, requests).await,
        "embeddings-gdl" => measure_embed(fixture, items, requests).await,
        other => panic!("unknown row {other}"),
    }
}

/// The counter itself: an empty window counts nothing, one `Box` counts one allocation.
async fn check_counter() {
    // The first yield inside this `block_on` allocates the runtime's deferred-waker list.
    window(async {}).await;
    let (empty, ()) = window(async {}).await;
    assert_eq!(empty, Counts::default(), "an empty window allocated");
    let (one, ()) = window(async {
        std::hint::black_box(Box::new(7u64));
    })
    .await;
    assert_eq!(
        (one.on_allocs, one.on_bytes, one.off_allocs),
        (1, 8, 0),
        "one Box::new<u64> must count one allocation of 8 bytes"
    );

    // Blocking work a request leaves behind counts in that request's window.
    let (detached, ()) = window(async {
        drop(tokio::task::spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            std::hint::black_box(Box::new(7u64));
        }));
    })
    .await;
    assert_eq!(
        (detached.off_allocs, detached.off_bytes),
        (1, 8),
        "a detached blocking job's allocation must count in its window"
    );
    // So does blocking work that blocking work spawns after the window began to settle.
    let (nested, ()) = window(async {
        drop(tokio::task::spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            drop(tokio::task::spawn_blocking(|| {
                std::thread::sleep(std::time::Duration::from_millis(20));
                std::hint::black_box(Box::new([0u8; 4096]));
            }));
        }));
    })
    .await;
    assert!(
        nested.off_bytes >= 4096,
        "a nested blocking job's allocation must count in its window: {nested:?}"
    );
}

/// The counts must not depend on the caller's environment or on the host's core count: run one
/// row in a child process with tokenizer parallelism set explicitly and an odd rayon thread
/// count, and compare it with the record.
fn check_environment() {
    let (name, items) = ("embeddings-gdl", 16);
    let output = std::process::Command::new(std::env::current_exe().expect("current exe"))
        .args(["row", name, &items.to_string(), "50"])
        .env("TOKENIZERS_PARALLELISM", "true")
        .env("RAYON_NUM_THREADS", "3")
        .output()
        .expect("run child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "child failed: {stdout}");
    let allocs: u64 = stdout
        .split_whitespace()
        .find_map(|word| word.strip_prefix("allocs="))
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no allocs= in child output: {stdout}"));
    let row = BASELINE
        .iter()
        .find(|r| r.name == name && r.items == items)
        .expect("recorded row");
    assert_eq!(
        allocs, row.allocs,
        "{name} items={items} in another environment: {allocs} allocations, recorded {}",
        row.allocs
    );
}

const ROWS: &[(&str, usize)] = &[
    ("engine", 1),
    ("engine", 16),
    ("a-gdl", 1),
    ("a-gdl", 16),
    ("b-gdl", 1),
    ("b-gdl", 16),
    ("c-gdl", 1),
    ("c-gdl", 16),
    ("embed-engine", 1),
    ("embed-engine", 16),
    ("embeddings-gdl", 1),
    ("embeddings-gdl", 16),
];

fn main() {
    // Pin the environment that changes the path. `TOKENIZERS_PARALLELISM` unset is the
    // frontend's default; spelling it out costs an allocation per encode, where the tokenizer
    // reads it. A batch encode splits across rayon threads, and one thread splits it differently,
    // so a fixed count keeps the embeddings rows the same on every host.
    // SAFETY: no other thread exists yet.
    for (key, _) in std::env::vars_os() {
        let key_str = key.to_string_lossy();
        if key_str.starts_with("DYN_") || key_str == "TOKENIZERS_PARALLELISM" {
            unsafe { std::env::remove_var(&key) };
        }
    }
    unsafe { std::env::set_var("RAYON_NUM_THREADS", "4") };
    // The test runs from the crate directory; the callgrind script may not.
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR")).expect("crate dir");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixture = Fixture::load();
    // One blocking thread, so a window can wait for the blocking work a request leaves. It must
    // not exit when idle: starting a new one allocates.
    let mut builder = tokio::runtime::Builder::new_current_thread();
    builder
        .enable_all()
        .max_blocking_threads(1)
        .thread_keep_alive(std::time::Duration::from_secs(24 * 3600));
    let rt = callgrind::uncount_parked(&mut builder)
        .build()
        .expect("runtime");
    rt.block_on(run(&fixture, &args));
}

async fn run(fixture: &Fixture, args: &[String]) {
    // Start the blocking thread and set up its parking_lot per-thread data inside the job, so
    // the setup is done when the job returns. The thread waits for work on a parking_lot
    // condvar, and its first wait would otherwise allocate in whichever window is open.
    tokio::task::spawn_blocking(|| {
        let lock = parking_lot::Mutex::new(());
        parking_lot::Condvar::new().wait_for(&mut lock.lock(), std::time::Duration::from_millis(1));
    })
    .await
    .expect("blocking pool");

    if args.first().map(String::as_str) == Some("row") {
        let name = args.get(1).expect("row");
        let items: usize = args.get(2).expect("items").parse().expect("items");
        let requests: usize = args.get(3).expect("requests").parse().expect("requests");
        measure_row(fixture, name, items, requests).await.report();
        return;
    }

    check_counter().await;
    check_environment();
    let mut failures = Vec::new();
    for &(name, items) in ROWS {
        let measured = measure_row(fixture, name, items, REQUESTS).await;
        measured.report();
        let Some(row) = BASELINE.iter().find(|r| r.name == name && r.items == items) else {
            failures.push(format!("{name} items={items}: no recorded baseline"));
            continue;
        };
        let ((allocs, bytes), n) = measured.typical();
        if allocs < row.allocs {
            println!(
                "{name} items={items}: {allocs} allocations per request, below the recorded {}; lower the record",
                row.allocs
            );
        }
        if n * 2 <= measured.counts.len() {
            failures.push(format!(
                "{name} items={items}: no count is shared by most requests ({n} of {})",
                measured.counts.len()
            ));
        }
        if allocs > row.allocs {
            failures.push(format!(
                "{name} items={items}: {allocs} allocations per request, recorded {}",
                row.allocs
            ));
        }
        if bytes != row.bytes {
            println!(
                "{name} items={items}: {bytes} bytes per request, recorded {}",
                row.bytes
            );
        }
    }
    assert!(
        failures.is_empty(),
        "allocation baseline exceeded:\n{}",
        failures.join("\n")
    );
}
