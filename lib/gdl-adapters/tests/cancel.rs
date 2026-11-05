// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation scenarios, each run through the frontend's own pipelines on both local
//! pipelines (`DYN_LOCAL_PIPELINE` unset and `gdl`): what the caller receives, and what the
//! worker's context sees.
//!
//! Dynamo's `link` chain is the reference. A scenario where the gdl composition differs is
//! either pinned as an accepted difference, with the reason, or asserted against the reference
//! and ignored with the reason it fails.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::input::{LocalPipeline, PreprocessedRouting, build_pipeline};
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, RouterMode, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::{StreamExt, stream};
use tokio::sync::mpsc;

type Out = ManyOut<Annotated<LLMEngineOutput>>;
type Engine = Arc<dyn AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error>>;
type Pipeline<Req, T> = Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>;
type Chat = NvCreateChatCompletionRequest;
type ChatOut = NvCreateChatCompletionStreamResponse;

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// Both local pipelines, the reference first.
///
/// Transition: remove the Dynamo case and keep the gdl assertions when the gdl pipeline is the
/// only local pipeline.
const BOTH: [LocalPipeline; 2] = [LocalPipeline::Dynamo, LocalPipeline::Gdl];

/// What a worker does after the items of one stream.
#[derive(Clone, Copy, Debug)]
enum Then {
    /// End the stream.
    End,
    /// Wait for the request's kill, then end with no more items: a stalled worker.
    HoldUntilKilled,
    /// Kill the request itself, then send `after` more items and end.
    KillThenSend { after: u32 },
}

/// One call of [`Worker`].
#[derive(Clone, Debug)]
enum Turn {
    /// Fail the call with an eager `WorkerUnavailable`.
    Unavailable,
    /// Wait in the eager phase for the request's kill, then answer with no item.
    HangUntilKilled,
    /// Stream these items, then do as `Then` says.
    Stream(Vec<Annotated<LLMEngineOutput>>, Then),
    /// Stream ready items for as long as the caller reads them.
    Endless,
}

/// What one call of [`Worker`] saw.
struct Call {
    context: Arc<dyn AsyncEngineContext>,
    /// Set when the call's response stream is dropped.
    dropped: Arc<AtomicBool>,
}

/// A worker that answers its calls with `turns` in order; the last turn answers every call
/// after it. It reports each call's index on `entered` as the call arrives.
struct Worker {
    turns: Vec<Turn>,
    calls: Mutex<Vec<Call>>,
    produced: Arc<AtomicUsize>,
    entered: mpsc::UnboundedSender<usize>,
}

impl Worker {
    fn new(turns: Vec<Turn>) -> (Arc<Self>, mpsc::UnboundedReceiver<usize>) {
        let (entered, arrivals) = mpsc::unbounded_channel();
        let worker = Arc::new(Self {
            turns,
            calls: Mutex::default(),
            produced: Arc::default(),
            entered,
        });
        (worker, arrivals)
    }

    fn call(&self, index: usize) -> (Arc<dyn AsyncEngineContext>, bool) {
        let calls = self.calls.lock().unwrap();
        let call = &calls[index];
        (call.context.clone(), call.dropped.load(Ordering::Acquire))
    }

    fn calls(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

/// Sets its flag when dropped: kept in a response stream, it tells when the stream is dropped.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Worker {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        let context = request.context();
        let dropped = Arc::new(AtomicBool::new(false));
        let index = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(Call {
                context: context.clone(),
                dropped: dropped.clone(),
            });
            calls.len() - 1
        };
        let _ = self.entered.send(index);
        let turn = self.turns[index.min(self.turns.len() - 1)].clone();
        let flag = DropFlag(dropped);
        let watched = context.clone();
        match turn {
            Turn::Unavailable => Err(worker_unavailable()),
            Turn::HangUntilKilled => {
                context.killed().await;
                Ok(ResponseStream::new(Box::pin(stream::empty()), context))
            }
            Turn::Stream(items, then) => {
                let tail = stream::once(async move {
                    let _flag = &flag;
                    match then {
                        Then::End => 0,
                        Then::HoldUntilKilled => {
                            watched.killed().await;
                            0
                        }
                        Then::KillThenSend { after } => {
                            watched.kill();
                            after
                        }
                    }
                })
                .flat_map(|after| stream::iter((0..after).map(|i| token(100 + i))));
                Ok(ResponseStream::new(
                    Box::pin(stream::iter(items).chain(tail)),
                    context,
                ))
            }
            Turn::Endless => {
                let produced = self.produced.clone();
                let items = stream::iter(0u32..).map(move |i| {
                    let _flag = &flag;
                    produced.fetch_add(1, Ordering::AcqRel);
                    token(i)
                });
                Ok(ResponseStream::new(Box::pin(items), context))
            }
        }
    }
}

fn worker_unavailable() -> Error {
    DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message("worker is not ready")
        .build()
        .into()
}

/// An error item a worker's stream carries when the worker goes away mid-stream: Migration
/// retries it.
fn went_away() -> Annotated<LLMEngineOutput> {
    Annotated::from_err(
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("worker went away")
            .build(),
    )
}

/// An error item Migration does not retry.
fn internal_error() -> Annotated<LLMEngineOutput> {
    Annotated::from_err(
        DynamoError::builder()
            .class(ErrorClass::Internal)
            .message("an internal error item")
            .build(),
    )
}

fn token(id: u32) -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        token_ids: vec![id],
        ..Default::default()
    })
}

fn last_token(id: u32) -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        token_ids: vec![id],
        ..LLMEngineOutput::length()
    })
}

fn card() -> ModelDeploymentCard {
    ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap()
}

fn routing(engine: Engine, local_pipeline: LocalPipeline) -> PreprocessedRouting {
    PreprocessedRouting::new(
        engine,
        PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None),
        EncoderRouter::disabled(),
    )
    .with_local_pipeline(local_pipeline)
}

/// The in-process pipeline over a token engine (A): preprocessor, Backend, engine.
async fn pipeline_a(local_pipeline: LocalPipeline, engine: Engine) -> Pipeline<Chat, ChatOut> {
    let card = card();
    let tokenizer = card.tokenizer().unwrap();
    build_pipeline::<Chat, ChatOut>(&card, engine, tokenizer, local_pipeline)
        .await
        .unwrap()
}

/// The routed chat pipeline (B): preprocessor, Migration, Backend, the encoder and prefill
/// routers (disabled, as in aggregated serving), the worker.
fn pipeline_b(
    local_pipeline: LocalPipeline,
    engine: Engine,
    migration_limit: u32,
) -> Pipeline<Chat, ChatOut> {
    let card = card();
    let tokenizer = card.tokenizer().unwrap();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone()).unwrap();
    routing(engine, local_pipeline)
        .build_pipeline::<Chat, ChatOut>(
            &card,
            preprocessor,
            tokenizer,
            migration_limit,
            None,
            Arc::new(Metrics::new()),
        )
        .unwrap()
}

/// The routed pipeline for callers that bring their own pre- and post-processing (C):
/// Migration, the routers, the worker.
fn pipeline_c(
    local_pipeline: LocalPipeline,
    engine: Engine,
    migration_limit: u32,
) -> Pipeline<PreprocessedRequest, LLMEngineOutput> {
    pipeline_c_with_metrics(
        local_pipeline,
        engine,
        migration_limit,
        Arc::new(Metrics::new()),
    )
}

/// [`pipeline_c`], recording into `metrics`.
fn pipeline_c_with_metrics(
    local_pipeline: LocalPipeline,
    engine: Engine,
    migration_limit: u32,
    metrics: Arc<Metrics>,
) -> Pipeline<PreprocessedRequest, LLMEngineOutput> {
    routing(engine, local_pipeline)
        .build_preprocessed_pipeline(&card(), migration_limit, None, metrics)
        .unwrap()
}

fn preprocessed() -> SingleIn<PreprocessedRequest> {
    let request = PreprocessedRequest::builder()
        .model("TinyLlama_v1.1".to_string())
        .token_ids(vec![1, 450, 4996])
        .stop_conditions(StopConditions {
            max_tokens: Some(512),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![2])
        .annotations(vec![])
        .build()
        .unwrap();
    Context::with_id_and_metadata(request, "req-1".to_string(), BTreeMap::new())
}

fn chat() -> SingleIn<Chat> {
    let chat: Chat = serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "messages": [{"role": "user", "content": "Write one sentence about a fox."}],
        "max_tokens": 512,
        "stream": true,
    }))
    .unwrap();
    Context::with_id_and_metadata(chat, "req-1".to_string(), BTreeMap::new())
}

async fn next_within<T>(stream: &mut ManyOut<T>, wait: Duration) -> Option<Option<T>>
where
    T: Data,
{
    tokio::time::timeout(wait, stream.next()).await.ok()
}

async fn rest<T: Data>(stream: ManyOut<T>) -> Vec<T> {
    tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends")
}

/// A request whose worker sends one token and then stalls, with no deadline: the caller reads
/// the token, sees no more within a while, and then kills the request.
///
/// It reports how many items came first, whether the stall held, whether the kill reached the
/// worker at once, while the stream was held unread, and what came after the kill.
async fn stall_then_kill<Req: Data, T: Data>(
    pipeline: Pipeline<Req, T>,
    request: SingleIn<Req>,
    worker: &Worker,
) -> String {
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    let first = next_within(&mut stream, Duration::from_secs(5))
        .await
        .expect("the first item arrives")
        .expect("the stream is not over");
    let stalled = next_within(&mut stream, Duration::from_millis(200))
        .await
        .is_none();
    context.kill();
    let (worker_context, _) = worker.call(0);
    let killed = worker_context.is_killed();
    let after = rest(stream).await;
    format!(
        "first error: {}, stalled: {stalled}, worker killed: {killed}, after the kill: {} items, {} errors",
        first.is_error(),
        after.len(),
        after.iter().filter(|item| item.is_error()).count()
    )
}

const STALL_THEN_KILL: &str =
    "first error: false, stalled: true, worker killed: true, after the kill: 0 items, 0 errors";

/// A client disconnect mid-stream while the worker stalls. No deadline exists on either local
/// pipeline, so the stall holds until the kill; the kill reaches the worker's context at once,
/// also while the caller holds the stream unread (a slow client), and the stream then ends
/// quietly. Pipeline A is rooted so that a kill does not cancel its `Cx`.
#[tokio::test]
async fn a_kill_during_a_stall_ends_pipeline_a_quietly_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![Turn::Stream(vec![token(450)], Then::HoldUntilKilled)]);
        let pipeline = pipeline_a(local_pipeline, worker.clone()).await;
        outcomes.push(stall_then_kill(pipeline, chat(), &worker).await);
    }
    assert_eq!(outcomes, [STALL_THEN_KILL; 2]);
}

/// As for pipeline A, through pipeline B, which on gdl is rooted so that a kill cancels its
/// `Cx`.
#[tokio::test]
async fn a_kill_during_a_stall_ends_pipeline_b_quietly_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![Turn::Stream(vec![token(450)], Then::HoldUntilKilled)]);
        let pipeline = pipeline_b(local_pipeline, worker.clone(), 3);
        outcomes.push(stall_then_kill(pipeline, chat(), &worker).await);
    }
    assert_eq!(outcomes, [STALL_THEN_KILL; 2]);
}

/// As for pipeline B, through pipeline C.
#[tokio::test]
async fn a_kill_during_a_stall_ends_pipeline_c_quietly_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![Turn::Stream(vec![token(450)], Then::HoldUntilKilled)]);
        let pipeline = pipeline_c(local_pipeline, worker.clone(), 3);
        outcomes.push(stall_then_kill(pipeline, preprocessed(), &worker).await);
    }
    assert_eq!(outcomes, [STALL_THEN_KILL; 2]);
}

/// A request whose worker has answered but holds its first item: the caller sees nothing for a
/// while, then kills the request. It reports what came after the kill and what the worker saw.
async fn kill_before_the_first_token<Req: Data, T: Data>(
    pipeline: Pipeline<Req, T>,
    request: SingleIn<Req>,
    worker: &Worker,
) -> String {
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    let held = next_within(&mut stream, Duration::from_millis(100))
        .await
        .is_none();
    context.kill();
    let after = rest(stream).await;
    let (worker_context, _) = worker.call(0);
    format!(
        "held: {held}, after the kill: {} items, {} errors, worker killed: {}",
        after.len(),
        after.iter().filter(|item| item.is_error()).count(),
        worker_context.is_killed()
    )
}

/// A client disconnect before the first token: the worker has answered, but holds its first
/// item. The stream ends with no item, and the worker sees the kill.
#[tokio::test]
async fn a_kill_before_the_first_token_ends_pipelines_b_and_c_quietly_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let hold = || Worker::new(vec![Turn::Stream(vec![], Then::HoldUntilKilled)]).0;
        let worker = hold();
        let pipeline = pipeline_b(local_pipeline, worker.clone(), 3);
        outcomes.push(kill_before_the_first_token(pipeline, chat(), &worker).await);
        let worker = hold();
        let pipeline = pipeline_c(local_pipeline, worker.clone(), 3);
        outcomes.push(kill_before_the_first_token(pipeline, preprocessed(), &worker).await);
    }
    assert_eq!(
        outcomes,
        ["held: true, after the kill: 0 items, 0 errors, worker killed: true"; 4]
    );
}

/// What a migratable failure on `turns` gives through pipeline C: the tokens the caller
/// received, whether any item was an error, and per call whether the worker's context was
/// stopped or killed and whether the call's stream was dropped. Also whether the request's own
/// context was stopped or killed.
async fn migrate_on_c(local_pipeline: LocalPipeline, turns: Vec<Turn>) -> String {
    let (worker, _) = Worker::new(turns);
    let request = preprocessed();
    let context = request.context();
    let stream = pipeline_c(local_pipeline, worker.clone(), 3)
        .generate(request)
        .await
        .unwrap();
    let items = rest(stream).await;
    let tokens: Vec<u32> = items
        .iter()
        .filter_map(|item| item.data.as_ref())
        .flat_map(|data| data.token_ids.clone())
        .collect();
    let calls: Vec<String> = (0..worker.calls())
        .map(|index| {
            let (call, dropped) = worker.call(index);
            format!(
                "call {index}: stopped {}, killed {}, dropped {dropped}",
                call.is_stopped(),
                call.is_killed()
            )
        })
        .collect();
    format!(
        "tokens {tokens:?}, errors {}, request stopped {}, killed {}; {}",
        items.iter().filter(|item| item.is_error()).count(),
        context.is_stopped(),
        context.is_killed(),
        calls.join("; ")
    )
}

/// A worker that dies mid-stream: Migration retries on another call, and only the failed
/// attempt is torn down. Its stream is dropped, the request goes on, and nothing stops or
/// kills a worker's context.
#[tokio::test]
async fn a_worker_that_dies_mid_stream_is_retried_and_only_its_attempt_ends_on_both_local_pipelines()
 {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            migrate_on_c(
                local_pipeline,
                vec![
                    Turn::Stream(vec![token(450), went_away()], Then::End),
                    Turn::Stream(vec![last_token(4996)], Then::End),
                ],
            )
            .await,
        );
    }
    let expected = "tokens [450, 4996], errors 0, request stopped false, killed false; \
                    call 0: stopped false, killed false, dropped true; \
                    call 1: stopped false, killed false, dropped true";
    assert_eq!(outcomes, [expected; 2]);
}

/// A worker that is unavailable at dispatch: an eager error, then a retry that streams.
#[tokio::test]
async fn an_unavailable_worker_is_retried_at_dispatch_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            migrate_on_c(
                local_pipeline,
                vec![
                    Turn::Unavailable,
                    Turn::Stream(vec![last_token(450)], Then::End),
                ],
            )
            .await,
        );
    }
    let expected = "tokens [450], errors 0, request stopped false, killed false; \
                    call 0: stopped false, killed false, dropped true; \
                    call 1: stopped false, killed false, dropped true";
    assert_eq!(outcomes, [expected; 2]);
}

/// An error item Migration does not retry passes on to the caller, and the stream goes on: a
/// Dynamo error item does not end a stream.
#[tokio::test]
async fn a_non_migratable_error_item_passes_and_the_stream_goes_on_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            migrate_on_c(
                local_pipeline,
                vec![Turn::Stream(
                    vec![token(450), internal_error(), last_token(4996)],
                    Then::End,
                )],
            )
            .await,
        );
    }
    let expected = "tokens [450, 4996], errors 1, request stopped false, killed false; \
                    call 0: stopped false, killed false, dropped true";
    assert_eq!(outcomes, [expected; 2]);
}

/// A worker that kills its own call's context and then ends cleanly. The kill stays in that
/// call: Migration makes each attempt's context a child of the request's, and a kill goes down
/// the tree, not up. So the request goes on, and the caller receives everything the worker
/// sent, with no error, on both local pipelines; gdl's `Cx` follows the request's context only,
/// so no gdl scope is cancelled either.
#[tokio::test]
async fn a_worker_that_kills_its_own_call_then_ends_is_read_to_its_end_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            migrate_on_c(
                local_pipeline,
                vec![Turn::Stream(
                    vec![token(450)],
                    Then::KillThenSend { after: 2 },
                )],
            )
            .await,
        );
    }
    let expected = "tokens [450, 100, 101], errors 0, request stopped false, killed false; \
                    call 0: stopped true, killed true, dropped true";
    assert_eq!(outcomes, [expected; 2]);
}

/// What a client disconnect while Migration dispatches its retry gives through pipeline C: the
/// first attempt sends one token and fails with a migratable error item, and the retry's
/// worker waits in its dispatch until the request is killed.
async fn kill_during_the_retry(local_pipeline: LocalPipeline) -> String {
    let (worker, mut arrivals) = Worker::new(vec![
        Turn::Stream(vec![token(450), went_away()], Then::End),
        Turn::HangUntilKilled,
    ]);
    let request = preprocessed();
    let context = request.context();
    let mut stream = pipeline_c(local_pipeline, worker.clone(), 3)
        .generate(request)
        .await
        .unwrap();
    let first = next_within(&mut stream, Duration::from_secs(5))
        .await
        .expect("the first item arrives")
        .expect("the stream is not over");
    assert!(!first.is_error(), "{first:?}");
    let reader = tokio::spawn(rest(stream));
    // Calls 0 and 1: the retry is in its dispatch.
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
            .await
            .expect("the retry reaches the worker")
            .unwrap();
    }
    context.kill();
    let after = reader.await.unwrap();
    let errors: Vec<String> = after
        .iter()
        .filter_map(|item| item.error.as_ref().map(|error| error.to_string()))
        .collect();
    format!(
        "after the kill: {} items, errors {errors:?}, calls {}",
        after.len(),
        worker.calls()
    )
}

/// A client disconnect while Migration's retry is still dispatching. On Dynamo's chain the
/// retry's worker sees the kill and answers with no item, and the stream ends quietly: the
/// failure that caused the retry is not sent, because the retry was made.
#[tokio::test]
async fn a_kill_during_a_retry_ends_pipeline_c_quietly_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(kill_during_the_retry(local_pipeline).await);
    }
    assert_eq!(outcomes, ["after the kill: 0 items, errors [], calls 2"; 2]);
}

/// A kill while Migration's retry after an eager `WorkerUnavailable` is still dispatching:
/// the request is still in its dispatch, so this is the case `kill_during_dispatch` in
/// `llm.rs` pins, at the second attempt.
#[tokio::test]
async fn a_kill_during_the_dispatch_of_a_retry_ends_pipeline_c_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, mut arrivals) = Worker::new(vec![Turn::Unavailable, Turn::HangUntilKilled]);
        let request = preprocessed();
        let context = request.context();
        let pipeline = pipeline_c(local_pipeline, worker.clone(), 3);
        let call = tokio::spawn(async move { pipeline.generate(request).await });
        for _ in 0..2 {
            tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
                .await
                .expect("the retry reaches the worker")
                .unwrap();
        }
        context.kill();
        let outcome = tokio::time::timeout(Duration::from_secs(5), call)
            .await
            .expect("the call ends after the kill")
            .unwrap();
        outcomes.push(match outcome {
            Ok(stream) => format!("ok, {} items", rest(stream).await.len()),
            Err(error) => match error.downcast_ref::<DynamoError>() {
                Some(error) => format!("error {:?}", error.class()),
                None => format!("error {error:#}"),
            },
        });
    }
    // An accepted difference, as in `kill_during_dispatch`: on gdl the kill drops the call in
    // flight and the call fails as `Cancelled`. The HTTP handler reports both as a client
    // disconnect.
    assert_eq!(outcomes, ["ok, 0 items", "error Cancelled"]);
}

/// A worker that ignores the kill and keeps sending ready items, 200 in all. The kill is sent
/// after the first item, and the caller keeps reading.
///
/// Dynamo's chain checks nothing: the caller reads everything the worker sends. On gdl,
/// pipeline C is rooted so that a kill cancels its `Cx`, and each attempt boundary checks the
/// cancel every 64 ready items, so the stream ends within 64 items of the kill. In the
/// frontend the HTTP handler stops reading at the kill, so a client never sees the overrun.
#[tokio::test]
async fn a_worker_that_ignores_the_kill_is_cut_off_only_on_gdl() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let items: Vec<_> = (0..200).map(token).collect();
        let (worker, _) = Worker::new(vec![Turn::Stream(items, Then::End)]);
        let request = preprocessed();
        let context = request.context();
        let mut stream = pipeline_c(local_pipeline, worker.clone(), 3)
            .generate(request)
            .await
            .unwrap();
        next_within(&mut stream, Duration::from_secs(5))
            .await
            .expect("the first item arrives")
            .expect("the stream is not over");
        context.kill();
        let after = rest(stream).await;
        outcomes.push((
            after.len() < 64,
            after.len() == 199,
            after.iter().any(|item| item.is_error()),
        ));
    }
    // An accepted difference: gdl ends sooner; neither sends an error.
    assert_eq!(outcomes, [(false, true, false), (true, false, false)]);
}

/// Every migration outcome `metrics` recorded, as `type/outcome xN`.
fn migration_outcomes(metrics: &Metrics) -> Vec<String> {
    use dynamo_runtime::metrics::prometheus_names::frontend_service::{
        migration_outcome, migration_type,
    };
    let model = card().display_name;
    let mut recorded = Vec::new();
    for kind in [migration_type::NEW_REQUEST, migration_type::ONGOING_REQUEST] {
        for outcome in [
            migration_outcome::SUCCESS,
            migration_outcome::FAILURE,
            migration_outcome::CANCELLED,
        ] {
            let count = metrics.get_migration_duration_sample_count(&model, kind, outcome);
            if count > 0 {
                recorded.push(format!("{kind}/{outcome} x{count}"));
            }
        }
    }
    recorded
}

/// The migration outcomes a request on `turns` records through pipeline C with
/// `migration_limit`, read after the caller has read the whole stream and dropped it and the
/// pipeline.
async fn outcomes_on_c(
    local_pipeline: LocalPipeline,
    turns: Vec<Turn>,
    migration_limit: u32,
) -> Vec<String> {
    let (worker, _) = Worker::new(turns);
    let metrics = Arc::new(Metrics::new());
    let pipeline =
        pipeline_c_with_metrics(local_pipeline, worker, migration_limit, metrics.clone());
    let stream = pipeline.generate(preprocessed()).await.unwrap();
    rest(stream).await;
    drop(pipeline);
    migration_outcomes(&metrics)
}

/// A migration that finishes, at dispatch or mid-stream, records one `success` and nothing
/// else, on both local pipelines: dropping the finished stream records no second outcome.
#[tokio::test]
async fn a_finished_migration_records_one_success_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            outcomes_on_c(
                local_pipeline,
                vec![
                    Turn::Unavailable,
                    Turn::Stream(vec![last_token(450)], Then::End),
                ],
                3,
            )
            .await,
        );
        outcomes.push(
            outcomes_on_c(
                local_pipeline,
                vec![
                    Turn::Stream(vec![token(450), went_away()], Then::End),
                    Turn::Stream(vec![last_token(4996)], Then::End),
                ],
                3,
            )
            .await,
        );
    }
    let new_request = vec!["new_request/success x1".to_string()];
    let ongoing = vec!["ongoing_request/success x1".to_string()];
    assert_eq!(
        outcomes,
        [new_request.clone(), ongoing.clone(), new_request, ongoing]
    );
}

/// A migration that runs out of attempts records one `failure` and nothing else, on both
/// local pipelines: the caller receives the error, and dropping the stream records no second
/// outcome.
#[tokio::test]
async fn a_failed_migration_records_one_failure_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(
            outcomes_on_c(
                local_pipeline,
                vec![
                    Turn::Stream(vec![token(450), went_away()], Then::End),
                    Turn::Unavailable,
                ],
                1,
            )
            .await,
        );
    }
    let failure = vec!["ongoing_request/failure x1".to_string()];
    assert_eq!(outcomes, [failure.clone(), failure]);
}

/// A retry whose dispatch panics unwinds a live request, with no stop or kill: Migration
/// records no outcome for it, on both local pipelines.
#[tokio::test]
async fn a_retry_dispatch_panic_is_not_a_request_cancellation_on_either_local_pipeline() {
    struct PanicOnRetry {
        eager_failure: bool,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for PanicOnRetry {
        async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
            if self.calls.fetch_add(1, Ordering::AcqRel) > 0 {
                panic!("injected retry dispatch panic");
            }
            if self.eager_failure {
                Err(worker_unavailable())
            } else {
                Ok(ResponseStream::new(
                    Box::pin(stream::iter(vec![went_away()])),
                    request.context(),
                ))
            }
        }
    }

    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        for eager_failure in [true, false] {
            let engine = Arc::new(PanicOnRetry {
                eager_failure,
                calls: AtomicUsize::new(0),
            });
            let metrics = Arc::new(Metrics::new());
            let pipeline =
                pipeline_c_with_metrics(local_pipeline, engine.clone(), 3, metrics.clone());
            let request = preprocessed();
            let context = request.context();
            let task = tokio::spawn(async move {
                let stream = pipeline.generate(request).await.unwrap();
                rest(stream).await;
            });
            let panic = task.await.expect_err("the retry must panic");
            assert!(panic.is_panic());
            let payload = panic.into_panic();
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"injected retry dispatch panic")
            );
            assert_eq!(engine.calls.load(Ordering::Acquire), 2);
            assert!(!context.is_stopped());
            assert!(!context.is_killed());
            outcomes.push(migration_outcomes(&metrics));
        }
    }
    assert_eq!(outcomes, vec![Vec::<String>::new(); 4]);
}

async fn migration_outcome_after_http_stream_drop(
    local_pipeline: LocalPipeline,
    kill_first: bool,
) -> Vec<String> {
    use dynamo_llm::http::service::disconnect::{
        create_connection_monitor, monitor_for_disconnects,
    };
    use dynamo_llm::http::service::metrics::{CancellationLabels, Endpoint};

    let (worker, mut arrivals) = Worker::new(vec![
        Turn::Stream(vec![token(450), went_away()], Then::End),
        Turn::HangUntilKilled,
    ]);
    let metrics = Arc::new(Metrics::new());
    let pipeline = pipeline_c_with_metrics(local_pipeline, worker.clone(), 3, metrics.clone());
    let request = preprocessed();
    let context = request.context();
    let response = pipeline.generate(request).await.unwrap();
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(metrics.clone()),
        CancellationLabels {
            model: card().display_name.clone(),
            endpoint: Endpoint::Generate.to_string(),
            request_type: "stream".to_string(),
        },
    )
    .await;
    let inflight = metrics.clone().create_inflight_guard(
        &card().display_name,
        Endpoint::Generate,
        true,
        context.id(),
    );
    let monitored = monitor_for_disconnects(
        response.map(|_| Ok(Default::default())),
        context.clone(),
        inflight,
        stream_handle,
    );
    connection_handle.disarm();
    drop(connection_handle);
    let reader = tokio::spawn(async move { monitored.collect::<Vec<_>>().await });
    for index in 0..2 {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
                .await
                .expect("the retry reaches its dispatch"),
            Some(index),
        );
    }
    assert!(!context.is_stopped());
    assert!(!context.is_killed());
    assert!(migration_outcomes(&metrics).is_empty());
    if kill_first {
        context.kill();
    }
    // Dropping the HTTP body drops its source and signals the disconnect monitor.
    reader.abort();
    let cancelled = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .expect("the dropped HTTP body releases its source")
        .expect_err("the body reader was aborted");
    assert!(cancelled.is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), context.killed())
        .await
        .expect("the HTTP disconnect monitor kills the request");
    assert_eq!(worker.calls(), 2);
    assert_eq!(metrics.get_client_disconnect_count(), 1);
    migration_outcomes(&metrics)
}

/// A kill delivered before the HTTP body drops records one cancelled migration.
#[tokio::test]
async fn a_kill_before_http_stream_drop_records_one_cancelled_migration() {
    for local_pipeline in BOTH {
        assert_eq!(
            migration_outcome_after_http_stream_drop(local_pipeline, true).await,
            ["ongoing_request/cancelled x1"],
            "{}",
            local_pipeline.as_str(),
        );
    }
}

/// An HTTP body dropped during a retry records one cancellation after its monitor kills the request.
#[tokio::test]
#[ignore = "R3: HTTP body drop destroys Migration before the disconnect monitor kills the request"]
async fn an_http_stream_drop_during_a_retry_records_one_cancelled_migration() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        outcomes.push(migration_outcome_after_http_stream_drop(local_pipeline, false).await);
    }
    assert_eq!(outcomes, vec![vec!["ongoing_request/cancelled x1"]; 2]);
}

/// A slow client: the caller reads a few items of an endless ready stream and then stops
/// reading. Neither local pipeline reads ahead of its caller: the worker has produced at most
/// one item more than the caller took (Migration's stream polls its source once per item).
#[tokio::test]
async fn a_slow_client_holds_the_worker_back_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![Turn::Endless]);
        let mut stream = pipeline_c(local_pipeline, worker.clone(), 3)
            .generate(preprocessed())
            .await
            .unwrap();
        for _ in 0..5 {
            next_within(&mut stream, Duration::from_secs(5))
                .await
                .expect("an item arrives")
                .expect("the stream is not over");
        }
        // Give a pipeline that reads ahead the time to do so.
        tokio::time::sleep(Duration::from_millis(50)).await;
        outcomes.push(worker.produced.load(Ordering::Acquire));
        drop(stream);
        let (_, dropped) = worker.call(0);
        assert!(
            dropped,
            "{local_pipeline:?}: dropping the stream drops the worker's"
        );
    }
    assert_eq!(outcomes, [5, 5]);
}

/// Migration holds a failed attempt's response stream until its replacement's dispatch
/// returns: it replaces the stream only then. So a replacement whose dispatch stalls keeps the
/// failed stream, on both local pipelines; it is dropped when the request ends.
#[tokio::test]
async fn a_failed_attempt_is_held_until_its_replacement_dispatch_returns_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, mut arrivals) = Worker::new(vec![
            Turn::Stream(vec![token(450), went_away()], Then::End),
            Turn::HangUntilKilled,
        ]);
        let request = preprocessed();
        let context = request.context();
        let mut stream = pipeline_c(local_pipeline, worker.clone(), 3)
            .generate(request)
            .await
            .unwrap();
        let first = next_within(&mut stream, Duration::from_secs(5))
            .await
            .expect("the first item arrives")
            .expect("the stream is not over");
        assert!(!first.is_error());
        let dropped_before_failure = worker.call(0).1;
        let reader = tokio::spawn(rest(stream));
        for index in 0..2 {
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
                    .await
                    .expect("the replacement reaches dispatch"),
                Some(index)
            );
        }
        let dropped_during_replacement = worker.call(0).1;
        context.kill();
        reader.await.unwrap();
        outcomes.push((
            local_pipeline.as_str(),
            dropped_before_failure,
            dropped_during_replacement,
            worker.call(0).1,
        ));
    }
    assert_eq!(
        outcomes,
        [("dynamo", false, false, true), ("gdl", false, false, true)]
    );
}

/// A replacement whose dispatch fails: Migration returns the error that caused the retry and
/// still holds the failed attempt's stream, which goes when the caller drops the response, on
/// both local pipelines. Only a replacement's successful dispatch replaces the stream.
#[tokio::test]
async fn a_failed_attempt_is_held_after_a_failed_replacement_dispatch_on_both_local_pipelines() {
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![
            Turn::Stream(vec![token(450), went_away()], Then::End),
            Turn::Unavailable,
        ]);
        let mut stream = pipeline_c(local_pipeline, worker.clone(), 1)
            .generate(preprocessed())
            .await
            .unwrap();
        let first = next_within(&mut stream, Duration::from_secs(5))
            .await
            .expect("the first item arrives")
            .expect("the stream is not over");
        assert!(!first.is_error());
        let failure = next_within(&mut stream, Duration::from_secs(5))
            .await
            .expect("replacement dispatch returns its error")
            .expect("the triggering error reaches the caller");
        assert!(failure.is_error());
        assert_eq!(worker.calls(), 2);
        let dropped_after_failed_dispatch = worker.call(0).1;
        drop(stream);
        outcomes.push((
            local_pipeline.as_str(),
            dropped_after_failed_dispatch,
            worker.call(0).1,
        ));
    }
    assert_eq!(outcomes, [("dynamo", false, true), ("gdl", false, true)]);
}

/// A slow client with the request plane between the pipeline and the worker: the transport
/// reads ahead of the caller, the same on both local pipelines. So the local composition adds
/// no read-ahead (`a_slow_client_holds_the_worker_back_on_both_local_pipelines`), and the
/// request plane's buffer is what bounds a remote worker's lead.
#[tokio::test]
async fn a_slow_client_through_the_request_plane_is_read_ahead_of_on_both_local_pipelines() {
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::pipeline::network::{Ingress, egress::push_router::PushRouter};
    use dynamo_runtime::{DistributedRuntime, Runtime};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let mut outcomes = Vec::new();
    for local_pipeline in BOTH {
        let (worker, _) = Worker::new(vec![Turn::Endless]);
        let endpoint = distributed
            .namespace(format!("review-backpressure-{}", local_pipeline.as_str()))
            .unwrap()
            .component("worker")
            .unwrap()
            .endpoint("generate");
        let started = tokio::time::timeout(
            Duration::from_secs(5),
            endpoint
                .clone()
                .endpoint_builder()
                .handler(Ingress::for_engine(worker.clone()).unwrap())
                .graceful_shutdown(false)
                .start_with_registration(),
        )
        .await
        .expect("the worker starts")
        .unwrap();
        let client = endpoint.client().await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), client.wait_for_instances())
            .await
            .expect("the client discovers the worker")
            .unwrap();
        let remote = Arc::new(
            PushRouter::from_client_with_options(
                client,
                RouterMode::RoundRobin,
                dynamo_llm::kv_router::plugins::push_router_options(),
            )
            .await
            .unwrap(),
        );
        let request = preprocessed();
        let context = request.context();
        let mut stream = tokio::time::timeout(
            Duration::from_secs(5),
            pipeline_c(local_pipeline, remote, 3).generate(request),
        )
        .await
        .expect("dispatch completes")
        .unwrap();
        for _ in 0..5 {
            next_within(&mut stream, Duration::from_secs(5))
                .await
                .expect("an item arrives")
                .expect("the stream is not over");
        }
        let _ = tokio::time::timeout(Duration::from_millis(200), async {
            while worker.produced.load(Ordering::Acquire) <= 5 {
                tokio::task::yield_now().await;
            }
        })
        .await;
        outcomes.push((
            local_pipeline.as_str(),
            worker.produced.load(Ordering::Acquire),
        ));
        context.kill();
        drop(stream);
        tokio::time::timeout(Duration::from_secs(5), started.shutdown())
            .await
            .expect("the worker shuts down")
            .unwrap();
    }
    runtime.shutdown();
    assert!(
        outcomes.iter().all(|(_, produced)| *produced > 5),
        "{outcomes:?}"
    );
}
