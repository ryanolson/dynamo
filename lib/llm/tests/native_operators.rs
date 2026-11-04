// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo's operators as native gdl operators (`GdlOperator`) against the same operators as
//! Dynamo operators, over the same scripted terminal engine: Backend, the preprocessor,
//! Migration, and the encoder and prefill routers.
//!
//! Each test runs one request both ways and compares what comes out: the items, an eager
//! error and what it downcasts to, and what the terminal saw. The gdl side is composed as
//! the frontend composes it: `PipelineEngine` above, native stages (Backend and the chat and
//! completions preprocessors as around nodes of the hop below them), `EngineSink` below.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use dynamo_gdl_adapters::{EngineSink, PipelineEngine};
use dynamo_llm::backend::Backend;
use dynamo_llm::discovery::ModelManager;
use dynamo_llm::http::service::axum::http::HeaderMap;
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::migration::Migration;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::operator::GdlOperator;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::preprocessor::{
    OpenAIPreprocessor, PreprocessedEmbeddingRequest, PreprocessorNode,
};
use dynamo_llm::protocols::common::extensions::{
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, session_affinity_from_headers,
};
use dynamo_llm::protocols::common::llm_backend::{
    BackendOutput, EmbeddingsEngineOutput, FinishReason, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::types::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_llm::types::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};
use dynamo_llm::types::openai::embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, Operator, RouterMode, SingleIn,
    async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::{StreamExt, stream};
use gdl::{BackwardHalf, ForwardHalf, Hop, OperatorStage, SinkStage, Source};
use serde::Serialize;

const MODEL_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";

type Engine<Req, T> = Arc<DynEngine<Req, T>>;

/// A Dynamo operator over `next`, as a Dynamo engine: the call a `link` chain makes.
///
/// Transition: remove when the gdl pipeline is the only local pipeline.
struct Linked<O, DReq: Data, DT: Data> {
    op: Arc<O>,
    next: Engine<DReq, DT>,
}

#[async_trait]
impl<O, Req, T, DReq, DT> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>
    for Linked<O, DReq, DT>
where
    O: Operator<SingleIn<Req>, ManyOut<Annotated<T>>, SingleIn<DReq>, ManyOut<Annotated<DT>>>,
    Req: Data,
    T: Data,
    DReq: Data,
    DT: Data,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<T>>, Error> {
        self.op.generate(request, self.next.clone()).await
    }
}

/// Transition: remove when the gdl pipeline is the only local pipeline.
fn linked<O, Req, T, DReq, DT>(op: Arc<O>, next: Engine<DReq, DT>) -> Engine<Req, T>
where
    O: Operator<SingleIn<Req>, ManyOut<Annotated<T>>, SingleIn<DReq>, ManyOut<Annotated<DT>>>,
    Req: Data,
    T: Data,
    DReq: Data,
    DT: Data,
{
    Arc::new(Linked { op, next })
}

type DynEngine<Req, T> = dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>;
type SinkHop<Req, T> = Hop<(), SinkStage<EngineSink<DynEngine<Req, T>, Req, T>>, ()>;

/// The terminal as the inner end of a gdl hop.
fn sink<Req: Data, T: Data>(engine: Engine<Req, T>) -> SinkHop<Req, T> {
    Hop::new((), sink_stage(engine), ())
}

fn sink_stage<Req: Data, T: Data>(
    engine: Engine<Req, T>,
) -> SinkStage<EngineSink<DynEngine<Req, T>, Req, T>> {
    SinkStage::new(Arc::new(EngineSink::new(engine)))
}

type AroundHop<O, Req, I> =
    Hop<(ForwardHalf<GdlOperator<O, Req>>, ()), I, (BackwardHalf<GdlOperator<O, Req>>, ())>;

/// An around node of the hop to `inner`.
fn around<O, Req, I>(op: Arc<O>, inner: I) -> AroundHop<O, Req, I>
where
    GdlOperator<O, Req>: gdl::Around,
{
    let (forward, backward) = gdl::around(Arc::new(GdlOperator::new(op)));
    Hop::new((forward, ()), inner, (backward, ()))
}

/// A native operator over the hop `next`.
fn native<O, Req, H>(op: Arc<O>, next: H) -> Hop<(), OperatorStage<GdlOperator<O, Req>, H>, ()> {
    Hop::new(
        (),
        OperatorStage::new(Arc::new(GdlOperator::new(op)), next),
        (),
    )
}

/// A gdl pipeline as the Dynamo engine HTTP calls, rooted as the frontend roots one whose
/// stages include Dynamo operators and engines.
fn pipeline<N, Req, T>(next: N) -> Engine<Req, T>
where
    N: gdl::Next<SingleIn<Req>, Annotated<T>>,
    Req: Data,
    T: Data,
{
    // Transition: root with PipelineEngine::new when no Dynamo stage runs inside a gdl pipeline.
    Arc::new(PipelineEngine::<_, Req, T>::for_dynamo_stages(Source::new(
        next,
    )))
}

/// What one call of [`Script`] saw.
struct Call {
    id: String,
    tenant: Option<String>,
    session: Option<String>,
    request: PreprocessedRequest,
    context: Arc<dyn AsyncEngineContext>,
}

enum Reply {
    Items(Vec<Annotated<LLMEngineOutput>>),
    /// The items, then wait for a kill before the stream ends.
    ItemsThenHold(Vec<Annotated<LLMEngineOutput>>),
    /// The items, then wait for a stop or a kill before the stream ends.
    ItemsThenStopped(Vec<Annotated<LLMEngineOutput>>),
    Fail(fn() -> Error),
}

/// A worker that answers each call with `reply`, and a speculative prefill with one
/// finished token.
struct Script {
    /// The reply to each call in turn; the last one answers every call after it.
    replies: Vec<Reply>,
    calls: Mutex<Vec<Call>>,
}

impl Script {
    fn new(reply: Reply) -> Arc<Self> {
        Self::sequence(vec![reply])
    }

    fn sequence(replies: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies,
            calls: Mutex::default(),
        })
    }

    fn calls(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for Script
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        let reply = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(Call {
                id: request.id().to_string(),
                tenant: request.metadata().get("tenant").cloned(),
                session: request
                    .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                    .unwrap()
                    .map(|session| session.as_str().to_string()),
                request: request.content().clone(),
                context: context.clone(),
            });
            &self.replies[(calls.len() - 1).min(self.replies.len() - 1)]
        };
        if request.content().model == "speculative_prefill" {
            let mut output = LLMEngineOutput::stop();
            output.token_ids = vec![450];
            let items = stream::iter([Annotated::from_data(output)]);
            return Ok(ResponseStream::new(Box::pin(items), context));
        }
        match reply {
            Reply::Items(items) => Ok(ResponseStream::new(
                Box::pin(stream::iter(items.clone())),
                context,
            )),
            Reply::ItemsThenHold(items) => {
                let watched = context.clone();
                let held = stream::iter(items.clone()).chain(
                    stream::once(async move { watched.killed().await })
                        .filter_map(|()| async { None }),
                );
                Ok(ResponseStream::new(Box::pin(held), context))
            }
            Reply::ItemsThenStopped(items) => {
                let watched = context.clone();
                let held = stream::iter(items.clone()).chain(
                    stream::once(async move { watched.stopped().await })
                        .filter_map(|()| async { None }),
                );
                Ok(ResponseStream::new(Box::pin(held), context))
            }
            Reply::Fail(error) => Err(error()),
        }
    }
}

fn worker_unavailable() -> Error {
    DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message("prefill routing is still activating")
        .build()
        .into()
}

fn forbidden() -> Error {
    HttpError {
        code: 403,
        message: "access denied".to_string(),
    }
    .into()
}

struct Fixture {
    card: ModelDeploymentCard,
    tokenizer: dynamo_llm::tokenizers::Tokenizer,
}

impl Fixture {
    fn load() -> Self {
        let card = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        let tokenizer = card.tokenizer().unwrap();
        Self { card, tokenizer }
    }

    fn token_ids(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text).unwrap().token_ids().to_vec()
    }

    /// One output per token of `text`, the last finished with `Stop`.
    fn tokens(&self, text: &str) -> Vec<Annotated<LLMEngineOutput>> {
        let ids = self.token_ids(text);
        let last = ids.len() - 1;
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| {
                let mut output = LLMEngineOutput::stop();
                output.token_ids = vec![id];
                output.finish_reason = (i == last).then_some(FinishReason::Stop);
                Annotated::from_data(output)
            })
            .collect()
    }

    fn backend(&self) -> Arc<Backend> {
        Backend::from_tokenizer(self.tokenizer.clone())
    }

    fn chat_preprocessor(&self) -> Arc<OpenAIPreprocessor> {
        let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&self.card).unwrap();
        OpenAIPreprocessor::new_with_parts(self.card.clone(), formatter, self.tokenizer.clone())
            .unwrap()
    }

    fn completions_preprocessor(&self) -> Arc<OpenAIPreprocessor> {
        let PromptFormatter::OAI(formatter) = PromptFormatter::no_op();
        OpenAIPreprocessor::new_with_parts(self.card.clone(), formatter, self.tokenizer.clone())
            .unwrap()
    }
}

fn request<T: Data>(content: T) -> SingleIn<T> {
    let metadata = BTreeMap::from([("tenant".to_string(), "a".to_string())]);
    Context::with_id_and_metadata(content, "req-1".to_string(), metadata)
}

fn preprocessed(fixture: &Fixture, no_stop_trim: bool) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("TinyLlama_v1.1".to_string())
        .token_ids(fixture.token_ids("A fox by the river"))
        .stop_conditions(StopConditions {
            max_tokens: Some(32),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions {
            no_stop_trim: no_stop_trim.then_some(true),
            ..Default::default()
        })
        .eos_token_ids(vec![2])
        .annotations(vec![])
        .build()
        .unwrap()
}

fn chat(extra: serde_json::Value) -> NvCreateChatCompletionRequest {
    let mut body = serde_json::json!({
        "model": "TinyLlama_v1.1",
        "messages": [{"role": "user", "content": "Write one sentence about a fox."}],
        "max_tokens": 32,
        "stream": true,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

/// The items as JSON, with what differs between runs zeroed: the creation time, and the
/// latencies in the metrics annotation (a JSON string in `comment`).
fn normalized<T: Serialize>(items: &[Annotated<T>]) -> Vec<serde_json::Value> {
    fn zero_created(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, value) in map.iter_mut() {
                    if key == "created" || key.ends_with("latency") {
                        *value = serde_json::Value::from(0);
                    } else {
                        zero_created(value);
                    }
                }
            }
            serde_json::Value::Array(values) => values.iter_mut().for_each(zero_created),
            serde_json::Value::String(text) if text.contains("latency") => {
                if let Ok(mut inner) = serde_json::from_str::<serde_json::Value>(text) {
                    zero_created(&mut inner);
                    *text = inner.to_string();
                }
            }
            _ => {}
        }
    }
    items
        .iter()
        .map(|item| {
            let mut value = serde_json::to_value(item).unwrap();
            zero_created(&mut value);
            value
        })
        .collect()
}

async fn collect<T: Data>(stream: ManyOut<Annotated<T>>) -> Vec<Annotated<T>> {
    tokio::time::timeout(Duration::from_secs(10), stream.collect())
        .await
        .expect("the stream ends")
}

/// Runs `make()` both ways and returns the dynamo and gdl results.
///
/// Transition: remove the Dynamo case and keep the gdl assertions when the gdl pipeline is the only
/// local pipeline.
async fn both<Req: Data, T: Data>(
    dynamo: Engine<Req, T>,
    gdl: Engine<Req, T>,
    make: impl Fn() -> SingleIn<Req>,
) -> [Result<Vec<Annotated<T>>, Error>; 2] {
    let mut out = Vec::new();
    for engine in [dynamo, gdl] {
        out.push(match engine.generate(make()).await {
            Ok(stream) => Ok(collect(stream).await),
            Err(error) => Err(error),
        });
    }
    out.try_into().unwrap_or_else(|_| unreachable!())
}

fn backend_both(
    fixture: &Fixture,
    terminal: Arc<Script>,
) -> [Engine<PreprocessedRequest, BackendOutput>; 2] {
    let backend = fixture.backend();
    [
        linked(backend.clone(), terminal.clone()),
        backend_gdl(backend, terminal),
    ]
}

/// `backend` over `terminal` on the gdl path, as the frontend composes it.
fn backend_gdl(
    backend: Arc<Backend>,
    terminal: Arc<Script>,
) -> Engine<PreprocessedRequest, BackendOutput> {
    pipeline(around::<_, PreprocessedRequest, _>(
        backend,
        sink_stage(terminal as Engine<_, _>),
    ))
}

#[tokio::test]
async fn backend_detokenizes_the_same_both_ways() {
    let fixture = Fixture::load();
    let reply = fixture.tokens("The fox crossed the river at dawn.");
    let terminals = [
        Script::new(Reply::Items(reply.clone())),
        Script::new(Reply::Items(reply)),
    ];
    let [dynamo, _] = backend_both(&fixture, terminals[0].clone());
    let [_, gdl] = backend_both(&fixture, terminals[1].clone());
    let [d, g] = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    let (d, g) = (d.unwrap(), g.unwrap());
    let text: String = g
        .iter()
        .filter_map(|i| i.data.as_ref()?.text.clone())
        .collect();
    assert!(text.contains("fox crossed"), "{text:?}");
    assert_eq!(normalized(&d), normalized(&g));

    // The request reaches the terminal as it was sent: id, metadata and content.
    for terminal in &terminals {
        let calls = terminal.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "req-1");
        assert_eq!(calls[0].tenant.as_deref(), Some("a"));
        assert_eq!(
            calls[0].request.token_ids,
            preprocessed(&fixture, false).token_ids
        );
    }
}

#[tokio::test]
async fn backend_passes_an_error_item_on_the_same_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox");
    reply
        .last_mut()
        .unwrap()
        .data
        .as_mut()
        .unwrap()
        .finish_reason = None;
    reply.push(Annotated::from_err(
        DynamoError::builder()
            .class(ErrorClass::Internal)
            .message("worker went away")
            .build(),
    ));
    let [dynamo, gdl] = backend_both(&fixture, Script::new(Reply::Items(reply)));
    let [d, g] = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    let (d, g) = (d.unwrap(), g.unwrap());
    assert!(g.last().unwrap().is_error(), "{g:?}");
    assert_eq!(normalized(&d), normalized(&g));
}

#[tokio::test]
async fn backend_returns_the_terminals_eager_error_both_ways() {
    let fixture = Fixture::load();
    let [dynamo, gdl] = backend_both(&fixture, Script::new(Reply::Fail(worker_unavailable)));
    let results = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    let [d, g] = results.map(|result| {
        let error = result.expect_err("an eager error");
        // What Migration reads to decide on a retry.
        let found = error.downcast_ref::<DynamoError>().cloned();
        (found.map(|e| e.class()), error.to_string())
    });
    assert!(d.0.is_some(), "{d:?}");
    assert_eq!(d, g);
}

/// A Backend that cannot decode (no tokenizer) on the Dynamo path: it sends the request on
/// first and checks after, so an eager error from below wins, and with none the terminal is
/// called before the decoder's error.
///
/// Transition: remove when the Dynamo path checks before it sends the request on.
#[tokio::test]
async fn a_backend_that_cannot_decode_checks_after_dispatch_on_dynamo() {
    let fixture = Fixture::load();
    let blank = Backend::from_mdc(&ModelDeploymentCard::with_name_only("blank"));

    let failing = Script::new(Reply::Fail(worker_unavailable));
    let engine = linked::<_, _, BackendOutput, _, _>(blank.clone(), failing.clone());
    let error = engine
        .generate(request(preprocessed(&fixture, false)))
        .await
        .expect_err("an eager error");
    assert!(
        error
            .to_string()
            .contains("prefill routing is still activating"),
        "the terminal's error wins: {error:#}"
    );
    assert_eq!(failing.calls(), 1);

    let answering = Script::new(Reply::Items(fixture.tokens("fox")));
    let engine = linked::<_, _, BackendOutput, _, _>(blank, answering.clone());
    let error = engine
        .generate(request(preprocessed(&fixture, false)))
        .await
        .expect_err("the decoder's error");
    assert!(error.to_string().contains("no tokenizer"), "{error:#}");
    assert_eq!(answering.calls(), 1, "the terminal was called first");
}

/// A Backend that cannot decode (no tokenizer) on the gdl path fails before it sends the
/// request on: the terminal is not called, even one that would fail at once.
#[tokio::test]
async fn a_backend_that_cannot_decode_fails_before_dispatch_on_gdl() {
    let fixture = Fixture::load();
    let blank = Backend::from_mdc(&ModelDeploymentCard::with_name_only("blank"));
    for terminal in [
        Script::new(Reply::Fail(worker_unavailable)),
        Script::new(Reply::Items(fixture.tokens("fox"))),
    ] {
        let error = backend_gdl(blank.clone(), terminal.clone())
            .generate(request(preprocessed(&fixture, false)))
            .await
            .expect_err("the decoder's error");
        assert!(error.to_string().contains("no tokenizer"), "{error:#}");
        assert_eq!(terminal.calls(), 0, "the terminal was not called");
    }
}

/// With `no_stop_trim`, a text-only output is an error, and Backend stops the request below
/// it. On the gdl path that stop must reach the terminal's Dynamo context, not only a `Cx`.
#[tokio::test]
async fn backend_stops_the_terminal_on_a_text_only_output_both_ways() {
    let fixture = Fixture::load();
    let mut output = LLMEngineOutput::stop();
    output.text = Some("fox".to_string());
    output.finish_reason = None;
    let reply = vec![Annotated::from_data(output)];
    let terminals = [
        Script::new(Reply::ItemsThenHold(reply.clone())),
        Script::new(Reply::ItemsThenHold(reply)),
    ];
    let [dynamo, _] = backend_both(&fixture, terminals[0].clone());
    let [_, gdl] = backend_both(&fixture, terminals[1].clone());
    let [d, g] = both(dynamo, gdl, || request(preprocessed(&fixture, true))).await;
    let (d, g) = (d.unwrap(), g.unwrap());
    assert_eq!(g.len(), 1, "{g:?}");
    assert!(g[0].is_error(), "{g:?}");
    assert_eq!(normalized(&d), normalized(&g));
    for terminal in &terminals {
        let calls = terminal.calls.lock().unwrap();
        assert!(calls[0].context.is_stopped());
        assert!(!calls[0].context.is_killed());
    }
}

/// A kill of the caller's context mid-stream reaches the terminal and ends the stream.
#[tokio::test]
async fn a_kill_mid_stream_reaches_the_terminal_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox");
    reply
        .last_mut()
        .unwrap()
        .data
        .as_mut()
        .unwrap()
        .finish_reason = None;
    for (way, at) in [("dynamo", 0), ("gdl", 1)] {
        let terminal = Script::new(Reply::ItemsThenHold(reply.clone()));
        let engine = backend_both(&fixture, terminal.clone())[at].clone();
        let request = request(preprocessed(&fixture, false));
        let context = request.context();
        let mut stream = engine.generate(request).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the first item arrives")
            .unwrap();
        assert!(!first.is_error(), "{way}: {first:?}");
        context.kill();
        assert!(
            terminal.calls.lock().unwrap()[0].context.is_killed(),
            "{way}"
        );
        let rest = collect(stream).await;
        assert!(rest.iter().all(|item| !item.is_error()), "{way}: {rest:?}");
    }
}

/// A kill mid-stream through chat over Backend reaches the terminal and ends the stream. On
/// gdl the two are around nodes of one hop, so Backend's backward half runs in a scope of its
/// own, not in the hop's attempt.
#[tokio::test]
async fn a_kill_mid_stream_through_chat_reaches_the_terminal_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox crossed the river.");
    reply
        .last_mut()
        .unwrap()
        .data
        .as_mut()
        .unwrap()
        .finish_reason = None;
    for (way, at) in [("dynamo", 0), ("gdl", 1)] {
        let terminal = Script::new(Reply::ItemsThenHold(reply.clone()));
        let engine = chat_both(&fixture, terminal.clone())[at].clone();
        let request = request(chat(serde_json::json!({})));
        let context = request.context();
        let mut stream = engine.generate(request).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the first item arrives")
            .unwrap();
        assert!(!first.is_error(), "{way}: {first:?}");
        context.kill();
        assert!(
            terminal.calls.lock().unwrap()[0].context.is_killed(),
            "{way}"
        );
        let rest = tokio::time::timeout(Duration::from_secs(5), collect(stream))
            .await
            .unwrap_or_else(|_| panic!("{way}: the stream did not end"));
        assert!(rest.iter().all(|item| !item.is_error()), "{way}: {rest:?}");
    }
}

fn chat_both(
    fixture: &Fixture,
    terminal: Arc<Script>,
) -> [Engine<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>; 2] {
    let (preprocessor, backend) = (fixture.chat_preprocessor(), fixture.backend());
    let terminal: Engine<_, _> = terminal;
    [
        linked(
            preprocessor.clone(),
            linked::<_, _, BackendOutput, _, _>(backend.clone(), terminal.clone()),
        ),
        preprocessor_gdl(preprocessor, backend, terminal),
    ]
}

/// The preprocessor and `backend` over `terminal` on the gdl path, as the frontend composes
/// pipeline A: one hop to the terminal, with both as its around nodes, and the speculative
/// prefill sent to Backend's own hop.
fn preprocessor_gdl<Req: Data, Resp: Data>(
    preprocessor: Arc<OpenAIPreprocessor>,
    backend: Arc<Backend>,
    terminal: Engine<PreprocessedRequest, LLMEngineOutput>,
) -> Engine<Req, Resp>
where
    PreprocessorNode<Req>: gdl::Around<
            UpReq = SingleIn<Req>,
            UpItem = Annotated<Resp>,
            DownReq = SingleIn<PreprocessedRequest>,
            DownItem = Annotated<BackendOutput>,
        >,
{
    let terminal = sink_stage(terminal);
    let below = Arc::new(
        PipelineEngine::<_, PreprocessedRequest, BackendOutput>::for_dynamo_stages(around::<
            _,
            PreprocessedRequest,
            _,
        >(
            backend.clone(),
            terminal.clone(),
        )),
    );
    let (preprocessor_forward, preprocessor_backward) =
        gdl::around(Arc::new(PreprocessorNode::<Req>::new(preprocessor, below)));
    let (backend_forward, backend_backward) = gdl::around(Arc::new(GdlOperator::<
        _,
        PreprocessedRequest,
    >::new(backend)));
    pipeline(Hop::new(
        (preprocessor_forward, (backend_forward, ())),
        terminal,
        (backend_backward, (preprocessor_backward, ())),
    ))
}

/// The chat content of the items, and whether one finished.
fn chat_text(items: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> (String, bool) {
    let mut text = String::new();
    let mut finished = false;
    for value in normalized(items) {
        for choice in value["data"]["choices"].as_array().into_iter().flatten() {
            text.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
            finished |= !choice["finish_reason"].is_null();
        }
    }
    (text, finished)
}

#[tokio::test]
async fn chat_is_the_same_both_ways() {
    let fixture = Fixture::load();
    let reply = fixture.tokens("The fox crossed the river at dawn.");
    let terminals = [
        Script::new(Reply::Items(reply.clone())),
        Script::new(Reply::Items(reply)),
    ];
    let [dynamo, _] = chat_both(&fixture, terminals[0].clone());
    let [_, gdl] = chat_both(&fixture, terminals[1].clone());
    for extra in [
        serde_json::json!({}),
        serde_json::json!({"stream_options": {"include_usage": true}}),
        serde_json::json!({"stream": false}),
    ] {
        let [d, g] = both(dynamo.clone(), gdl.clone(), || request(chat(extra.clone()))).await;
        let (d, g) = (d.unwrap(), g.unwrap());
        let (text, finished) = chat_text(&g);
        assert!(text.contains("fox crossed") && finished, "{extra}: {g:?}");
        assert_eq!(normalized(&d), normalized(&g), "{extra}");
    }
    let [d, g] = [&terminals[0], &terminals[1]].map(|t| {
        let calls = t.calls.lock().unwrap();
        calls
            .iter()
            .map(|c| (c.id.clone(), c.tenant.clone(), c.request.token_ids.clone()))
            .collect::<Vec<_>>()
    });
    assert_eq!(d.len(), 3);
    assert_eq!(d, g);
}

#[tokio::test]
async fn chat_rejects_an_invalid_request_before_dispatch_both_ways() {
    let fixture = Fixture::load();
    let terminal = Script::new(Reply::Items(fixture.tokens("unused")));
    let [dynamo, gdl] = chat_both(&fixture, terminal.clone());
    let invalid = || {
        request(chat(
            serde_json::json!({"nvext": {"token_data": [4_000_000]}}),
        ))
    };
    let [d, g] = both(dynamo, gdl, invalid).await.map(|result| {
        let error = result.expect_err("an eager error");
        let found = error
            .chain()
            .find_map(|e| e.downcast_ref::<DynamoError>())
            .map(|e| e.error_type());
        (found, format!("{error:#}"))
    });
    assert!(d.0.is_some(), "{d:?}");
    assert_eq!(d, g);
    assert_eq!(terminal.calls(), 0);
}

/// Tool definitions that do not validate fail the chat request before it is sent on, both ways,
/// even against a terminal that would fail at once: the client sees the validation error.
#[tokio::test]
async fn invalid_tools_fail_before_an_eager_error_below_both_ways() {
    let fixture = Fixture::load();
    let terminal = Script::new(Reply::Fail(worker_unavailable));
    let [dynamo, gdl] = chat_both(&fixture, terminal.clone());
    let invalid = || request(chat(serde_json::json!({"tools": []})));
    let [d, g] = both(dynamo, gdl, invalid).await.map(|result| {
        let error = result.expect_err("an eager error");
        let found = error
            .chain()
            .find_map(|e| e.downcast_ref::<DynamoError>())
            .map(|e| e.error_type());
        (found, format!("{error:#}"))
    });
    assert!(d.1.contains("tools array cannot be empty"), "{d:?}");
    assert_eq!(d, g);
    assert_eq!(terminal.calls(), 0);
}

/// The HTTP handler picks the status by downcasting the eager error to `HttpError`, so it
/// must leave both native stages as the error the terminal returned.
#[tokio::test]
async fn the_terminals_http_error_crosses_both_native_stages() {
    let fixture = Fixture::load();
    let [dynamo, gdl] = chat_both(&fixture, Script::new(Reply::Fail(forbidden)));
    for result in both(dynamo, gdl, || request(chat(serde_json::json!({})))).await {
        let error = result.expect_err("an eager error");
        assert_eq!(
            error.downcast_ref::<HttpError>().map(|e| e.code),
            Some(403),
            "{error:#}"
        );
    }
}

/// The speculative prefill is sent after the turn's response ends, when the request's `Cx`
/// is already cancelled; on the gdl path it still reaches the terminal.
#[tokio::test]
async fn speculative_prefill_reaches_the_terminal_both_ways() {
    let fixture = Fixture::load();
    for (way, at) in [("dynamo", 0), ("gdl", 1)] {
        let terminal = Script::new(Reply::Items(fixture.tokens("The fox crossed the river.")));
        let engine = chat_both(&fixture, terminal.clone())[at].clone();
        let hint = serde_json::json!({"nvext": {"agent_hints": {"speculative_prefill": true}}});
        let items = collect(engine.generate(request(chat(hint))).await.unwrap()).await;
        assert!(chat_text(&items).1, "{way}: {items:?}");
        tokio::time::timeout(Duration::from_secs(10), async {
            while terminal.calls() < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{way}: no speculative prefill"));
        let calls = terminal.calls.lock().unwrap();
        assert_eq!(calls[1].request.model, "speculative_prefill", "{way}");
        assert_eq!(
            calls[1].request.stop_conditions.max_tokens,
            Some(1),
            "{way}"
        );
    }
}

/// A client disconnect before the turn finishes: no speculative prefill is sent. The prefill
/// waits for the turn's finish reason, and the request's end without one cancels it.
#[tokio::test]
async fn speculative_prefill_is_not_sent_after_a_disconnect_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox crossed the river.");
    for item in &mut reply {
        item.data.as_mut().unwrap().finish_reason = None;
    }
    for (way, at) in [("dynamo", 0), ("gdl", 1)] {
        let terminal = Script::new(Reply::ItemsThenHold(reply.clone()));
        let engine = chat_both(&fixture, terminal.clone())[at].clone();
        let hint = serde_json::json!({"nvext": {"agent_hints": {"speculative_prefill": true}}});
        let request = request(chat(hint));
        let context = request.context();
        let mut stream = engine.generate(request).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the first item arrives")
            .unwrap();
        // The disconnect: the HTTP handler kills the request and drops its stream.
        context.kill();
        drop(stream);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(terminal.calls(), 1, "{way}: a speculative prefill was sent");
    }
}

/// A stop string: Backend ends the response at the string and stops the worker's context,
/// which ends the worker's stream. The worker is stopped, not killed.
#[tokio::test]
async fn a_stop_string_stops_the_terminal_and_ends_the_stream_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox crossed the river at dawn.");
    for item in &mut reply {
        item.data.as_mut().unwrap().finish_reason = None;
    }
    let terminals = [
        Script::new(Reply::ItemsThenStopped(reply.clone())),
        Script::new(Reply::ItemsThenStopped(reply)),
    ];
    let [dynamo, _] = chat_both(&fixture, terminals[0].clone());
    let [_, gdl] = chat_both(&fixture, terminals[1].clone());
    let [d, g] = both(dynamo, gdl, || {
        request(chat(serde_json::json!({"stop": ["river"]})))
    })
    .await;
    let (d, g) = (d.unwrap(), g.unwrap());
    let (text, finished) = chat_text(&g);
    assert!(
        text.contains("fox crossed") && !text.contains("river") && finished,
        "{text:?}: {g:?}"
    );
    assert_eq!(normalized(&d), normalized(&g));
    for terminal in &terminals {
        let calls = terminal.calls.lock().unwrap();
        assert!(calls[0].context.is_stopped());
        assert!(!calls[0].context.is_killed());
    }
}

#[tokio::test]
async fn completions_are_the_same_both_ways() {
    let fixture = Fixture::load();
    let (preprocessor, backend) = (fixture.completions_preprocessor(), fixture.backend());
    let reply = fixture.tokens(" crossed the river at dawn.");
    let terminal: Engine<_, _> = Script::new(Reply::Items(reply));
    let dynamo: Engine<NvCreateCompletionRequest, NvCreateCompletionResponse> = linked(
        preprocessor.clone(),
        linked::<_, _, BackendOutput, _, _>(backend.clone(), terminal.clone()),
    );
    let gdl = preprocessor_gdl(preprocessor, backend, terminal);
    for extra in [
        serde_json::json!({}),
        serde_json::json!({"echo": true}),
        serde_json::json!({"stream": false}),
    ] {
        let make = || {
            let mut body = serde_json::json!({
                "model": "TinyLlama_v1.1",
                "prompt": "The fox",
                "max_tokens": 32,
                "stream": true,
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            request(serde_json::from_value::<NvCreateCompletionRequest>(body).unwrap())
        };
        let [d, g] = both(dynamo.clone(), gdl.clone(), make).await;
        let (d, g) = (d.unwrap(), g.unwrap());
        let text = serde_json::to_string(&normalized(&g)).unwrap();
        assert!(text.contains("crossed"), "{extra}: {text}");
        assert_eq!(normalized(&d), normalized(&g), "{extra}");
    }
}

/// An embeddings worker that returns one vector per input.
struct Embeddings;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
        Error,
    > for Embeddings
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<EmbeddingsEngineOutput>>, Error> {
        let (request, context) = request.into_parts();
        // As the worker sends it: base64 of little-endian f32.
        let raw: Vec<u8> = (0..8u32)
            .flat_map(|i| (i as f32 / 8.0).to_le_bytes())
            .collect();
        let vector = base64::engine::general_purpose::STANDARD.encode(raw);
        let tokens: usize = request.token_ids.iter().map(Vec::len).sum();
        let output = EmbeddingsEngineOutput {
            embeddings: request.token_ids.iter().map(|_| vector.clone()).collect(),
            prompt_tokens: tokens as u32,
            total_tokens: tokens as u32,
        };
        let items = stream::iter([Annotated::from_data(output)]);
        Ok(ResponseStream::new(Box::pin(items), context.context()))
    }
}

/// Embeddings give the same response on Dynamo's chain, with Backend, and on gdl's, without
/// it: Backend passes an embeddings response through unchanged.
#[tokio::test]
async fn embeddings_are_the_same_both_ways() {
    let fixture = Fixture::load();
    let mut card = fixture.card.clone();
    card.model_type = ModelType::Embedding;
    let preprocessor = OpenAIPreprocessor::new_for_embeddings(card.clone()).unwrap();
    let backend = Backend::from_mdc(&card);
    let terminal: Engine<PreprocessedEmbeddingRequest, EmbeddingsEngineOutput> =
        Arc::new(Embeddings);
    let dynamo: Engine<NvCreateEmbeddingRequest, NvCreateEmbeddingResponse> = linked(
        preprocessor.clone(),
        linked::<_, _, EmbeddingsEngineOutput, _, _>(backend.clone(), terminal.clone()),
    );
    // The gdl chain has no Backend, a pass-through for embeddings, as the frontend composes it.
    let gdl = pipeline(native::<_, NvCreateEmbeddingRequest, _>(
        preprocessor,
        sink(terminal),
    ));
    for format in [serde_json::json!("float"), serde_json::json!("base64")] {
        let make = || {
            request(
                serde_json::from_value::<NvCreateEmbeddingRequest>(serde_json::json!({
                    "model": "TinyLlama_v1.1",
                    "input": ["A fox.", "A river."],
                    "encoding_format": format,
                }))
                .unwrap(),
            )
        };
        let [d, g] = both(dynamo.clone(), gdl.clone(), make).await;
        let (d, g) = (d.unwrap(), g.unwrap());
        assert_eq!(g.len(), 1, "{format}: {g:?}");
        assert!(!g[0].is_error(), "{format}: {g:?}");
        assert_eq!(normalized(&d), normalized(&g), "{format}");
    }
}

fn migration(limit: u32) -> Arc<Migration> {
    Migration::new(
        limit,
        None,
        "TinyLlama_v1.1".to_string(),
        Arc::new(Metrics::new()),
    )
}

/// Migration over `terminal`, on Dynamo's chain and as a native gdl operator.
fn migration_both(
    limit: u32,
    terminals: [Arc<Script>; 2],
) -> [Engine<PreprocessedRequest, LLMEngineOutput>; 2] {
    let [dynamo, gdl] = terminals;
    type Pre = PreprocessedRequest;
    type Out = LLMEngineOutput;
    [
        linked::<_, Pre, Out, Pre, Out>(migration(limit), dynamo),
        pipeline(native::<_, Out, _>(
            migration(limit),
            sink(gdl as Engine<_, _>),
        )),
    ]
}

fn with_session(mut request: SingleIn<PreprocessedRequest>) -> SingleIn<PreprocessedRequest> {
    let mut headers = HeaderMap::new();
    headers.insert("x-dynamo-session-id", "session-123".parse().unwrap());
    request.insert(
        SESSION_AFFINITY_CONTEXT_KEY,
        session_affinity_from_headers(&headers).unwrap(),
    );
    request
}

/// What the terminal saw of one call: id, tenant, session, prompt and token budget.
type Attempt = (
    String,
    Option<String>,
    Option<String>,
    Vec<u32>,
    Option<u32>,
);

/// What the terminal saw of each call.
fn attempts(terminal: &Script) -> Vec<Attempt> {
    terminal
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|call| {
            (
                call.id.clone(),
                call.tenant.clone(),
                call.session.clone(),
                call.request.token_ids.to_vec(),
                call.request.stop_conditions.max_tokens,
            )
        })
        .collect()
}

#[tokio::test]
async fn migration_retries_an_eager_worker_unavailable_the_same_both_ways() {
    let fixture = Fixture::load();
    let replies = || {
        Script::sequence(vec![
            Reply::Fail(worker_unavailable),
            Reply::Items(fixture.tokens("The fox")),
        ])
    };
    let terminals = [replies(), replies()];
    let [dynamo, gdl] = migration_both(3, terminals.clone());
    let [d, g] = both(dynamo, gdl, || {
        with_session(request(preprocessed(&fixture, false)))
    })
    .await;
    let (d, g) = (d.unwrap(), g.unwrap());
    assert!(!g.is_empty());
    assert_eq!(normalized(&d), normalized(&g));
    let [d, g] = terminals.map(|terminal| attempts(&terminal));
    assert_eq!(d.len(), 2, "{d:?}");
    assert_eq!(d[1].2.as_deref(), Some("session-123"));
    assert_eq!(d, g);
}

/// A migratable error item mid-stream: Migration sends the delivered tokens again, with the
/// rest of the token budget, on both paths.
#[tokio::test]
async fn migration_retries_a_stream_error_the_same_both_ways() {
    let fixture = Fixture::load();
    let mut first = fixture.tokens("The fox");
    first
        .last_mut()
        .unwrap()
        .data
        .as_mut()
        .unwrap()
        .finish_reason = None;
    first.push(Annotated::from_err(
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("worker went away")
            .build(),
    ));
    let replies = || {
        Script::sequence(vec![
            Reply::Items(first.clone()),
            Reply::Items(fixture.tokens("crossed the river.")),
        ])
    };
    let terminals = [replies(), replies()];
    let [dynamo, gdl] = migration_both(3, terminals.clone());
    let [d, g] = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    let (d, g) = (d.unwrap(), g.unwrap());
    assert!(g.iter().all(|item| !item.is_error()), "{g:?}");
    assert_eq!(normalized(&d), normalized(&g));
    let [d, g] = terminals.map(|terminal| attempts(&terminal));
    assert_eq!(d.len(), 2, "{d:?}");
    assert!(
        d[1].3.len() > d[0].3.len(),
        "the retry replays the delivered tokens"
    );
    assert_eq!(d, g);
}

/// An eager error Migration does not retry leaves the native operator as the error the
/// terminal raised: Migration keeps no clone of it, so `PipelineEngine` gives back the
/// original `anyhow::Error`, and the HTTP handler still finds its `HttpError`.
#[tokio::test]
async fn migration_returns_an_eager_error_it_does_not_retry_as_itself_both_ways() {
    let fixture = Fixture::load();
    let terminals = [
        Script::new(Reply::Fail(forbidden)),
        Script::new(Reply::Fail(forbidden)),
    ];
    let [dynamo, gdl] = migration_both(3, terminals.clone());
    let results = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    for result in results {
        let error = result.expect_err("an eager error");
        let status = error
            .downcast::<HttpError>()
            .expect("the terminal's HttpError");
        assert_eq!(status.code, 403);
    }
    assert_eq!(terminals.map(|terminal| terminal.calls()), [1, 1]);
}

/// With its retries spent, Migration returns the last attempt's error, still as itself.
#[tokio::test]
async fn migration_returns_the_last_eager_error_when_its_retries_run_out_both_ways() {
    let fixture = Fixture::load();
    let terminals = [
        Script::new(Reply::Fail(worker_unavailable)),
        Script::new(Reply::Fail(worker_unavailable)),
    ];
    let [dynamo, gdl] = migration_both(1, terminals.clone());
    let results = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    for result in results {
        let error = result.expect_err("an eager error");
        let error = error
            .downcast::<DynamoError>()
            .expect("the terminal's DynamoError");
        assert_eq!(error.reason().as_str(), "backend.worker_unavailable");
    }
    assert_eq!(terminals.map(|terminal| terminal.calls()), [2, 2]);
}

/// The encoder router over the prefill router over `terminal`, both disabled as in
/// aggregated serving, on Dynamo's chain and as native gdl operators.
fn routers_both(terminals: [Arc<Script>; 2]) -> [Engine<PreprocessedRequest, LLMEngineOutput>; 2] {
    let [dynamo, gdl] = terminals;
    let prefill = || PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
    type Pre = PreprocessedRequest;
    type Out = LLMEngineOutput;
    [
        linked::<_, Pre, Out, Pre, Out>(
            EncoderRouter::disabled(),
            linked::<_, Pre, Out, Pre, Out>(prefill(), dynamo),
        ),
        pipeline(native::<_, Pre, _>(
            EncoderRouter::disabled(),
            native::<_, Pre, _>(prefill(), sink(gdl as Engine<_, _>)),
        )),
    ]
}

#[tokio::test]
async fn the_routers_pass_a_request_through_the_same_both_ways() {
    let fixture = Fixture::load();
    let mut reply = fixture.tokens("The fox");
    reply.insert(
        1,
        Annotated::from_err(
            DynamoError::builder()
                .class(ErrorClass::Internal)
                .message("a worker error item")
                .build(),
        ),
    );
    let terminals = [
        Script::new(Reply::Items(reply.clone())),
        Script::new(Reply::Items(reply)),
    ];
    let [dynamo, gdl] = routers_both(terminals.clone());
    let [d, g] = both(dynamo, gdl, || {
        with_session(request(preprocessed(&fixture, false)))
    })
    .await;
    let (d, g) = (d.unwrap(), g.unwrap());
    assert!(g[1].is_error(), "{g:?}");
    assert_eq!(normalized(&d), normalized(&g));
    let [d, g] = terminals.map(|terminal| attempts(&terminal));
    assert_eq!(d.len(), 1);
    assert_eq!(d, g);
}

#[tokio::test]
async fn the_routers_return_the_terminals_eager_error_as_itself_both_ways() {
    let fixture = Fixture::load();
    let [dynamo, gdl] = routers_both([
        Script::new(Reply::Fail(forbidden)),
        Script::new(Reply::Fail(forbidden)),
    ]);
    let results = both(dynamo, gdl, || request(preprocessed(&fixture, false))).await;
    for result in results {
        let error = result.expect_err("an eager error");
        assert_eq!(error.downcast::<HttpError>().unwrap().code, 403);
    }
}
