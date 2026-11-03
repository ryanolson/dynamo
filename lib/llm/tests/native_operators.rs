// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend and the preprocessor as native gdl operators (`GdlOperator`) against the same
//! operators as Dynamo operators, over the same scripted terminal engine.
//!
//! Each test runs one request both ways and compares what comes out: the items, an eager
//! error and what it downcasts to, and what the terminal saw. The gdl side is composed as
//! the frontend composes it: `PipelineEngine` above, native stages, `EngineSink` below.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use dynamo_gdl_adapters::{EngineSink, PipelineEngine};
use dynamo_llm::backend::Backend;
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::operator::GdlOperator;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest};
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
    AsyncEngineContextProvider, Context, Error, ManyOut, Operator, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::{StreamExt, stream};
use gdl::{Hop, OperatorStage, SinkStage, Source};
use serde::Serialize;

const MODEL_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";

type Engine<Req, T> = Arc<DynEngine<Req, T>>;

/// A Dynamo operator over `next`, as a Dynamo engine: the call a `link` chain makes.
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
    Hop::new((), SinkStage::new(Arc::new(EngineSink::new(engine))), ())
}

/// A native operator over the hop `next`.
fn native<O, Req, H>(op: Arc<O>, next: H) -> Hop<(), OperatorStage<GdlOperator<O, Req>, H>, ()> {
    Hop::new(
        (),
        OperatorStage::new(Arc::new(GdlOperator::new(op)), next),
        (),
    )
}

/// A gdl pipeline as the Dynamo engine HTTP calls.
fn pipeline<N, Req, T>(next: N) -> Engine<Req, T>
where
    N: gdl::Next<SingleIn<Req>, Annotated<T>>,
    Req: Data,
    T: Data,
{
    Arc::new(PipelineEngine::<_, Req, T>::new(Source::new(next)))
}

/// What one call of [`Script`] saw.
struct Call {
    id: String,
    tenant: Option<String>,
    request: PreprocessedRequest,
    context: Arc<dyn AsyncEngineContext>,
}

enum Reply {
    Items(Vec<Annotated<LLMEngineOutput>>),
    /// The items, then wait for a kill before the stream ends.
    ItemsThenHold(Vec<Annotated<LLMEngineOutput>>),
    Fail(fn() -> Error),
}

/// A worker that answers each call with `reply`, and a speculative prefill with one
/// finished token.
struct Script {
    reply: Reply,
    calls: Mutex<Vec<Call>>,
}

impl Script {
    fn new(reply: Reply) -> Arc<Self> {
        Arc::new(Self {
            reply,
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
        self.calls.lock().unwrap().push(Call {
            id: request.id().to_string(),
            tenant: request.metadata().get("tenant").cloned(),
            request: request.content().clone(),
            context: context.clone(),
        });
        if request.content().model == "speculative_prefill" {
            let mut output = LLMEngineOutput::stop();
            output.token_ids = vec![450];
            let items = stream::iter([Annotated::from_data(output)]);
            return Ok(ResponseStream::new(Box::pin(items), context));
        }
        match &self.reply {
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
        pipeline(native::<_, PreprocessedRequest, _>(
            backend,
            sink(terminal as Engine<_, _>),
        )),
    ]
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
        pipeline(native::<_, NvCreateChatCompletionRequest, _>(
            preprocessor,
            native::<_, PreprocessedRequest, _>(backend, sink(terminal)),
        )),
    ]
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
    let gdl = pipeline(native::<_, NvCreateCompletionRequest, _>(
        preprocessor,
        native::<_, PreprocessedRequest, _>(backend, sink(terminal)),
    ));
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
    let gdl = pipeline(native::<_, NvCreateEmbeddingRequest, _>(
        preprocessor,
        native::<_, PreprocessedEmbeddingRequest, _>(backend, sink(terminal)),
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
