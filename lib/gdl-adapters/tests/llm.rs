// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapters around Dynamo's own operators and HTTP service: Migration
//! retries through them, and the HTTP handler maps an engine's eager error to
//! the same status code.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_gdl_adapters::{Boxed, CxBridge, EngineSink, PipelineEngine, REQUEST_ID};
use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::input::{
    PreprocessedRouting, build_embeddings_pipeline, build_pipeline,
};
use dynamo_llm::http::service::axum::http::HeaderMap;
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::migration::Migration;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::operator::GdlOperator;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest};
use dynamo_llm::protocols::common::extensions::{
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, session_affinity_from_headers,
};
use dynamo_llm::protocols::common::llm_backend::{
    EmbeddingsEngineOutput, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_llm::protocols::openai::embeddings::{
    NvCreateEmbeddingRequest, NvCreateEmbeddingResponse,
};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::CancellationToken;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, RouterMode, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use gdl::{Hop, OperatorStage, SinkStage, Source};

type Out = ManyOut<Annotated<LLMEngineOutput>>;
type DynEngine = dyn AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error>;
type Engine = Arc<DynEngine>;

const MODEL: &str = "test-model";

fn preprocessed() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(MODEL.to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(StopConditions {
            max_tokens: Some(2),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![])
        .annotations(vec![])
        .build()
        .unwrap()
}

fn session(id: &str) -> SessionAffinityId {
    let mut headers = HeaderMap::new();
    headers.insert("x-dynamo-session-id", id.parse().unwrap());
    session_affinity_from_headers(&headers).unwrap()
}

/// The eager error prefill routing returns while its binding activates.
fn worker_unavailable() -> Error {
    DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message("prefill routing is still activating")
        .build()
        .into()
}

/// What one call of [`Flaky`] saw.
struct Call {
    id: String,
    session: Option<String>,
    context: Arc<dyn AsyncEngineContext>,
}

/// Fails its first `failures` calls with `worker_unavailable()`, then
/// streams two tokens. With `hold`, the stream waits for a kill after the
/// first token.
struct Flaky {
    failures: usize,
    hold: bool,
    calls: Mutex<Vec<Call>>,
}

impl Flaky {
    fn new(failures: usize, hold: bool) -> Arc<Self> {
        Arc::new(Self {
            failures,
            hold,
            calls: Mutex::default(),
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Flaky {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        let context = request.context();
        let session = request
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .unwrap()
            .map(|session| session.as_str().to_string());
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(Call {
                id: request.id().to_string(),
                session,
                context: context.clone(),
            });
            calls.len()
        };
        if call <= self.failures {
            return Err(worker_unavailable());
        }
        let token = |id| LLMEngineOutput {
            token_ids: vec![id],
            ..Default::default()
        };
        let first = stream::iter([Annotated::from_data(token(10))]);
        let hold = self.hold;
        let watched = context.clone();
        let rest = stream::once(async move {
            if hold {
                watched.killed().await;
            }
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![11],
                ..LLMEngineOutput::length()
            })
        });
        Ok(ResponseStream::new(Box::pin(first.chain(rest)), context))
    }
}

fn migration() -> Arc<Migration> {
    Migration::new(3, None, MODEL.to_string(), Arc::new(Metrics::new()))
}

/// Migration as a native gdl operator over `next`.
fn native_migration<H>(
    next: H,
) -> Hop<(), OperatorStage<GdlOperator<Migration, LLMEngineOutput>, H>, ()> {
    Hop::new(
        (),
        OperatorStage::new(Arc::new(GdlOperator::new(migration())), next),
        (),
    )
}

/// Migration above `engine` as a native gdl operator, under a `PipelineEngine::new`, as for
/// gdl-native nodes: a kill cancels the pipeline.
fn gdl_migration(engine: Engine) -> Engine {
    let sink = Hop::new((), SinkStage::new(Arc::new(EngineSink::new(engine))), ());
    Arc::new(PipelineEngine::<
        _,
        PreprocessedRequest,
        LLMEngineOutput,
        Boxed<PreprocessedRequest>,
    >::new(Source::new(native_migration(sink))))
}

fn request() -> SingleIn<PreprocessedRequest> {
    let mut request =
        Context::with_id_and_metadata(preprocessed(), "req-1".to_string(), BTreeMap::new());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session("session-123"));
    request
}

fn tokens(items: &[Annotated<LLMEngineOutput>]) -> Vec<Vec<u32>> {
    items
        .iter()
        .filter_map(|item| item.data.as_ref().map(|data| data.token_ids.clone()))
        .collect()
}

#[tokio::test]
async fn migration_retries_an_eager_worker_unavailable_through_the_adapters() {
    let engine = Flaky::new(1, false);
    let items: Vec<_> = gdl_migration(engine.clone())
        .generate(request())
        .await
        .unwrap()
        .collect()
        .await;
    let calls = engine.calls.lock().unwrap();
    assert_eq!(calls.len(), 2, "the eager error is retried once");
    for call in calls.iter() {
        assert_eq!(call.id, "req-1");
        assert_eq!(call.session.as_deref(), Some("session-123"));
    }
    assert_eq!(tokens(&items), [vec![10], vec![11]]);
}

#[tokio::test]
async fn a_stop_and_a_kill_reach_the_retried_attempt() {
    let engine = Flaky::new(1, true);
    let pipeline = gdl_migration(engine.clone());
    let request = request();
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(tokens(&[stream.next().await.unwrap()]), [vec![10]]);

    let retried = engine.calls.lock().unwrap()[1].context.clone();
    context.stop_generating();
    assert!(retried.is_stopped());
    assert!(!retried.is_killed());
    context.kill();
    assert!(retried.is_killed());
    // gdl ends the stream at the kill, without an item.
    let rest: Vec<_> = stream.collect().await;
    assert!(rest.iter().all(|item| !item.is_error()), "{rest:?}");
}

/// Each attempt runs in a child of the request's `Cx`: it has the request's id and Dynamo id,
/// and a cancel of the request's `Cx` reaches the attempt that is still streaming.
#[tokio::test]
async fn each_attempt_is_a_child_of_the_requests_cx() {
    let engine = Flaky::new(2, true);
    let sink = Arc::new(EngineSink::new(engine.clone() as Engine));
    let seen: Arc<Mutex<Vec<gdl::Cx>>> = Arc::default();
    let recorder = Recorder {
        sink,
        seen: seen.clone(),
    };
    let pipeline = Source::new(native_migration(Hop::new(
        (),
        SinkStage::new(Arc::new(recorder)),
        (),
    )));
    let request = request();
    let root = CxBridge::cx(request.context().as_ref());
    let mut items = pipeline
        .call(root.clone(), Box::new(request))
        .await
        .unwrap();
    assert_eq!(tokens(&[items.next().await.unwrap().unwrap()]), [vec![10]]);

    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen.len(),
        3,
        "two eager failures, then the attempt that streams"
    );
    for cx in &seen {
        assert_eq!(cx.id(), root.id());
        assert_eq!(cx.meta().get(REQUEST_ID).map(String::as_str), Some("req-1"));
    }
    let streaming = &seen[2];
    assert!(!streaming.is_cancelled());
    root.cancel();
    assert!(streaming.is_cancelled());
}

/// A gdl sink that keeps each `Cx` it gets.
struct Recorder {
    sink: Arc<
        EngineSink<DynEngine, PreprocessedRequest, LLMEngineOutput, Boxed<PreprocessedRequest>>,
    >,
    seen: Arc<Mutex<Vec<gdl::Cx>>>,
}

impl gdl::Sink for Recorder {
    type Req = Boxed<PreprocessedRequest>;
    type Item = Annotated<LLMEngineOutput>;

    fn generate(
        &self,
        cx: gdl::Cx,
        req: Boxed<PreprocessedRequest>,
    ) -> impl Future<Output = gdl::Result<gdl::Items<Annotated<LLMEngineOutput>>>> + Send {
        self.seen.lock().unwrap().push(cx.clone());
        gdl::Sink::generate(&*self.sink, cx, req)
    }
}

/// An engine that waits in its eager phase until the request is killed.
struct Hanging;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Hanging {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        let context = request.context();
        context.killed().await;
        Ok(ResponseStream::new(Box::pin(stream::empty()), context))
    }
}

/// A kill while an attempt is still dispatching. Under a `PipelineEngine::new`, the kill
/// cancels the hop, which drops Migration's future and fails the call as `Cancelled` (see
/// `kill_during_dispatch`).
#[tokio::test]
async fn a_kill_during_dispatch_ends_the_request() {
    let pipeline = gdl_migration(Arc::new(Hanging));
    assert_eq!(
        kill_during_dispatch(pipeline, request()).await,
        "error Cancelled"
    );
}

/// Fails as the authorization fixtures in `lib/llm/tests/http-service.rs` do.
struct AlwaysFail;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for AlwaysFail
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        Err(HttpError {
            code: 403,
            message: "Always fail".to_string(),
        })?
    }
}

/// A worker that refuses every request as the authorization fixtures do.
struct Forbidden;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Forbidden {
    async fn generate(&self, _request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        Err(HttpError {
            code: 403,
            message: "Always fail".to_string(),
        })?
    }
}

#[tokio::test]
async fn an_engines_http_error_keeps_its_status_through_the_adapters() {
    type Req = NvCreateChatCompletionRequest;
    type Resp = NvCreateChatCompletionStreamResponse;
    let engine: Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>> =
        Arc::new(AlwaysFail);
    let sink = || {
        Hop::new(
            (),
            SinkStage::new(Arc::new(EngineSink::new(engine.clone()))),
            (),
        )
    };
    let direct = PipelineEngine::<_, Req, Resp>::new(Source::new(sink()));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .port(port)
        .host("127.0.0.1")
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    let card = ModelDeploymentCard::with_name_only("direct");
    service
        .model_manager()
        .add_chat_completions_model("direct", card.mdcsum(), Arc::new(direct))
        .unwrap();
    // The whole routed pipeline on gdl, whose native Migration may retry: it must not keep the
    // terminal's eager error past its retry decision (see `tests/retained.rs`).
    let card = ModelDeploymentCard::with_name_only("routed");
    service
        .model_manager()
        .add_chat_completions_model("routed", card.mdcsum(), pipeline_b(Arc::new(Forbidden), 3))
        .unwrap();
    let cancel = CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    tokio::time::timeout(Duration::from_secs(5), async {
        while client.get(format!("{base}/health")).send().await.is_err() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the service is up");

    for model in ["direct", "routed"] {
        for stream in [true, false] {
            let response = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": stream,
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::FORBIDDEN,
                "{model}, stream {stream}"
            );
        }
    }

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), join).await;
}

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// The frontend's routed chat pipeline over `engine`, as the model watcher
/// builds it. Prefill and encoder routing are disabled, as in aggregated serving.
fn pipeline_b(
    engine: Engine,
    migration_limit: u32,
) -> Arc<
    dyn AsyncEngine<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            Error,
        >,
> {
    let card = ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap();
    let tokenizer = card.tokenizer().unwrap();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone()).unwrap();
    PreprocessedRouting::new(
        engine,
        PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None),
        EncoderRouter::disabled(),
    )
    .build_pipeline::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>(
        &card,
        preprocessor,
        tokenizer,
        migration_limit,
        None,
        Arc::new(Metrics::new()),
    )
    .unwrap()
}

fn chat_request() -> SingleIn<NvCreateChatCompletionRequest> {
    let chat: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "messages": [{"role": "user", "content": "Write one sentence about a fox."}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let mut request = Context::with_id_and_metadata(chat, "req-1".to_string(), BTreeMap::new());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session("session-123"));
    request
}

/// The streamed content, and whether any item was an error.
fn content(items: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> (String, bool) {
    let mut text = String::new();
    for item in items {
        let Some(data) = &item.data else { continue };
        let data = serde_json::to_value(data).unwrap();
        for choice in data["choices"].as_array().into_iter().flatten() {
            text.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        }
    }
    (text, items.iter().any(Annotated::is_error))
}

/// What the terminal engine saw, per call: the request id and session.
fn calls(engine: &Flaky) -> Vec<(String, Option<String>)> {
    engine
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|call| (call.id.clone(), call.session.clone()))
        .collect()
}

#[tokio::test]
async fn pipeline_b_streams_a_chat() {
    let engine = Flaky::new(0, false);
    let items: Vec<_> = pipeline_b(engine.clone(), 0)
        .generate(chat_request())
        .await
        .unwrap()
        .collect()
        .await;
    let (text, errored) = content(&items);
    assert!(!text.is_empty(), "no content in {items:?}");
    assert!(!errored, "{items:?}");
    assert_eq!(
        calls(&engine),
        [("req-1".to_string(), Some("session-123".to_string()))]
    );
}

/// The terminal's eager error crosses the prefill router, the encoder router
/// and the backend, three adapter round trips, before Migration sees it.
#[tokio::test]
async fn pipeline_b_retries_an_eager_worker_unavailable_from_its_terminal() {
    let engine = Flaky::new(1, false);
    let items: Vec<_> = pipeline_b(engine.clone(), 3)
        .generate(chat_request())
        .await
        .unwrap()
        .collect()
        .await;
    let (text, errored) = content(&items);
    assert!(!text.is_empty(), "no content in {items:?}");
    assert!(!errored, "{items:?}");
    let attempt = ("req-1".to_string(), Some("session-123".to_string()));
    assert_eq!(calls(&engine), [attempt.clone(), attempt], "retried once");
}

/// A stop and a kill of the HTTP handler's context reach the terminal
/// engine's context through all five adapted stages.
#[tokio::test]
async fn a_stop_and_a_kill_reach_the_terminal_through_pipeline_b() {
    let engine = Flaky::new(0, true);
    let pipeline = pipeline_b(engine.clone(), 0);
    let request = chat_request();
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the first item arrives")
        .unwrap();
    assert!(!first.is_error(), "{first:?}");

    let terminal = engine.calls.lock().unwrap()[0].context.clone();
    context.stop_generating();
    assert!(terminal.is_stopped());
    assert!(!terminal.is_killed());
    context.kill();
    assert!(terminal.is_killed());
    let rest: Vec<_> = tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends after the kill");
    assert!(rest.iter().all(|item| !item.is_error()), "{rest:?}");
}

/// A kill while the terminal is still dispatching, through all of pipeline B (see
/// `kill_during_dispatch`). Pipeline B is native and rooted with `PipelineEngine::new`, so the
/// kill cancels it and gdl drops the terminal's call.
#[tokio::test]
async fn a_kill_during_dispatch_ends_pipeline_b() {
    let pipeline = pipeline_b(Arc::new(Hanging), 3);
    assert_eq!(
        kill_during_dispatch(pipeline, chat_request()).await,
        "error Cancelled"
    );
}

/// What a kill while the terminal is still dispatching does to `pipeline`'s call.
///
/// On a pipeline with a Dynamo engine inside it (A and embeddings, rooted with
/// `PipelineEngine::for_dynamo_stages`), the stages see the kill through their own context and
/// end the request with an empty stream. A native gdl pipeline (B and C) is rooted with
/// `PipelineEngine::new`: the kill cancels its `Cx`, gdl drops every call still in flight, as a
/// client disconnect drops the handler's call, and the call fails as `Cancelled`. The HTTP
/// handler reports both as a client disconnect.
async fn kill_during_dispatch<Req: Data, T: Data>(
    pipeline: Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>,
    request: SingleIn<Req>,
) -> String {
    let context = request.context();
    let call = tokio::spawn(async move { pipeline.generate(request).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    context.kill();
    let outcome = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("the call ends after the kill")
        .unwrap();
    match outcome {
        Ok(stream) => format!("ok, {} items", stream.collect::<Vec<_>>().await.len()),
        Err(error) => match error.downcast_ref::<DynamoError>() {
            Some(error) => format!("error {:?}", error.class()),
            None => format!("error {error:#}"),
        },
    }
}

/// The routed pipeline for callers that bring their own pre- and post-processing (the Generate
/// engine and the Python `chat_engine_factory`), over `engine`.
fn pipeline_c(engine: Engine, migration_limit: u32) -> Engine {
    let card = ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap();
    PreprocessedRouting::new(
        engine,
        PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None),
        EncoderRouter::disabled(),
    )
    .build_preprocessed_pipeline(&card, migration_limit, None, Arc::new(Metrics::new()))
    .unwrap()
}

#[tokio::test]
async fn pipeline_c_streams_the_tokens() {
    let engine = Flaky::new(0, false);
    let items: Vec<_> = pipeline_c(engine.clone(), 0)
        .generate(request())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(items.iter().all(|item| !item.is_error()), "{items:?}");
    assert_eq!(tokens(&items), [vec![10], vec![11]]);
    assert_eq!(
        calls(&engine),
        [("req-1".to_string(), Some("session-123".to_string()))]
    );
}

/// The terminal's eager error crosses the prefill and encoder routers, two adapter round
/// trips, before Migration sees it.
#[tokio::test]
async fn pipeline_c_retries_an_eager_worker_unavailable_from_its_terminal() {
    let engine = Flaky::new(1, false);
    let items: Vec<_> = pipeline_c(engine.clone(), 3)
        .generate(request())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(items.iter().all(|item| !item.is_error()), "{items:?}");
    assert_eq!(tokens(&items), [vec![10], vec![11]]);
    let attempt = ("req-1".to_string(), Some("session-123".to_string()));
    assert_eq!(calls(&engine), [attempt.clone(), attempt], "retried once");
}

/// A stop and a kill of the caller's context reach the terminal engine's context through the
/// three adapted stages.
#[tokio::test]
async fn a_stop_and_a_kill_reach_the_terminal_through_pipeline_c() {
    let engine = Flaky::new(0, true);
    let pipeline = pipeline_c(engine.clone(), 0);
    let request = request();
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the first item arrives")
        .unwrap();
    assert_eq!(tokens(&[first]), [vec![10]]);

    let terminal = engine.calls.lock().unwrap()[0].context.clone();
    context.stop_generating();
    assert!(terminal.is_stopped());
    assert!(!terminal.is_killed());
    context.kill();
    assert!(terminal.is_killed());
    let rest: Vec<_> = tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends after the kill");
    assert!(rest.iter().all(|item| !item.is_error()), "{rest:?}");
}

#[tokio::test]
async fn a_kill_during_dispatch_ends_pipeline_c() {
    let pipeline = pipeline_c(Arc::new(Hanging), 3);
    assert_eq!(
        kill_during_dispatch(pipeline, request()).await,
        "error Cancelled"
    );
}

type EmbedOut = ManyOut<Annotated<EmbeddingsEngineOutput>>;
type EmbedEngine = Arc<dyn AsyncEngine<SingleIn<PreprocessedEmbeddingRequest>, EmbedOut, Error>>;

/// What one call of [`FlakyEmbeddings`] saw.
struct EmbedCall {
    id: String,
    token_ids: Vec<Vec<u32>>,
    context: Arc<dyn AsyncEngineContext>,
}

/// An embeddings worker that fails its first `failures` calls with `worker_unavailable()`,
/// then answers with one vector per input. With `hold`, the answer waits for a kill.
struct FlakyEmbeddings {
    failures: usize,
    hold: bool,
    calls: Mutex<Vec<EmbedCall>>,
}

impl FlakyEmbeddings {
    fn new(failures: usize, hold: bool) -> Arc<Self> {
        Arc::new(Self {
            failures,
            hold,
            calls: Mutex::default(),
        })
    }

    fn calls(&self) -> Vec<(String, Vec<Vec<u32>>)> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| (call.id.clone(), call.token_ids.clone()))
            .collect()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedEmbeddingRequest>, EmbedOut, Error> for FlakyEmbeddings {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<EmbedOut, Error> {
        let context = request.context();
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(EmbedCall {
                id: request.id().to_string(),
                token_ids: request.token_ids.clone(),
                context: context.clone(),
            });
            calls.len()
        };
        if call <= self.failures {
            return Err(worker_unavailable());
        }
        let tokens: usize = request.token_ids.iter().map(Vec::len).sum();
        let output = EmbeddingsEngineOutput {
            // Two little-endian f32s per input, base64 as the worker sends them.
            embeddings: vec!["AACAPwAAAEA=".to_string(); request.token_ids.len()],
            prompt_tokens: tokens as u32,
            total_tokens: tokens as u32,
        };
        let hold = self.hold;
        let watched = context.clone();
        let answer = stream::once(async move {
            if hold {
                watched.killed().await;
            }
            Annotated::from_data(output)
        });
        Ok(ResponseStream::new(Box::pin(answer), context))
    }
}

/// The embeddings pipeline the model watcher builds for a token-input embeddings worker, over
/// `engine` in place of its `PushRouter`.
fn pipeline_embeddings(
    engine: EmbedEngine,
) -> Arc<
    dyn AsyncEngine<
            SingleIn<NvCreateEmbeddingRequest>,
            ManyOut<Annotated<NvCreateEmbeddingResponse>>,
            Error,
        >,
> {
    let mut card = ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap();
    card.model_type = ModelType::Embedding;
    let preprocessor = OpenAIPreprocessor::new_for_embeddings(card).unwrap();
    build_embeddings_pipeline(preprocessor, engine)
}

fn embedding_request() -> SingleIn<NvCreateEmbeddingRequest> {
    let embedding: NvCreateEmbeddingRequest = serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "input": ["A fox crossed the river.", "It was cold."],
    }))
    .unwrap();
    Context::with_id_and_metadata(embedding, "req-1".to_string(), BTreeMap::new())
}

#[tokio::test]
async fn pipeline_embeddings_answers() {
    let engine = FlakyEmbeddings::new(0, false);
    let items: Vec<_> = pipeline_embeddings(engine.clone())
        .generate(embedding_request())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(items.iter().all(|item| !item.is_error()), "{items:?}");
    let answers: Vec<_> = items
        .iter()
        .map(|item| serde_json::to_value(&item.data).unwrap())
        .collect();
    let calls = engine.calls();
    assert_eq!(answers.len(), 1);
    assert_eq!(
        answers[0]["data"].as_array().unwrap().len(),
        2,
        "{answers:?}"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "req-1");
    assert_eq!(calls[0].1.len(), 2);
}

/// The embeddings pipeline has no Migration: the worker's eager error reaches the caller as
/// it left the router, through the preprocessor.
#[tokio::test]
async fn pipeline_embeddings_returns_its_terminals_eager_error() {
    let engine = FlakyEmbeddings::new(1, false);
    let error = match pipeline_embeddings(engine.clone())
        .generate(embedding_request())
        .await
    {
        Ok(_) => panic!("the eager error was lost"),
        Err(error) => error,
    };
    let error_type = error
        .downcast_ref::<DynamoError>()
        .map(DynamoError::error_type)
        .unwrap_or_else(|| panic!("{error:#}"));
    assert_eq!(error_type, ErrorClass::WorkerUnavailable);
    assert_eq!(engine.calls().len(), 1);
}

/// A stop and a kill of the HTTP handler's context reach the worker's context through both
/// adapted stages, while the worker holds its answer.
#[tokio::test]
async fn a_stop_and_a_kill_reach_the_terminal_through_pipeline_embeddings() {
    let engine = FlakyEmbeddings::new(0, true);
    let pipeline = pipeline_embeddings(engine.clone());
    let request = embedding_request();
    let context = request.context();
    let stream = pipeline.generate(request).await.unwrap();

    let terminal = engine.calls.lock().unwrap()[0].context.clone();
    context.stop_generating();
    assert!(terminal.is_stopped());
    assert!(!terminal.is_killed());
    context.kill();
    assert!(terminal.is_killed());
    let rest: Vec<_> = tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends after the kill");
    assert!(rest.iter().all(|item| !item.is_error()), "{rest:?}");
}

/// An embeddings worker that waits in its eager phase until the request is killed.
struct HangingEmbeddings;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedEmbeddingRequest>, EmbedOut, Error> for HangingEmbeddings {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<EmbedOut, Error> {
        let context = request.context();
        context.killed().await;
        Ok(ResponseStream::new(Box::pin(stream::empty()), context))
    }
}

#[tokio::test]
async fn a_kill_during_dispatch_ends_pipeline_embeddings() {
    let pipeline = pipeline_embeddings(Arc::new(HangingEmbeddings));
    assert_eq!(
        kill_during_dispatch(pipeline, embedding_request()).await,
        "ok, 0 items"
    );
}

/// The in-process pipeline the HTTP and gRPC inputs build over a token engine, over `engine`.
async fn pipeline_a(
    engine: Engine,
) -> Arc<
    dyn AsyncEngine<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            Error,
        >,
> {
    let card = ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap();
    let tokenizer = card.tokenizer().unwrap();
    build_pipeline::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>(
        &card, engine, tokenizer,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn pipeline_a_streams_a_chat() {
    let engine = Flaky::new(0, false);
    let items: Vec<_> = pipeline_a(engine.clone())
        .await
        .generate(chat_request())
        .await
        .unwrap()
        .collect()
        .await;
    let (text, errored) = content(&items);
    assert!(!text.is_empty(), "no content in {items:?}");
    assert!(!errored, "{items:?}");
    assert_eq!(
        calls(&engine),
        [("req-1".to_string(), Some("session-123".to_string()))]
    );
}

/// Pipeline A has no Migration: the terminal's eager error reaches the caller as it left the
/// engine, through the backend and the preprocessor.
#[tokio::test]
async fn pipeline_a_returns_its_terminals_eager_error() {
    let engine = Flaky::new(1, false);
    let error = match pipeline_a(engine.clone())
        .await
        .generate(chat_request())
        .await
    {
        Ok(_) => panic!("the eager error was lost"),
        Err(error) => error,
    };
    let error_type = error
        .downcast_ref::<DynamoError>()
        .map(DynamoError::error_type)
        .unwrap_or_else(|| panic!("{error:#}"));
    assert_eq!(error_type, ErrorClass::WorkerUnavailable);
    assert_eq!(calls(&engine).len(), 1);
}

/// A stop and a kill of the HTTP handler's context reach the terminal engine's context through
/// both adapted stages.
#[tokio::test]
async fn a_stop_and_a_kill_reach_the_terminal_through_pipeline_a() {
    let engine = Flaky::new(0, true);
    let pipeline = pipeline_a(engine.clone()).await;
    let request = chat_request();
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the first item arrives")
        .unwrap();
    assert!(!first.is_error(), "{first:?}");

    let terminal = engine.calls.lock().unwrap()[0].context.clone();
    context.stop_generating();
    assert!(terminal.is_stopped());
    assert!(!terminal.is_killed());
    context.kill();
    assert!(terminal.is_killed());
    let rest: Vec<_> = tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends after the kill");
    assert!(rest.iter().all(|item| !item.is_error()), "{rest:?}");
}

#[tokio::test]
async fn a_kill_during_dispatch_ends_pipeline_a() {
    let pipeline = pipeline_a(Arc::new(Hanging)).await;
    assert_eq!(
        kill_during_dispatch(pipeline, chat_request()).await,
        "ok, 0 items"
    );
}

/// Pipeline A behind the HTTP service, as the in-process HTTP input serves it: the same chat,
/// streamed and not.
#[tokio::test]
async fn pipeline_a_serves_the_same_chat_over_http_streamed_and_not() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .port(port)
        .host("127.0.0.1")
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    let engine = Flaky::new(0, false);
    let model = "gdl";
    let card = ModelDeploymentCard::with_name_only(model);
    service
        .model_manager()
        .add_chat_completions_model(model, card.mdcsum(), pipeline_a(engine.clone()).await)
        .unwrap();
    let cancel = CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    tokio::time::timeout(Duration::from_secs(5), async {
        while client.get(format!("{base}/health")).send().await.is_err() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the service is up");

    let mut texts = Vec::new();
    for stream in [false, true] {
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Write one sentence about a fox."}],
                "max_tokens": 8,
                "stream": stream,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::OK,
            "stream {stream}"
        );
        let body = response.text().await.unwrap();
        let text = if stream {
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str::<serde_json::Value>(data).unwrap())
                .filter_map(|chunk| {
                    chunk["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(str::to_string)
                })
                .collect::<String>()
        } else {
            let body: serde_json::Value = serde_json::from_str(&body).unwrap();
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        };
        assert!(!text.is_empty(), "stream {stream}: {body}");
        texts.push(text);
    }
    assert_eq!(texts[0], texts[1]);
    assert_eq!(engine.calls.lock().unwrap().len(), 2);

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), join).await;
}

/// Signals each time a request reaches it, then waits for the request's kill.
struct DispatchProbe {
    entered: tokio::sync::mpsc::UnboundedSender<()>,
}

impl DispatchProbe {
    fn new() -> (Arc<Self>, tokio::sync::mpsc::UnboundedReceiver<()>) {
        let (entered, started) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(Self { entered }), started)
    }

    async fn hold<Req: Data, T: Data>(&self, request: SingleIn<Req>) -> Result<ManyOut<T>, Error> {
        self.entered.send(()).unwrap();
        let context = request.context();
        context.killed().await;
        Ok(ResponseStream::new(Box::pin(stream::empty()), context))
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for DispatchProbe {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        self.hold(request).await
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedEmbeddingRequest>, EmbedOut, Error> for DispatchProbe {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<EmbedOut, Error> {
        self.hold(request).await
    }
}

/// What a second request with the id of one still dispatching does. A gdl composition keeps no
/// map of requests by id, so it dispatches the second. Frontend request ids are generated by the
/// server, so two in-flight requests never share one there, and a worker's ingress still
/// refuses a duplicate. (Dynamo's `link` chain kept a response map per stage keyed by request
/// id, and refused the second.)
async fn duplicate_id<Req: Data, T: Data>(
    pipeline: Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>,
    mut started: tokio::sync::mpsc::UnboundedReceiver<()>,
    request: impl Fn() -> SingleIn<Req>,
) -> String {
    let first = request();
    let first_context = first.context();
    let first_call = tokio::spawn({
        let pipeline = pipeline.clone();
        async move { pipeline.generate(first).await.map(|_| ()) }
    });
    tokio::time::timeout(Duration::from_secs(5), started.recv())
        .await
        .expect("the first request reaches the terminal")
        .unwrap();

    let second = request();
    let second_context = second.context();
    let mut second_call = tokio::spawn(async move { pipeline.generate(second).await.map(|_| ()) });
    let outcome = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            result = &mut second_call => match result.unwrap() {
                Err(error) => format!("refused: {error}"),
                Ok(()) => "returned a stream".to_string(),
            },
            _ = started.recv() => "dispatched".to_string(),
        }
    })
    .await
    .expect("the second request is refused or dispatched");

    first_context.kill();
    second_context.kill();
    for call in [first_call, second_call] {
        if !call.is_finished() {
            let _ = tokio::time::timeout(Duration::from_secs(5), call)
                .await
                .expect("the call ends after the kill");
        }
    }
    outcome
}

#[tokio::test]
async fn a_duplicate_request_id_is_dispatched_by_pipeline_a() {
    let (engine, started) = DispatchProbe::new();
    let pipeline = pipeline_a(engine).await;
    assert_eq!(
        duplicate_id(pipeline, started, chat_request).await,
        "dispatched"
    );
}

#[tokio::test]
async fn a_duplicate_request_id_is_dispatched_by_pipeline_b() {
    let (engine, started) = DispatchProbe::new();
    let pipeline = pipeline_b(engine, 0);
    assert_eq!(
        duplicate_id(pipeline, started, chat_request).await,
        "dispatched"
    );
}

#[tokio::test]
async fn a_duplicate_request_id_is_dispatched_by_pipeline_c() {
    let (engine, started) = DispatchProbe::new();
    let pipeline = pipeline_c(engine, 0);
    assert_eq!(duplicate_id(pipeline, started, request).await, "dispatched");
}

#[tokio::test]
async fn a_duplicate_request_id_is_dispatched_by_pipeline_embeddings() {
    let (engine, started) = DispatchProbe::new();
    let pipeline = pipeline_embeddings(engine);
    assert_eq!(
        duplicate_id(pipeline, started, embedding_request).await,
        "dispatched"
    );
}
