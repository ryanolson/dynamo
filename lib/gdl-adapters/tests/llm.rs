// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapters around Dynamo's own operators and HTTP service: Migration
//! retries through them, and the HTTP handler maps an engine's eager error to
//! the same status code.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_gdl_adapters::{DynOperatorStage, EngineSink, PipelineEngine, REQUEST_ID};
use dynamo_llm::http::service::axum::http::HeaderMap;
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::migration::Migration;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::extensions::{
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, session_affinity_from_headers,
};
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, Operator, ServiceBackend, ServiceFrontend,
    SingleIn, Source as _, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use gdl::{Hop, SinkStage, Source};

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

/// Migration above `engine`, composed with gdl through the adapters.
fn gdl_migration(engine: Engine) -> Engine {
    let sink = Hop::new((), SinkStage::new(Arc::new(EngineSink::new(engine))), ());
    let stage =
        DynOperatorStage::<_, PreprocessedRequest, LLMEngineOutput, _, _>::new(migration(), sink);
    Arc::new(
        PipelineEngine::<_, PreprocessedRequest, LLMEngineOutput>::new(Source::new(Hop::new(
            (),
            stage,
            (),
        ))),
    )
}

/// Migration above `engine`, composed with Dynamo's `link` chain.
fn link_migration(engine: Engine) -> Engine {
    let frontend = ServiceFrontend::<SingleIn<PreprocessedRequest>, Out>::new();
    let operator = Operator::<SingleIn<PreprocessedRequest>, Out, SingleIn<PreprocessedRequest>, Out>::into_operator(&migration());
    frontend
        .link(operator.forward_edge())
        .unwrap()
        .link(ServiceBackend::from_engine(engine))
        .unwrap()
        .link(operator.backward_edge())
        .unwrap()
        .link_terminal(frontend)
        .unwrap()
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
    let mut results = Vec::new();
    for compose in [link_migration, gdl_migration] {
        let engine = Flaky::new(1, false);
        let items: Vec<_> = compose(engine.clone())
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
        results.push(tokens(&items));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], [vec![10], vec![11]]);
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

#[tokio::test]
async fn each_attempt_reaches_the_sink_with_the_request_id() {
    // The Cx each attempt gets is derived from the attempt's own context,
    // which Migration builds with the request's id.
    let engine = Flaky::new(2, false);
    let sink = Arc::new(EngineSink::new(engine.clone() as Engine));
    let ids: Arc<Mutex<Vec<String>>> = Arc::default();
    let recorder = Recorder {
        sink,
        ids: ids.clone(),
    };
    let stage = DynOperatorStage::<_, PreprocessedRequest, LLMEngineOutput, _, _>::new(
        migration(),
        Hop::new((), SinkStage::new(Arc::new(recorder)), ()),
    );
    let pipeline = PipelineEngine::<_, PreprocessedRequest, LLMEngineOutput>::new(Source::new(
        Hop::new((), stage, ()),
    ));
    let _: Vec<_> = pipeline.generate(request()).await.unwrap().collect().await;
    assert_eq!(*ids.lock().unwrap(), ["req-1", "req-1", "req-1"]);
}

/// A gdl sink that notes the Dynamo id in each `Cx` it gets.
struct Recorder {
    sink: Arc<EngineSink<DynEngine, PreprocessedRequest, LLMEngineOutput>>,
    ids: Arc<Mutex<Vec<String>>>,
}

impl gdl::Sink for Recorder {
    type Req = SingleIn<PreprocessedRequest>;
    type Item = Annotated<LLMEngineOutput>;

    fn generate(
        &self,
        cx: gdl::Cx,
        req: SingleIn<PreprocessedRequest>,
    ) -> impl Future<Output = gdl::Result<gdl::Items<Annotated<LLMEngineOutput>>>> + Send {
        self.ids
            .lock()
            .unwrap()
            .push(cx.meta().get(REQUEST_ID).cloned().unwrap_or_default());
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

/// A kill while an attempt is still dispatching. Dynamo's `link` chain lets
/// Migration see the kill and end with its own outcome; with gdl, the kill
/// cancels the hop, which drops Migration's future and fails the call as
/// `Cancelled`. This test pins both outcomes, so a change to either shows.
#[tokio::test]
async fn a_kill_during_dispatch_ends_the_request_on_both_paths() {
    let mut outcomes = Vec::new();
    for compose in [link_migration, gdl_migration] {
        let pipeline = compose(Arc::new(Hanging));
        let request = request();
        let context = request.context();
        let call = tokio::spawn(async move { pipeline.generate(request).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        context.kill();
        let outcome = tokio::time::timeout(Duration::from_secs(5), call)
            .await
            .expect("the call ends after the kill")
            .unwrap();
        outcomes.push(match outcome {
            Ok(stream) => format!("ok, {} items", stream.collect::<Vec<_>>().await.len()),
            Err(error) => match error.downcast_ref::<DynamoError>() {
                Some(error) => format!("error {:?}", error.class()),
                None => format!("error {error:#}"),
            },
        });
    }
    assert_eq!(outcomes, ["ok, 0 items", "error Cancelled"]);
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

/// An operator that calls `next` with the request it gets.
struct Forward;

#[async_trait]
impl<Req: Data, T: Data>
    Operator<SingleIn<Req>, ManyOut<Annotated<T>>, SingleIn<Req>, ManyOut<Annotated<T>>>
    for Forward
{
    async fn generate(
        &self,
        request: SingleIn<Req>,
        next: Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>,
    ) -> Result<ManyOut<Annotated<T>>, Error> {
        next.generate(request).await
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
    let staged = PipelineEngine::<_, Req, Resp>::new(Source::new(Hop::new(
        (),
        DynOperatorStage::<_, Req, Resp, Req, Resp>::new(Arc::new(Forward), sink()),
        (),
    )));

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
    let card = ModelDeploymentCard::with_name_only("staged");
    service
        .model_manager()
        .add_chat_completions_model("staged", card.mdcsum(), Arc::new(staged))
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

    for model in ["direct", "staged"] {
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
