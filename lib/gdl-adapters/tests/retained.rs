// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl node that keeps a clone of an eager error.
//!
//! The error's source is shared among its clones, so while one is alive the
//! original `anyhow::Error` cannot be moved out on the way back, and the HTTP
//! handler's downcast to `HttpError` fails. No node in a Phase 1 pipeline
//! keeps one.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_gdl_adapters::{EngineSink, PipelineEngine};
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::engine::AsyncEngine;
use dynamo_runtime::pipeline::{Context, ManyOut, SingleIn, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;
use gdl::{Cx, Hop, Items, Sink, SinkStage, Source};

type Req = NvCreateChatCompletionRequest;
type Resp = NvCreateChatCompletionStreamResponse;
type Out = ManyOut<Annotated<Resp>>;
type DynEngine = dyn AsyncEngine<SingleIn<Req>, Out, anyhow::Error>;
type Engine = Arc<DynEngine>;

struct Forbidden;

#[async_trait]
impl AsyncEngine<SingleIn<Req>, Out, anyhow::Error> for Forbidden {
    async fn generate(&self, _: Context<Req>) -> anyhow::Result<Out> {
        Err(HttpError {
            code: 403,
            message: "access denied".to_string(),
        }
        .into())
    }
}

struct RetainError {
    sink: EngineSink<DynEngine, Req, Resp>,
    retained: Arc<Mutex<Vec<gdl::Error>>>,
}

impl Sink for RetainError {
    type Req = SingleIn<Req>;
    type Item = Annotated<Resp>;

    async fn generate(&self, cx: Cx, req: Self::Req) -> gdl::Result<Items<Self::Item>> {
        let result = self.sink.generate(cx, req).await;
        if let Err(error) = &result {
            self.retained.lock().unwrap().push(error.clone());
        }
        result
    }
}

#[tokio::test]
#[ignore = "a retained clone of an eager gdl::Error keeps its shared source, so PipelineEngine cannot return the original anyhow::Error and HTTP maps 500 instead of 403; open design question"]
async fn retaining_a_gdl_error_clone_preserves_the_http_status() {
    let engine: Engine = Arc::new(Forbidden);
    let retained = Arc::new(Mutex::new(Vec::new()));
    let control: Engine = Arc::new(PipelineEngine::<_, Req, Resp>::new(Source::new(Hop::new(
        (),
        SinkStage::new(Arc::new(EngineSink::new(engine.clone()))),
        (),
    ))));
    let held: Engine = Arc::new(PipelineEngine::<_, Req, Resp>::new(Source::new(Hop::new(
        (),
        SinkStage::new(Arc::new(RetainError {
            sink: EngineSink::new(engine),
            retained: retained.clone(),
        })),
        (),
    ))));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .port(port)
        .host("127.0.0.1")
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    for (name, pipeline) in [("control", control), ("held", held)] {
        let card = ModelDeploymentCard::with_name_only(name);
        service
            .model_manager()
            .add_chat_completions_model(name, card.mdcsum(), pipeline)
            .unwrap();
    }
    let cancel = CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let mut statuses = Vec::new();
    for model in ["control", "held"] {
        let response = client
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": false,
            }))
            .send()
            .await
            .unwrap();
        statuses.push(response.status().as_u16());
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), join)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(retained.lock().unwrap().len(), 1);
    assert_eq!(statuses[0], 403, "control without a retained clone");
    assert_eq!(statuses[1], 403, "the same error with a retained clone");
}
