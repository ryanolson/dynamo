// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The closing output of the frontend's own chat pipelines when the worker's stream ends
//! cleanly: written for a whole answer, and not for one cut short by a kill. Run on the
//! in-process pipeline (A) and on the routed one (B). A kill reaches neither as a cancel of the
//! `Cx`, so a node's end check reads the request's context.

use std::collections::BTreeMap;
use std::sync::Arc;

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::input::{PreprocessedRouting, build_pipeline};
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, RouterMode, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

type Out = ManyOut<Annotated<LLMEngineOutput>>;
type Chat = NvCreateChatCompletionRequest;
type ChatOut = NvCreateChatCompletionStreamResponse;
type Pipeline = Arc<dyn AsyncEngine<SingleIn<Chat>, ManyOut<Annotated<ChatOut>>, Error>>;

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// A worker that sends two tokens, the second with a finish, and ends its stream. Before the
/// end it kills `kill`, the request's context, when given: a client that leaves as the answer
/// ends.
struct Worker {
    kill: Option<Arc<dyn AsyncEngineContext>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Worker {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        let items = [
            LLMEngineOutput {
                token_ids: vec![450],
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![4996],
                ..LLMEngineOutput::length()
            },
        ]
        .map(Annotated::from_data);
        let kill = self.kill.clone();
        let end = stream::poll_fn(move |_| {
            if let Some(context) = &kill {
                context.kill();
            }
            std::task::Poll::Ready(None)
        });
        Ok(ResponseStream::new(
            Box::pin(stream::iter(items).chain(end)),
            request.context(),
        ))
    }
}

fn card() -> ModelDeploymentCard {
    ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap()
}

/// The in-process pipeline over a token engine (A): preprocessor, Backend, engine.
async fn pipeline_a(worker: Worker) -> Pipeline {
    let card = card();
    let tokenizer = card.tokenizer().unwrap();
    build_pipeline::<Chat, ChatOut>(&card, Arc::new(worker), tokenizer)
        .await
        .unwrap()
}

/// The routed chat pipeline (B): preprocessor, Migration, Backend, the routers (disabled, as in
/// aggregated serving), the worker.
async fn pipeline_b(worker: Worker) -> Pipeline {
    let card = card();
    let tokenizer = card.tokenizer().unwrap();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone()).unwrap();
    PreprocessedRouting::new(
        Arc::new(worker),
        PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None),
        EncoderRouter::disabled(),
    )
    .build_pipeline::<Chat, ChatOut>(
        &card,
        preprocessor,
        tokenizer,
        0,
        None,
        Arc::new(Metrics::new()),
    )
    .unwrap()
}

fn chat() -> SingleIn<Chat> {
    let chat: Chat = serde_json::from_value(serde_json::json!({
        "model": "TinyLlama_v1.1",
        "messages": [{"role": "user", "content": "Write one sentence about a fox."}],
        "max_tokens": 512,
        "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .unwrap();
    Context::with_id_and_metadata(chat, "req-1".to_string(), BTreeMap::new())
}

/// The closing output of a response: its finish reasons, usage chunks and error items.
#[derive(Debug, Default, PartialEq)]
struct Closing {
    finish_reasons: Vec<String>,
    usage_chunks: usize,
    errors: usize,
}

async fn closing(pipeline: Pipeline, request: SingleIn<Chat>) -> Closing {
    let items: Vec<_> = pipeline.generate(request).await.unwrap().collect().await;
    let mut closing = Closing::default();
    for item in &items {
        closing.errors += usize::from(item.error.is_some());
        let Some(data) = &item.data else { continue };
        closing.usage_chunks += usize::from(data.inner.usage.is_some());
        for choice in &data.inner.choices {
            if let Some(reason) = choice.finish_reason {
                closing.finish_reasons.push(format!("{reason:?}"));
            }
        }
    }
    closing
}

fn whole() -> Closing {
    Closing {
        finish_reasons: vec!["Length".to_string()],
        usage_chunks: 1,
        errors: 0,
    }
}

/// The finish the worker sent passes; the usage chunk written at the end does not, and the
/// stream still ends quietly.
fn cut() -> Closing {
    Closing {
        usage_chunks: 0,
        ..whole()
    }
}

#[tokio::test]
async fn pipeline_a_closes_a_whole_answer_and_not_a_killed_one() {
    assert_eq!(
        closing(pipeline_a(Worker { kill: None }).await, chat()).await,
        whole()
    );
    let request = chat();
    let kill = Some(request.context());
    assert_eq!(
        closing(pipeline_a(Worker { kill }).await, request).await,
        cut()
    );
}

#[tokio::test]
async fn pipeline_b_closes_a_whole_answer_and_not_a_killed_one() {
    assert_eq!(
        closing(pipeline_b(Worker { kill: None }).await, chat()).await,
        whole()
    );
    let request = chat();
    let kill = Some(request.context());
    assert_eq!(
        closing(pipeline_b(Worker { kill }).await, request).await,
        cut()
    );
}
