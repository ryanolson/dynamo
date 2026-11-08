// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a gdl ending reaches the caller: the last item and the closing output.
//!
//! A gdl stream can end with an `Err` item: a `Cx` deadline or cancel, or a node error. The
//! pipeline's nodes and its exit read that item by one rule (`last_item` in `engine.rs`):
//! the stream ends with a last error item, or with no item when the error is `Cancelled` and
//! the request was killed or its root `Cx` cancelled. These tests pin what the caller receives
//! today, so a change in how gdl carries its endings shows up here.
//!
//! The pipeline is the in-process chat pipeline (A) as `build_pipeline` composes it, from the
//! same parts, with two seams the frontend does not have: a root that sets a deadline or a
//! cancel on the request's `Cx`, and a terminal that can be a gdl sink.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use dynamo_gdl_adapters::{AnnotatedItems, Boxed, CheckedItems, EngineSink, PipelineEngine};
use dynamo_llm::backend::Backend;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::operator::GdlOperator;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::preprocessor::{OpenAIPreprocessor, PreprocessorNode};
use dynamo_llm::protocols::common::llm_backend::{
    BackendOutput, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt, stream};
use gdl::{Cx, Items};

type Pre = PreprocessedRequest;
type Out = ManyOut<Annotated<LLMEngineOutput>>;
type Chat = NvCreateChatCompletionRequest;
type ChatOut = NvCreateChatCompletionStreamResponse;
type Pipeline = Arc<dyn AsyncEngine<SingleIn<Chat>, ManyOut<Annotated<ChatOut>>, Error>>;

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// Long enough for the worker's ready tokens to pass before the deadline or cancel, and short
/// enough to keep the suite fast.
const MID_STREAM: Duration = Duration::from_millis(100);

fn card() -> ModelDeploymentCard {
    ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap()
}

/// The worker's answer: two tokens TinyLlama decodes to text, the second with a finish. The
/// preprocessor writes its usage chunk at the end only after a finish, so the ending that
/// follows decides whether the caller sees one.
fn answer() -> [Annotated<LLMEngineOutput>; 2] {
    [
        LLMEngineOutput {
            token_ids: vec![450],
            ..Default::default()
        },
        LLMEngineOutput {
            token_ids: vec![4996],
            ..LLMEngineOutput::length()
        },
    ]
    .map(Annotated::from_data)
}

/// What the root does to the request's `Cx` before the pipeline sees it.
#[derive(Clone, Copy, Debug)]
enum RootEdit {
    None,
    /// A deadline that passes mid-stream.
    Deadline,
    /// A cancel of the root `Cx` mid-stream.
    Cancel,
}

/// The pipeline's root: a gdl `Next` in front of the `Source`, where an edge that sets a
/// request deadline or cancels the request through gdl would be.
#[derive(Clone)]
struct Root<N> {
    next: N,
    edit: RootEdit,
}

impl<N, Req, Item> gdl::Next<Req, Item> for Root<N>
where
    N: gdl::Next<Req, Item>,
    Req: Send + 'static,
    Item: Send + 'static,
{
    fn generate(&self, cx: Cx, req: Req) -> impl Future<Output = gdl::Result<Items<Item>>> + Send {
        let cx = match self.edit {
            RootEdit::None => cx,
            RootEdit::Deadline => cx.with_timeout(MID_STREAM),
            RootEdit::Cancel => {
                let cancel = cx.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(MID_STREAM).await;
                    cancel.cancel();
                });
                cx
            }
        };
        self.next.generate(cx, req)
    }
}

/// Pipeline A as `build_pipeline` composes it, over `sink`, under `root`.
fn pipeline_a<S>(sink: S, edit: RootEdit) -> Pipeline
where
    S: gdl::Sink<Req = Boxed<Pre>, Item = Annotated<LLMEngineOutput>>,
{
    use gdl::{Hop, SinkStage};

    let card = card();
    let tokenizer = card.tokenizer().unwrap();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone()).unwrap();
    let backend = Backend::from_tokenizer(tokenizer);

    let engine = SinkStage::new(Arc::new(sink));
    let below = {
        let (forward, backward) =
            gdl::around(Arc::new(GdlOperator::<_, Pre>::new(backend.clone())));
        Arc::new(
            PipelineEngine::<_, Pre, BackendOutput, Boxed<Pre>>::for_dynamo_stages(Hop::new(
                (forward, ()),
                engine.clone(),
                (backward, ()),
            )),
        )
    };
    let (preprocessor_forward, preprocessor_backward) =
        gdl::around(Arc::new(PreprocessorNode::<Chat>::new(preprocessor, below)));
    let (backend_forward, backend_backward) =
        gdl::around(Arc::new(GdlOperator::<_, Pre>::new(backend)));
    let hop = Hop::new(
        (preprocessor_forward, (backend_forward, ())),
        engine,
        (backend_backward, (preprocessor_backward, ())),
    );
    Arc::new(
        PipelineEngine::<_, Chat, ChatOut, Boxed<Chat>>::for_dynamo_stages(Root {
            next: gdl::Source::new(hop),
            edit,
        }),
    )
}

/// A Dynamo worker that sends [`answer`], kills `kill` when given, and then holds its stream
/// open: a worker still generating when the request ends around it.
struct Holding {
    kill: Option<Arc<dyn AsyncEngineContext>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<Pre>, Out, Error> for Holding {
    async fn generate(&self, request: SingleIn<Pre>) -> Result<Out, Error> {
        let kill = self.kill.clone();
        let tail = stream::once(async move {
            if let Some(context) = kill {
                context.kill();
            }
        })
        .filter_map(|()| async { None });
        let items = stream::iter(answer()).chain(tail).chain(stream::pending());
        Ok(ResponseStream::new(Box::pin(items), request.context()))
    }
}

fn holding(
    kill: Option<Arc<dyn AsyncEngineContext>>,
) -> EngineSink<Holding, Pre, LLMEngineOutput, Boxed<Pre>> {
    EngineSink::new(Arc::new(Holding { kill }))
}

/// How a gdl worker ends its stream after [`answer`].
#[derive(Clone, Copy, Debug)]
enum GdlEnd {
    /// End cleanly: the control, a whole answer.
    Clean,
    /// Cancel its own `Cx`, then end: a node below a backward node cancels its own scope.
    CancelOwnScope,
    /// End with a gdl node error.
    NodeError,
}

/// A gdl-native worker: sends [`answer`], kills `kill` when given, and ends as `end` says.
struct GdlWorker {
    end: GdlEnd,
    kill: Option<Arc<dyn AsyncEngineContext>>,
}

impl gdl::Sink for GdlWorker {
    type Req = Boxed<Pre>;
    type Item = Annotated<LLMEngineOutput>;

    async fn generate(&self, cx: Cx, _req: Boxed<Pre>) -> gdl::Result<Items<Self::Item>> {
        let (end, kill) = (self.end, self.kill.clone());
        let tail = stream::once(async move {
            if let Some(context) = kill {
                context.kill();
            }
            match end {
                GdlEnd::Clean => None,
                GdlEnd::CancelOwnScope => {
                    cx.cancel();
                    None
                }
                GdlEnd::NodeError => Some(Err(gdl::Error::node("worker failed"))),
            }
        })
        .filter_map(|end| async { end });
        Ok(gdl::items(stream::iter(answer().map(Ok)).chain(tail)))
    }
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

/// One response item, as the caller tells it apart: an error item with its class, a usage
/// chunk, a chunk with a finish reason, a chunk with text, or an empty chunk.
fn describe(item: &Annotated<ChatOut>) -> String {
    if let Some(error) = &item.error {
        return format!("error {:?}", error.class());
    }
    if item.event.as_deref() == Some("error") {
        return format!("error event {:?}", item.comment);
    }
    let Some(data) = &item.data else {
        return "no data".to_string();
    };
    if data.inner.usage.is_some() {
        return "usage".to_string();
    }
    let choices = &data.inner.choices;
    if let Some(reason) = choices.iter().find_map(|choice| choice.finish_reason) {
        return format!("finish {reason:?}");
    }
    let text = choices
        .iter()
        .any(|choice| matches!(&choice.delta.content, Some(content) if !format!("{content:?}").is_empty()));
    if text { "text" } else { "empty" }.to_string()
}

/// Every item of the response, described. The response must end within 10 s.
async fn response(pipeline: Pipeline, request: SingleIn<Chat>) -> Vec<String> {
    let stream = pipeline.generate(request).await.unwrap();
    tokio::time::timeout(
        Duration::from_secs(10),
        stream.map(|item| describe(&item)).collect(),
    )
    .await
    .expect("the response ends")
}

fn kill_of(request: &SingleIn<Chat>) -> Option<Arc<dyn AsyncEngineContext>> {
    Some(request.context())
}

/// The control: a clean end through the same composition closes the finished answer with a
/// usage chunk, so its absence below is the ending's doing, not the harness's.
#[tokio::test]
async fn a_clean_end_closes_with_a_usage_chunk() {
    let worker = GdlWorker {
        end: GdlEnd::Clean,
        kill: None,
    };
    let items = response(pipeline_a(worker, RootEdit::None), chat()).await;
    assert_eq!(items, ["text", "finish Length", "usage"]);
}

// An ending of the request's root `Cx`.

#[tokio::test]
async fn a_root_deadline_mid_stream_ends_with_a_deadline_error_item() {
    let items = response(pipeline_a(holding(None), RootEdit::Deadline), chat()).await;
    assert_eq!(items, ["text", "finish Length", "error DeadlineExceeded"]);
}

/// The root `Cx` cancelled is a kill to the exit: the stream just ends. A root ending cuts the
/// request short, so neither case writes the usage chunk.
#[tokio::test]
async fn a_root_cancel_mid_stream_ends_with_no_item() {
    let items = response(pipeline_a(holding(None), RootEdit::Cancel), chat()).await;
    assert_eq!(items, ["text", "finish Length"]);
}

// An ending from the worker, which Backend's input reads first.

/// The worker's own cancel is not a kill of the request: the caller gets the cancel as an
/// error item. The preprocessor's end check then sees neither a kill nor an ending of the root
/// `Cx`, so it still writes the usage chunk, after the error item.
#[tokio::test]
async fn a_worker_that_cancels_its_own_scope_ends_with_a_cancelled_error_item() {
    let worker = GdlWorker {
        end: GdlEnd::CancelOwnScope,
        kill: None,
    };
    let items = response(pipeline_a(worker, RootEdit::None), chat()).await;
    assert_eq!(items, ["text", "finish Length", "error Cancelled", "usage"]);
}

/// As the worker's own cancel: the error item, then the usage chunk.
#[tokio::test]
async fn a_worker_node_error_ends_with_an_internal_error_item() {
    let worker = GdlWorker {
        end: GdlEnd::NodeError,
        kill: None,
    };
    let items = response(pipeline_a(worker, RootEdit::None), chat()).await;
    assert_eq!(items, ["text", "finish Length", "error Internal", "usage"]);
}

// A Dynamo kill together with a gdl ending: only a cancel is silenced, and the kill cuts the
// request short, so no usage chunk follows.

#[tokio::test]
async fn a_kill_with_a_root_deadline_still_ends_with_the_deadline_error_item() {
    let request = chat();
    let worker = holding(kill_of(&request));
    let items = response(pipeline_a(worker, RootEdit::Deadline), request).await;
    assert_eq!(items, ["text", "finish Length", "error DeadlineExceeded"]);
}

#[tokio::test]
async fn a_kill_with_a_worker_node_error_still_ends_with_the_error_item() {
    let request = chat();
    let worker = GdlWorker {
        end: GdlEnd::NodeError,
        kill: kill_of(&request),
    };
    let items = response(pipeline_a(worker, RootEdit::None), request).await;
    assert_eq!(items, ["text", "finish Length", "error Internal"]);
}

#[tokio::test]
async fn a_kill_with_a_worker_that_cancels_its_own_scope_ends_with_no_item() {
    let request = chat();
    let worker = GdlWorker {
        end: GdlEnd::CancelOwnScope,
        kill: kill_of(&request),
    };
    let items = response(pipeline_a(worker, RootEdit::None), request).await;
    assert_eq!(items, ["text", "finish Length"]);
}

// `last_item`'s rule, read directly at a node's input (`CheckedItems`, and `AnnotatedItems` as a
// node reads it) and at the exit (`AnnotatedItems`, as `PipelineEngine` reads its root).

/// One item, then `end`.
fn ended(end: gdl::Error) -> Items<Annotated<u32>> {
    gdl::items(stream::iter([Ok(Annotated::from_data(1u32)), Err(end)]))
}

/// The request's state when its stream ends.
#[derive(Clone, Copy, Debug)]
struct State {
    killed: bool,
    root_cancelled: bool,
}

const LIVE: State = State {
    killed: false,
    root_cancelled: false,
};
const KILLED: State = State {
    killed: true,
    root_cancelled: false,
};
const ROOT_CANCELLED: State = State {
    killed: false,
    root_cancelled: true,
};

fn parts(state: State) -> (Cx, Arc<dyn AsyncEngineContext>) {
    let context = Context::new(()).context();
    if state.killed {
        context.kill();
    }
    let root = Cx::new();
    if state.root_cancelled {
        root.cancel();
    }
    (root, context)
}

fn describe_u32(item: &Annotated<u32>) -> String {
    match (&item.error, item.data) {
        (Some(error), _) => format!("error {:?}", error.class()),
        (None, Some(n)) => format!("data {n}"),
        (None, None) => "no data".to_string(),
    }
}

async fn read<S: Stream + Unpin>(stream: S, each: impl Fn(S::Item) -> String) -> Vec<String> {
    stream.map(each).collect().await
}

/// What `AnnotatedItems` and `CheckedItems` yield for one item and then `end`, in `state`.
/// Both apply one rule, so the test also checks they agree.
async fn rule(end: fn() -> gdl::Error, state: State) -> Vec<String> {
    let (root, context) = parts(state);
    let annotated = read(
        AnnotatedItems::new(ended(end()), root.clone(), context.clone()),
        |item| describe_u32(&item),
    )
    .await;
    let checked = read(
        CheckedItems::new(ended(end()), root, context),
        |item| match item {
            Ok(item) => describe_u32(&item),
            Err(error) => format!("Err {:?}", error.kind()),
        },
    )
    .await;
    assert_eq!(
        annotated, checked,
        "the exit and a node's input apply one rule"
    );
    annotated
}

#[tokio::test]
async fn a_kill_with_a_deadline_error_yields_the_deadline_error_item() {
    for state in [LIVE, KILLED, ROOT_CANCELLED] {
        assert_eq!(
            rule(gdl::Error::deadline_exceeded, state).await,
            ["data 1", "error DeadlineExceeded"],
            "{state:?}"
        );
    }
}

#[tokio::test]
async fn a_kill_with_a_node_error_yields_the_node_error_item() {
    for state in [LIVE, KILLED, ROOT_CANCELLED] {
        assert_eq!(
            rule(|| gdl::Error::node("failed"), state).await,
            ["data 1", "error Internal"],
            "{state:?}"
        );
    }
}

#[tokio::test]
async fn a_kill_with_a_cancel_yields_no_item() {
    assert_eq!(
        rule(gdl::Error::cancelled, LIVE).await,
        ["data 1", "error Cancelled"],
        "a cancel of a live request is an error"
    );
    for state in [KILLED, ROOT_CANCELLED] {
        assert_eq!(
            rule(gdl::Error::cancelled, state).await,
            ["data 1"],
            "{state:?}"
        );
    }
}
