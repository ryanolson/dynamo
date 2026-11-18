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
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_gdl_adapters::{Boxed, EndedItems, EngineSink, PipelineEngine};
use dynamo_llm::backend::Backend;
use dynamo_llm::entrypoint::input::build_pipeline;
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
use gdl::{Cx, Ended, EndedForm, EndedStream};
use tokio::sync::Notify;
use tokio::time::Instant;

type Pre = PreprocessedRequest;
type Out = ManyOut<Annotated<LLMEngineOutput>>;
type Chat = NvCreateChatCompletionRequest;
type ChatOut = NvCreateChatCompletionStreamResponse;
type Pipeline = Arc<dyn AsyncEngine<SingleIn<Chat>, ManyOut<Annotated<ChatOut>>, Error>>;

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// The deadlines a root deadline case tries in turn. The deadline starts at the root, before
/// preprocessing, and gdl reads it on the system's monotonic clock, which a test cannot pause.
/// So a case first checks that the worker held its stream open before the deadline passed, and
/// tries the next, longer deadline when it did not: a slow dispatch makes the case slower, never
/// wrong.
const DEADLINES: [Duration; 3] = [
    Duration::from_millis(100),
    Duration::from_secs(1),
    Duration::from_secs(10),
];

/// How long a stream that should stay open must yield nothing, once the worker holds it.
const QUIET: Duration = Duration::from_millis(200);

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

/// When a worker started to hold its stream open: after the caller read its whole answer,
/// since the worker's stream is polled past its last item only after that item was passed on.
#[derive(Default)]
struct Held {
    at: Mutex<Option<Instant>>,
    signal: Notify,
}

impl Held {
    fn mark(&self) {
        *self.at.lock().unwrap() = Some(Instant::now());
        self.signal.notify_one();
    }

    fn at(&self) -> Option<Instant> {
        *self.at.lock().unwrap()
    }
}

/// What the root does to the request's `Cx` before the pipeline sees it.
#[derive(Clone)]
enum RootEdit {
    None,
    /// A deadline `timeout` from the dispatch; the root records when it passes in `at`.
    Deadline {
        timeout: Duration,
        at: Arc<Mutex<Option<Instant>>>,
    },
    /// A cancel of the root `Cx` once the worker holds its stream open.
    CancelWhenHeld(Arc<Held>),
}

/// The pipeline's root: a gdl `Next` in front of the `Source`, where an edge that sets a
/// request deadline or cancels the request through gdl would be.
#[derive(Clone)]
struct Root<N> {
    next: N,
    edit: RootEdit,
}

impl<N, Req, Item> gdl::Next<Req, Item, EndedForm> for Root<N>
where
    N: gdl::Next<Req, Item, EndedForm>,
    Req: Send + 'static,
    Item: Send + 'static,
{
    fn generate(&self, cx: Cx, req: Req) -> impl Future<Output = gdl::Result<Ended<Item>>> + Send {
        let cx = match &self.edit {
            RootEdit::None => cx,
            RootEdit::Deadline { timeout, at } => {
                let cx = cx.with_timeout(*timeout);
                *at.lock().unwrap() = cx.deadline().map(|deadline| deadline.instant());
                cx
            }
            RootEdit::CancelWhenHeld(held) => {
                let (cancel, held) = (cx.clone(), Arc::clone(held));
                tokio::spawn(async move {
                    held.signal.notified().await;
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
    S: gdl::Sink<EndedForm, Req = Boxed<Pre>, Item = Annotated<LLMEngineOutput>>,
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

/// A Dynamo worker that sends [`answer`], kills `kill` when given, and then either ends or
/// holds its stream open (`hold`): a worker still generating when the request ends around it.
struct DynamoWorker {
    kill: Option<Arc<dyn AsyncEngineContext>>,
    hold: bool,
    held: Arc<Held>,
}

impl DynamoWorker {
    fn holding(kill: Option<Arc<dyn AsyncEngineContext>>) -> (Arc<Self>, Arc<Held>) {
        let held = Arc::new(Held::default());
        let worker = Arc::new(Self {
            kill,
            hold: true,
            held: Arc::clone(&held),
        });
        (worker, held)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Pre>, Out, Error> for DynamoWorker {
    async fn generate(&self, request: SingleIn<Pre>) -> Result<Out, Error> {
        let (kill, hold, held) = (self.kill.clone(), self.hold, Arc::clone(&self.held));
        let tail = stream::once(async move {
            if let Some(context) = kill {
                context.kill();
            }
            if hold {
                held.mark();
                std::future::pending::<()>().await;
            }
        })
        .filter_map(|()| async { None });
        let items = stream::iter(answer()).chain(tail);
        Ok(ResponseStream::new(Box::pin(items), request.context()))
    }
}

fn sink(worker: Arc<DynamoWorker>) -> EngineSink<DynamoWorker, Pre, LLMEngineOutput, Boxed<Pre>> {
    EngineSink::new(worker)
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

impl gdl::Sink<EndedForm> for GdlWorker {
    type Req = Boxed<Pre>;
    type Item = Annotated<LLMEngineOutput>;

    async fn generate(&self, cx: Cx, _req: Boxed<Pre>) -> gdl::Result<Ended<Self::Item>> {
        Ok(gdl::ended(WorkerStream {
            answer: answer().into_iter().collect(),
            end: Some(self.end),
            kill: self.kill.clone(),
            cx,
            ending: None,
        }))
    }
}

/// [`GdlWorker`]'s stream: the answer, then its end as [`GdlEnd`] says. A node error is the
/// stream's ending.
struct WorkerStream {
    answer: std::collections::VecDeque<Annotated<LLMEngineOutput>>,
    end: Option<GdlEnd>,
    kill: Option<Arc<dyn AsyncEngineContext>>,
    cx: Cx,
    ending: Option<gdl::Error>,
}

impl Stream for WorkerStream {
    type Item = Annotated<LLMEngineOutput>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if let Some(item) = self.answer.pop_front() {
            return std::task::Poll::Ready(Some(item));
        }
        if let Some(end) = self.end.take() {
            if let Some(context) = self.kill.take() {
                context.kill();
            }
            match end {
                GdlEnd::Clean => {}
                GdlEnd::CancelOwnScope => self.cx.cancel(),
                GdlEnd::NodeError => self.ending = Some(gdl::Error::node("worker failed")),
            }
        }
        std::task::Poll::Ready(None)
    }
}

impl EndedStream for WorkerStream {
    fn ending(mut self: std::pin::Pin<&mut Self>) -> Option<gdl::Error> {
        self.ending.take()
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

/// Every item of `stream`, described. The stream must end within 10 s.
async fn collect(stream: ManyOut<Annotated<ChatOut>>) -> Vec<String> {
    tokio::time::timeout(
        Duration::from_secs(10),
        stream.map(|item| describe(&item)).collect(),
    )
    .await
    .expect("the response ends")
}

/// Every item of the response, described.
async fn response(pipeline: Pipeline, request: SingleIn<Chat>) -> Vec<String> {
    collect(pipeline.generate(request).await.unwrap()).await
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

/// The response's items, described, and whether the stream ended. Once the worker holds its
/// stream open, a stream that yields nothing for [`QUIET`] counts as open.
async fn read_open(mut stream: ManyOut<Annotated<ChatOut>>, held: &Held) -> (Vec<String>, bool) {
    let until = Instant::now() + Duration::from_secs(10);
    let mut items = Vec::new();
    loop {
        match tokio::time::timeout(QUIET, stream.next()).await {
            Ok(Some(item)) => items.push(describe(&item)),
            Ok(None) => return (items, true),
            Err(_) if held.at().is_some() => return (items, false),
            Err(_) => assert!(Instant::now() < until, "the response moves within 10 s"),
        }
    }
}

/// This file composes pipeline A from `build_pipeline`'s parts, to add its seams. This runs
/// the cases a Dynamo worker gives through both, so a change to the production composition
/// shows here: for example a root that lets a kill cancel the `Cx`, which ends a killed
/// worker's open stream where today it stays open.
#[tokio::test]
async fn the_copy_answers_as_build_pipeline_does() {
    struct Case {
        kill: bool,
        hold: bool,
        items: &'static [&'static str],
        ended: bool,
    }
    let cases = [
        Case {
            kill: false,
            hold: false,
            items: &["text", "finish Length", "usage"],
            ended: true,
        },
        Case {
            kill: true,
            hold: false,
            items: &["text", "finish Length"],
            ended: true,
        },
        Case {
            kill: true,
            hold: true,
            items: &["text", "finish Length"],
            ended: false,
        },
    ];
    for Case {
        kill,
        hold,
        items,
        ended,
    } in cases
    {
        for production in [true, false] {
            let request = chat();
            let held = Arc::new(Held::default());
            let worker = Arc::new(DynamoWorker {
                kill: kill.then(|| request.context()),
                hold,
                held: Arc::clone(&held),
            });
            let pipeline = if production {
                let card = card();
                let tokenizer = card.tokenizer().unwrap();
                build_pipeline::<Chat, ChatOut>(&card, worker, tokenizer)
                    .await
                    .unwrap()
            } else {
                pipeline_a(sink(worker), RootEdit::None)
            };
            let stream = pipeline.generate(request).await.unwrap();
            assert_eq!(
                read_open(stream, &held).await,
                (items.iter().map(|item| item.to_string()).collect(), ended),
                "kill {kill}, hold {hold}, production {production}"
            );
        }
    }
}

// An ending of the request's root `Cx`.

/// The response to a request under a root deadline that passes while the worker holds its
/// stream open, after its answer: see [`DEADLINES`]. `pause` blocks the dispatch of the first
/// try at its first wait, as a descheduled thread would.
async fn under_a_root_deadline(kill: bool, pause: Option<Duration>) -> Vec<String> {
    for (attempt, timeout) in DEADLINES.into_iter().enumerate() {
        let request = chat();
        let (worker, held) = DynamoWorker::holding(kill.then(|| request.context()));
        let at = Arc::new(Mutex::new(None));
        let edit = RootEdit::Deadline {
            timeout,
            at: Arc::clone(&at),
        };
        let pipeline = pipeline_a(sink(worker), edit);
        let mut dispatch = pipeline.generate(request);
        let mut pause = pause.filter(|_| attempt == 0);
        let dispatched = std::future::poll_fn(|task| {
            let poll = dispatch.as_mut().poll(task);
            if poll.is_pending()
                && let Some(pause) = pause.take()
            {
                std::thread::sleep(pause);
            }
            poll
        })
        .await;
        let items = match dispatched {
            Ok(stream) => Some(collect(stream).await),
            Err(_) => None,
        };
        let deadline = at.lock().unwrap().expect("the root set a deadline");
        if held.at().is_some_and(|held| held < deadline) {
            return items.expect("a deadline that passes after the answer does not fail the call");
        }
    }
    panic!("the worker never held its stream before the deadline passed");
}

#[tokio::test]
async fn a_root_deadline_mid_stream_ends_with_a_deadline_error_item() {
    let items = under_a_root_deadline(false, None).await;
    assert_eq!(items, ["text", "finish Length", "error DeadlineExceeded"]);
}

/// The review's case: the dispatch is descheduled past the first deadline. The first try fails
/// eagerly, and the next one, with a longer deadline, gives the case its answer.
#[tokio::test]
async fn a_root_deadline_case_survives_a_slow_dispatch() {
    let items = under_a_root_deadline(false, Some(DEADLINES[0] * 2)).await;
    assert_eq!(items, ["text", "finish Length", "error DeadlineExceeded"]);
}

/// The root `Cx` cancelled is a kill to the exit: the stream just ends. A root ending cuts the
/// request short, so neither case writes the usage chunk.
#[tokio::test]
async fn a_root_cancel_mid_stream_ends_with_no_item() {
    let (worker, held) = DynamoWorker::holding(None);
    let pipeline = pipeline_a(sink(worker), RootEdit::CancelWhenHeld(held));
    let items = response(pipeline, chat()).await;
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
    let items = under_a_root_deadline(true, None).await;
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

// `last_item`'s rule, read directly through `EndedItems`: a node's input, and the exit as
// `PipelineEngine` reads its root.

/// One item, then `None`, with `end` as its ending.
struct EndsWith {
    item: Option<Annotated<u32>>,
    end: Option<gdl::Error>,
}

impl Stream for EndsWith {
    type Item = Annotated<u32>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Annotated<u32>>> {
        std::task::Poll::Ready(self.item.take())
    }
}

impl gdl::EndedStream for EndsWith {
    fn ending(mut self: std::pin::Pin<&mut Self>) -> Option<gdl::Error> {
        assert!(self.item.is_none(), "asked only after `None`");
        self.end.take()
    }
}

fn ended(end: gdl::Error) -> gdl::Ended<Annotated<u32>> {
    gdl::ended(EndsWith {
        item: Some(Annotated::from_data(1)),
        end: Some(end),
    })
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

/// What `EndedItems` yields for one item and then `end` as its ending, in `state`.
async fn rule(end: fn() -> gdl::Error, state: State) -> Vec<String> {
    let (root, context) = parts(state);
    read(EndedItems::new(ended(end()), root, context), |item| {
        describe_u32(&item)
    })
    .await
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
