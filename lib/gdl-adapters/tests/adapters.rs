// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapters with synthetic engines and operators, against Dynamo alone.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_gdl_adapters::{CxBridge, EngineSink, PipelineEngine, REQUEST_ID};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use gdl::{Cx, Hop, Items, Source};

type Out = ManyOut<Annotated<u32>>;
type DynEngine = dyn AsyncEngine<SingleIn<u32>, Out, Error>;
type Engine = Arc<DynEngine>;
type SinkHop = Hop<(), gdl::SinkStage<EngineSink<DynEngine, u32, u32>>, ()>;

fn make_request(n: u32) -> SingleIn<u32> {
    Context::with_id_and_metadata(n, "req-1".to_string(), Default::default())
}

/// `engine` as the sink of a one-hop gdl pipeline.
fn sink_hop(engine: Engine) -> SinkHop {
    Hop::new(
        (),
        gdl::SinkStage::new(Arc::new(EngineSink::new(engine))),
        (),
    )
}

/// A gdl pipeline as a Dynamo engine.
fn engine_of<N: gdl::Next<SingleIn<u32>, Annotated<u32>>>(next: N) -> Engine {
    Arc::new(PipelineEngine::<_, u32, u32>::new(next))
}

fn data(items: &[Annotated<u32>]) -> Vec<Option<u32>> {
    items.iter().map(|item| item.data).collect()
}

async fn eventually(what: &str, check: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !check() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

/// Streams the request's number and then `items`, and keeps each context it
/// is called with.
#[derive(Default)]
struct Scripted {
    items: Vec<Annotated<u32>>,
    seen: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
}

impl Scripted {
    fn new(items: Vec<Annotated<u32>>) -> Arc<Self> {
        Arc::new(Self {
            items,
            seen: Mutex::default(),
        })
    }

    fn seen(&self) -> Vec<Arc<dyn AsyncEngineContext>> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for Scripted {
    async fn generate(&self, request: SingleIn<u32>) -> Result<Out, Error> {
        let context = request.context();
        self.seen.lock().unwrap().push(context.clone());
        let mut items = vec![Annotated::from_data(*request.content())];
        items.extend(self.items.iter().cloned());
        Ok(ResponseStream::new(Box::pin(stream::iter(items)), context))
    }
}

/// Streams 0, waits for a stop, then streams 1 unless it was killed.
#[derive(Default)]
struct StopAware {
    seen: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for StopAware {
    async fn generate(&self, request: SingleIn<u32>) -> Result<Out, Error> {
        let context = request.context();
        self.seen.lock().unwrap().push(context.clone());
        let watched = context.clone();
        let items = stream::unfold(0u8, move |state| {
            let context = watched.clone();
            async move {
                match state {
                    0 => Some((Annotated::from_data(0), 1)),
                    1 => {
                        context.stopped().await;
                        (!context.is_killed()).then(|| (Annotated::from_data(1), 2))
                    }
                    _ => None,
                }
            }
        });
        Ok(ResponseStream::new(Box::pin(items), context))
    }
}

/// The gdl counterpart of [`StopAware`].
#[derive(Default)]
struct StopAwareSink {
    seen: Mutex<Vec<Cx>>,
}

impl gdl::Sink for StopAwareSink {
    type Req = SingleIn<u32>;
    type Item = Annotated<u32>;

    fn generate(
        &self,
        cx: Cx,
        _req: SingleIn<u32>,
    ) -> impl Future<Output = gdl::Result<Items<Annotated<u32>>>> + Send {
        self.seen.lock().unwrap().push(cx.clone());
        let items = stream::unfold(0u8, move |state| {
            let cx = cx.clone();
            async move {
                match state {
                    0 => Some((Ok(Annotated::from_data(0)), 1)),
                    1 => {
                        cx.stopping().await;
                        (!cx.is_cancelled()).then(|| (Ok(Annotated::from_data(1)), 2))
                    }
                    _ => None,
                }
            }
        });
        async move { Ok(gdl::items(items)) }
    }
}

/// An error a test can find by downcasting.
#[derive(Debug, PartialEq)]
struct Status(u16);

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "status {}", self.0)
    }
}

impl std::error::Error for Status {}

/// Fails every call with `make()`.
struct Failing(fn() -> Error);

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for Failing {
    async fn generate(&self, _request: SingleIn<u32>) -> Result<Out, Error> {
        Err((self.0)())
    }
}

/// Adds one to the request and multiplies each item by ten, as a native gdl
/// operator over Dynamo's request and items.
#[derive(Default)]
struct Relay {
    /// Keep only the first downstream item, then drop the downstream stream.
    first_only: bool,
}

impl gdl::Operator for Relay {
    type UpReq = SingleIn<u32>;
    type UpItem = Annotated<u32>;
    type DownReq = SingleIn<u32>;
    type DownItem = Annotated<u32>;

    fn generate<N>(
        &self,
        cx: Cx,
        request: SingleIn<u32>,
        next: N,
    ) -> impl Future<Output = gdl::Result<Items<Annotated<u32>>>> + Send
    where
        N: gdl::Next<SingleIn<u32>, Annotated<u32>>,
    {
        let first_only = self.first_only;
        async move {
            let items = next.generate(cx, request.map(|n| n + 1)).await?;
            let items = items.map(|item| item.map(|item| item.map_data(|n| Ok(n * 10))));
            Ok(if first_only {
                gdl::items(items.take(1))
            } else {
                gdl::items(items)
            })
        }
    }
}

/// `op` above `engine`, composed with gdl.
fn gdl_stage(op: Arc<Relay>, engine: Engine) -> Engine {
    let stage = gdl::OperatorStage::new(op, sink_hop(engine));
    engine_of(Source::new(Hop::new((), stage, ())))
}

#[tokio::test]
async fn an_engine_sink_streams_what_the_engine_streams() {
    let script = vec![
        Annotated::from_data(1),
        Annotated::from_error("mid-stream failure"),
        Annotated::from_data(2),
    ];
    let direct: Vec<_> = Scripted::new(script.clone())
        .generate(make_request(7))
        .await
        .unwrap()
        .collect()
        .await;
    let adapted: Vec<_> = engine_of(Source::new(sink_hop(Scripted::new(script))))
        .generate(make_request(7))
        .await
        .unwrap()
        .collect()
        .await;

    // The error item does not end the stream.
    assert_eq!(data(&adapted), data(&direct));
    assert_eq!(data(&adapted), [Some(7), Some(1), None, Some(2)]);
    assert!(adapted[2].is_error());
}

#[tokio::test]
async fn the_response_stream_carries_the_request_context() {
    let engine = Scripted::new(vec![]);
    let pipeline = engine_of(Source::new(sink_hop(engine.clone())));
    let request = make_request(1);
    let context = request.context();
    let stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.context().id(), "req-1");
    stream.context().kill();
    assert!(context.is_killed());
    assert!(engine.seen()[0].is_killed());
}

#[tokio::test]
async fn a_normal_end_neither_stops_nor_kills_the_request() {
    let engine = Scripted::new(vec![Annotated::from_data(1)]);
    let pipeline = gdl_stage(Arc::default(), engine.clone());
    let request = make_request(1);
    let context = request.context();
    let items: Vec<_> = pipeline.generate(request).await.unwrap().collect().await;
    assert_eq!(items.len(), 2);
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert!(!context.is_stopped());
    assert!(!engine.seen()[0].is_stopped());
}

#[tokio::test]
async fn an_operator_that_drops_its_downstream_early_kills_nothing() {
    let engine = Scripted::new(vec![Annotated::from_data(1), Annotated::from_data(2)]);
    let pipeline = gdl_stage(Arc::new(Relay { first_only: true }), engine.clone());
    let request = make_request(1);
    let context = request.context();
    let items: Vec<_> = pipeline.generate(request).await.unwrap().collect().await;
    assert_eq!(data(&items), [Some(20)]);
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert!(!context.is_stopped());
    assert!(!engine.seen()[0].is_stopped());
}

#[tokio::test]
async fn a_stop_reaches_a_gdl_sink() {
    let sink = Arc::new(StopAwareSink::default());
    let pipeline = engine_of(Source::new(Hop::new(
        (),
        gdl::SinkStage::new(sink.clone()),
        (),
    )));
    let request = make_request(1);
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().data, Some(0));

    context.stop_generating();
    let cx = sink.seen.lock().unwrap()[0].clone();
    assert!(cx.is_stopping());
    // gdl cancels the sink's scope when the stream ends, so look before.
    assert!(!cx.is_cancelled());
    assert_eq!(stream.next().await.unwrap().data, Some(1));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_kill_reaches_a_gdl_sink_and_ends_the_stream_quietly() {
    let sink = Arc::new(StopAwareSink::default());
    let pipeline = engine_of(Source::new(Hop::new(
        (),
        gdl::SinkStage::new(sink.clone()),
        (),
    )));
    let request = make_request(1);
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().data, Some(0));

    context.kill();
    assert!(sink.seen.lock().unwrap()[0].is_cancelled());
    // A killed Dynamo stream just ends; gdl's `Cancelled` is not an item.
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_kill_after_a_stop_stays_a_kill_at_the_gdl_sink() {
    let sink = Arc::new(StopAwareSink::default());
    let pipeline = engine_of(Source::new(Hop::new(
        (),
        gdl::SinkStage::new(sink.clone()),
        (),
    )));
    let request = make_request(1);
    let context = request.context();
    let _stream = pipeline.generate(request).await.unwrap();
    context.stop_generating();
    context.kill();
    // A later stop leaves the kill in place; the sink's `Cx` stays cancelled.
    context.stop_generating();
    assert!(sink.seen.lock().unwrap()[0].is_cancelled());
}

#[tokio::test]
async fn a_stop_and_a_kill_from_dynamo_reach_a_dynamo_engine_through_gdl() {
    let engine = Arc::new(StopAware::default());
    let pipeline = gdl_stage(Arc::default(), engine.clone());

    let request = make_request(1);
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().data, Some(0));
    context.stop_generating();
    assert_eq!(stream.next().await.unwrap().data, Some(10));
    assert!(stream.next().await.is_none());

    let request = make_request(1);
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().data, Some(0));
    context.kill();
    assert!(stream.next().await.is_none());
    assert!(engine.seen.lock().unwrap()[1].is_killed());
}

#[tokio::test]
async fn a_stop_and_a_cancel_from_gdl_reach_a_dynamo_engine() {
    let engine = Arc::new(StopAware::default());
    let pipeline = Source::new(sink_hop(engine.clone()));

    let cx = Cx::new();
    let mut items = pipeline
        .call(cx.clone(), CxBridge::context(&cx, 1))
        .await
        .unwrap();
    assert_eq!(items.next().await.unwrap().unwrap().data, Some(0));
    cx.request_stop();
    assert_eq!(items.next().await.unwrap().unwrap().data, Some(1));
    assert!(items.next().await.is_none());
    assert!(!engine.seen.lock().unwrap()[0].is_killed());

    let cx = Cx::new();
    let mut items = pipeline
        .call(cx.clone(), CxBridge::context(&cx, 1))
        .await
        .unwrap();
    assert_eq!(items.next().await.unwrap().unwrap().data, Some(0));
    let context = engine.seen.lock().unwrap()[1].clone();
    cx.cancel();
    eventually("the kill", || context.is_killed()).await;
    let ended = items.next().await.unwrap().unwrap_err();
    assert_eq!(ended.kind(), gdl::ErrorKind::Cancelled);
}

#[tokio::test]
async fn an_eager_error_leaves_as_the_error_it_entered_as() {
    let failing: Engine = Arc::new(Failing(|| Status(403).into()));
    for pipeline in [
        engine_of(Source::new(sink_hop(failing.clone()))),
        gdl_stage(Arc::default(), failing.clone()),
    ] {
        let error = pipeline.generate(make_request(1)).await.err().unwrap();
        assert_eq!(error.downcast::<Status>().unwrap(), Status(403));
    }
}

#[tokio::test]
async fn a_request_killed_before_it_starts_fails_as_cancelled() {
    let engine = Scripted::new(vec![]);
    let pipeline = gdl_stage(Arc::default(), engine.clone());
    let request = make_request(1);
    request.context().kill();
    let error = pipeline.generate(request).await.err().unwrap();
    let error = error.downcast::<DynamoError>().unwrap();
    assert_eq!(error.class(), ErrorClass::Cancelled);
    assert!(engine.seen().is_empty());
}

/// A call that must reach its stage after the request is killed, such as the leg that
/// releases staged KV, runs under a cleanup `Cx`: the kill neither cancels nor stops it, so gdl
/// starts the call, and the stage still sees the kill through the request's context.
#[tokio::test]
async fn a_call_under_a_cleanup_cx_reaches_its_sink_after_a_kill() {
    let engine = Scripted::new(vec![Annotated::from_data(2)]);
    let pipeline = Source::new(sink_hop(engine.clone()));

    let request = make_request(1);
    let context = request.context();
    context.kill();
    let refused = pipeline
        .call(CxBridge::cx(context.as_ref()), make_request(1))
        .await;
    assert_eq!(
        refused.err().map(|error| error.kind()),
        Some(gdl::ErrorKind::Cancelled)
    );

    let cx = CxBridge::cx_for_cleanup(context.as_ref());
    assert_eq!(cx.meta().get(REQUEST_ID).map(String::as_str), Some("req-1"));
    assert!(!cx.is_stopping());
    let items: Vec<_> = pipeline.call(cx, request).await.unwrap().collect().await;
    assert_eq!(items.len(), 2);
    let seen = engine.seen();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].is_killed());
}

/// A Dynamo error item reaches the caller through a native stage as an item,
/// and the stream goes on after it.
#[tokio::test]
async fn an_error_item_passes_through_a_native_stage() {
    let script = vec![
        Annotated::from_data(1),
        Annotated::from_error("mid-stream failure"),
        Annotated::from_data(2),
    ];
    let composed: Vec<_> = gdl_stage(Arc::default(), Scripted::new(script))
        .generate(make_request(7))
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(data(&composed), [Some(80), Some(10), None, Some(20)]);
    assert!(composed[2].is_error());
}

#[tokio::test]
async fn a_stop_after_a_kill_still_ends_the_stream_quietly() {
    let sink = Arc::new(StopAwareSink::default());
    let pipeline = engine_of(Source::new(Hop::new(
        (),
        gdl::SinkStage::new(sink.clone()),
        (),
    )));
    let request = make_request(1);
    let context = request.context();
    let mut stream = pipeline.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().data, Some(0));

    context.stop_generating();
    context.kill();
    context.stop_generating();
    assert!(sink.seen.lock().unwrap()[0].is_cancelled());
    let item = stream.next().await;
    assert!(item.is_none(), "a killed stream emitted {item:?}");
}

/// gdl items of `0..n` as data, then `last`, for the readers below.
fn gdl_items(n: u32, last: Option<gdl::Error>) -> Items<Annotated<u32>> {
    let items = (0..n).map(|i| Ok(Annotated::from_data(i)));
    gdl::items(stream::iter(items.chain(last.map(Err))))
}

/// `CheckedItems` keeps each item in its `Result` and applies gdl's `Err` rule as
/// `AnnotatedItems` does: an `Err` ends the stream as a last `Ok` error item, so every item
/// it yields is `Ok`, and the stream stays ended.
#[tokio::test]
async fn checked_items_yield_ok_items_and_end_an_err_with_an_error_item() {
    use dynamo_gdl_adapters::{AnnotatedItems, CheckedItems};
    let context = make_request(0).context();
    let failure = || {
        Some(gdl::Error::new(
            gdl::ErrorKind::Node,
            "the stage below failed",
        ))
    };

    let checked: Vec<_> = CheckedItems::new(gdl_items(2, failure()), Cx::new(), context.clone())
        .collect()
        .await;
    assert_eq!(checked.len(), 3);
    assert!(checked.iter().all(Result::is_ok), "{checked:?}");
    let checked: Vec<Annotated<u32>> = checked.into_iter().map(Result::unwrap).collect();
    assert_eq!(data(&checked), [Some(0), Some(1), None]);
    assert!(checked[2].error.is_some(), "the Err became an error item");

    let annotated: Vec<_> =
        AnnotatedItems::new(gdl_items(2, failure()), Cx::new(), context.clone())
            .collect()
            .await;
    assert_eq!(
        serde_json::to_value(&annotated).unwrap(),
        serde_json::to_value(&checked).unwrap(),
        "AnnotatedItems reads the same items, out of their `Result`"
    );

    let mut checked = CheckedItems::new(gdl_items(0, failure()), Cx::new(), context);
    assert!(checked.next().await.is_some_and(|item| item.is_ok()));
    assert!(checked.next().await.is_none());
    assert!(
        checked.next().await.is_none(),
        "an ended stream stays ended"
    );
}

/// A cancel of a killed request ends `CheckedItems` with no item, as a killed Dynamo stream
/// just ends.
#[tokio::test]
async fn checked_items_end_a_killed_requests_cancel_with_no_item() {
    use dynamo_gdl_adapters::CheckedItems;
    let context = make_request(0).context();
    context.kill();
    let checked: Vec<_> = CheckedItems::new(
        gdl_items(1, Some(gdl::Error::cancelled())),
        Cx::new(),
        context,
    )
    .collect()
    .await;
    assert_eq!(checked.len(), 1, "{checked:?}");
    assert_eq!(checked[0].as_ref().unwrap().data, Some(0));
}
