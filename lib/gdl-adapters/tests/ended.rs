// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapters at a pipeline's edges in gdl's ended form: a Dynamo engine entering as a sink
//! (`EngineSink`), and the response leaving through `PipelineEngine` (`EndedItems`). Items
//! cross both edges as they are; the pipeline's ending leaves by the same rule as in the
//! `Result` form.

use std::sync::Arc;
use std::task::Poll;

use dynamo_gdl_adapters::{EngineSink, PipelineEngine};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::{StreamExt, stream};
use gdl::{Cx, Ended, EndedForm, Hop, SinkStage, Source};

type Out = ManyOut<Annotated<u32>>;
type Engine = Arc<dyn AsyncEngine<SingleIn<u32>, Out, Error>>;

fn request() -> SingleIn<u32> {
    Context::with_id_and_metadata(3, "req-1".to_string(), Default::default())
}

/// A one-hop ended-form pipeline over `sink`, as a Dynamo engine. `cancel_on_kill` roots it
/// with `PipelineEngine::new`, else with `for_dynamo_stages`.
fn pipeline<S>(sink: S, cancel_on_kill: bool) -> Engine
where
    S: gdl::Sink<EndedForm, Req = SingleIn<u32>, Item = Annotated<u32>>,
{
    let source = Source::new(Hop::new((), SinkStage::new(Arc::new(sink)), ()));
    if cancel_on_kill {
        Arc::new(PipelineEngine::<_, u32, u32, SingleIn<u32>, EndedForm>::new(source))
    } else {
        Arc::new(PipelineEngine::<_, u32, u32, SingleIn<u32>, EndedForm>::for_dynamo_stages(source))
    }
}

/// Each item of the response: its data, or its error's class.
async fn read(engine: Engine, request: SingleIn<u32>) -> Vec<String> {
    let stream = engine.generate(request).await.expect("dispatch");
    stream
        .map(|item| match (&item.error, item.data) {
            (Some(error), _) => format!("error {:?}", error.class()),
            (None, Some(n)) => format!("data {n}"),
            (None, None) => "empty".to_string(),
        })
        .collect()
        .await
}

/// Answers `n` with `0..n`, and an error item last when `fail`: a Dynamo engine whose items
/// carry its errors.
struct Counting {
    fail: bool,
}

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for Counting {
    async fn generate(&self, request: SingleIn<u32>) -> Result<Out, Error> {
        let n = *request;
        let mut items: Vec<_> = (0..n).map(Annotated::from_data).collect();
        if self.fail {
            items.push(Annotated::from_err(
                DynamoError::builder()
                    .class(ErrorClass::WorkerUnavailable)
                    .message("went away")
                    .build(),
            ));
        }
        Ok(ResponseStream::new(
            Box::pin(stream::iter(items)),
            request.context(),
        ))
    }
}

/// Refuses every call.
struct Refusing;

#[async_trait]
impl AsyncEngine<SingleIn<u32>, Out, Error> for Refusing {
    async fn generate(&self, _request: SingleIn<u32>) -> Result<Out, Error> {
        Err(DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("not ready")
            .build()
            .into())
    }
}

fn engine_sink<E>(engine: E) -> EngineSink<E, u32, u32>
where
    E: AsyncEngine<SingleIn<u32>, Out, Error>,
{
    EngineSink::new(Arc::new(engine))
}

#[tokio::test]
async fn a_dynamo_engine_answers_through_an_ended_pipeline() {
    let engine = pipeline(engine_sink(Counting { fail: false }), false);
    assert_eq!(
        read(engine, request()).await,
        ["data 0", "data 1", "data 2"]
    );
}

/// An engine's error item is data to gdl: it passes as an item, and the stream ends cleanly.
#[tokio::test]
async fn an_engines_error_item_passes_as_an_item() {
    let engine = pipeline(engine_sink(Counting { fail: true }), false);
    assert_eq!(
        read(engine, request()).await,
        ["data 0", "data 1", "data 2", "error Unavailable"]
    );
}

/// An engine's eager error leaves as the error it was: the caller can still find its
/// `DynamoError`.
#[tokio::test]
async fn an_engines_eager_error_leaves_as_itself() {
    let engine = pipeline(engine_sink(Refusing), false);
    let Err(error) = engine.generate(request()).await else {
        panic!("refused");
    };
    let dynamo = error
        .downcast_ref::<DynamoError>()
        .expect("the engine's own error");
    assert_eq!(dynamo.class(), ErrorClass::Unavailable);
}

/// How a gdl-native ended sink ends after its items `0..n`.
#[derive(Clone, Copy)]
enum End {
    /// Cancel its own `Cx`, then end: its node boundary reports `Cancelled`.
    CancelOwnScope,
    /// Wait until its `Cx` is cancelled, then end.
    WaitForCancel,
}

struct NativeSink {
    end: End,
}

impl gdl::Sink<EndedForm> for NativeSink {
    type Req = SingleIn<u32>;
    type Item = Annotated<u32>;

    async fn generate(&self, cx: Cx, request: SingleIn<u32>) -> gdl::Result<Ended<Annotated<u32>>> {
        let (n, end) = (*request, self.end);
        let items = stream::iter((0..n).map(Annotated::from_data));
        let tail = stream::once(async move {
            match end {
                End::CancelOwnScope => cx.cancel(),
                End::WaitForCancel => cx.cancelled().await,
            }
        })
        .filter_map(|()| async { None });
        Ok(gdl::ended::clean(items.chain(tail)))
    }
}

/// A sink's own cancel is not a kill of the request: it leaves as a last error item.
#[tokio::test]
async fn a_sinks_own_cancel_leaves_as_a_cancelled_error_item() {
    let engine = pipeline(
        NativeSink {
            end: End::CancelOwnScope,
        },
        false,
    );
    assert_eq!(
        read(engine, request()).await,
        ["data 0", "data 1", "data 2", "error Cancelled"]
    );
}

/// Under `PipelineEngine::new` a kill cancels the root `Cx`: the stream just ends, as a killed
/// Dynamo stream does.
#[tokio::test]
async fn a_kill_that_cancels_the_root_leaves_no_item() {
    let engine = pipeline(
        NativeSink {
            end: End::WaitForCancel,
        },
        true,
    );
    let request = request();
    let context: Arc<dyn AsyncEngineContext> = request.context();
    let mut stream = engine.generate(request).await.expect("dispatch");
    let mut items = Vec::new();
    for _ in 0..3 {
        items.push(stream.next().await.and_then(|item| item.data));
    }
    assert_eq!(items, [Some(0), Some(1), Some(2)]);
    context.kill();
    assert!(stream.next().await.is_none(), "no item after the kill");
    // The stream stays ended.
    assert!(matches!(futures::poll!(stream.next()), Poll::Ready(None)));
}
