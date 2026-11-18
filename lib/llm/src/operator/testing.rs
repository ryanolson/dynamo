// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native gdl operators as Dynamo engines, for unit tests. A test composes its operator over
//! its own Dynamo engine as the frontend's gdl local pipeline composes it
//! (`entrypoint::input::common`), calls the result as a Dynamo engine, and reads what comes
//! out as a Dynamo stream. So a test of an operator checks the node the frontend runs.

use std::sync::Arc;

use dynamo_gdl_adapters::{Boxed, Carried, EndedItems, EngineSink, PipelineEngine};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, Data, ResponseStream};
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

use super::{GdlDownstream, GdlOperator};
use crate::backend::Backend;
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest, PreprocessorNode};
use crate::protocols::common::llm_backend::{
    BackendOutput, EmbeddingsEngineOutput, LLMEngineOutput, PreprocessedRequest,
};
use crate::types::openai::embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse};

/// A Dynamo engine of `Req` and `Annotated<T>`.
pub(crate) type DynEngine<Req, T> = dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>;

/// A test's engine as a gdl sink, taking requests in the carrier `R`.
pub(crate) type EngineStage<Req, T, R = Boxed<Req>> =
    gdl::SinkStage<EngineSink<DynEngine<Req, T>, Req, T, R>>;

/// A hop to a test's engine.
pub(crate) type SinkHop<Req, T, R = Boxed<Req>> = gdl::Hop<(), EngineStage<Req, T, R>, ()>;

/// The downstream a native operator's body gets on gdl, over a test's engine.
pub(crate) type EngineDownstream<Req, T> = GdlDownstream<SinkHop<Req, T>, Boxed<Req>, Req>;

/// `engine` as the inner end of a gdl hop, taking requests in the carrier `R`.
pub(crate) fn sink_stage<Req: Data, T: Data, R: Carried<Req>>(
    engine: Arc<DynEngine<Req, T>>,
) -> EngineStage<Req, T, R> {
    gdl::SinkStage::new(Arc::new(EngineSink::new(engine)))
}

/// A hop to `engine`, taking requests in the carrier `R`.
pub(crate) fn sink<Req: Data, T: Data, R: Carried<Req>>(
    engine: Arc<DynEngine<Req, T>>,
) -> SinkHop<Req, T, R> {
    gdl::Hop::new((), sink_stage(engine), ())
}

/// `op` as a native gdl operator keyed by `K` over the hop `next`.
pub(crate) fn native<O, K, H>(
    op: Arc<O>,
    next: H,
) -> gdl::Hop<(), gdl::OperatorStage<GdlOperator<O, K>, H>, ()> {
    gdl::Hop::new(
        (),
        gdl::OperatorStage::new(Arc::new(GdlOperator::new(op)), next),
        (),
    )
}

/// A hop to `I` with `GdlOperator<O, K>` as its one around node.
pub(crate) type AroundHop<O, K, I> = gdl::Hop<
    (gdl::ForwardHalf<GdlOperator<O, K>, gdl::EndedForm>, ()),
    I,
    (gdl::BackwardHalf<GdlOperator<O, K>, gdl::EndedForm>, ()),
>;

/// A hop to `inner` with `op`, keyed by `K`, as its one around node.
pub(crate) fn around<O, K, I>(op: Arc<O>, inner: I) -> AroundHop<O, K, I>
where
    GdlOperator<O, K>: gdl::Around<gdl::EndedForm>,
{
    let (forward, backward) = gdl::around(Arc::new(GdlOperator::new(op)));
    gdl::Hop::new((forward, ()), inner, (backward, ()))
}

/// A gdl pipeline of boxed requests as a Dynamo engine, rooted as the frontend roots the
/// routed pipelines that hold Migration and the routers: a kill of the request cancels its
/// `Cx`.
pub(crate) fn engine<N, Req, T>(next: N) -> Arc<DynEngine<Req, T>>
where
    N: gdl::Next<Boxed<Req>, Annotated<T>, gdl::EndedForm>,
    Req: Data,
    T: Data,
{
    Arc::new(PipelineEngine::<_, Req, T, Boxed<Req>>::new(next))
}

/// `op`, keyed by `K`, over `next`, as the frontend composes Migration and the routers: a native
/// gdl operator over the hop to `next`, rooted as [`engine`] roots it.
pub(crate) fn operator<O, K, Req, T, DT>(
    op: Arc<O>,
    next: Arc<DynEngine<Req, DT>>,
) -> Arc<DynEngine<Req, T>>
where
    GdlOperator<O, K>: gdl::Operator<
            gdl::EndedForm,
            UpReq = Boxed<Req>,
            UpItem = Annotated<T>,
            DownReq = Boxed<Req>,
            DownItem = Annotated<DT>,
        >,
    O: Send + Sync + 'static,
    K: 'static,
    Req: Data,
    T: Data,
    DT: Data,
{
    engine(native::<_, K, _>(op, sink(next)))
}

/// A gdl pipeline of requests in the carrier `R` as a Dynamo engine, rooted as the frontend
/// roots a pipeline over an in-process engine (Backend and the preprocessor over the engine,
/// the embeddings pipeline): a kill reaches the engine through the request's context only.
pub(crate) fn engine_for_dynamo_stages<N, Req, T, R>(next: N) -> Arc<DynEngine<Req, T>>
where
    N: gdl::Next<R, Annotated<T>, gdl::EndedForm>,
    Req: Data,
    T: Data,
    R: Carried<Req>,
{
    Arc::new(PipelineEngine::<_, Req, T, R>::for_dynamo_stages(next))
}

/// `backend` over `next`, as the frontend composes it: an around node of the hop to `next`,
/// with its backward half as an inline node.
pub(crate) fn backend(
    backend: Arc<Backend>,
    next: Arc<DynEngine<PreprocessedRequest, LLMEngineOutput>>,
) -> Arc<DynEngine<PreprocessedRequest, BackendOutput>> {
    let node = Arc::new(GdlOperator::<_, PreprocessedRequest>::new(backend));
    let (forward, _) = gdl::around(Arc::clone(&node));
    let hop = gdl::Hop::new(
        (forward, ()),
        sink_stage(next),
        (gdl::inline::InlineHalf::new(node), ()),
    );
    engine_for_dynamo_stages::<_, _, _, Boxed<PreprocessedRequest>>(hop)
}

/// `backend` over `next` with its backward half as a stream node: the form Backend's own tests
/// check beside the inline one.
pub(crate) fn backend_stream(
    backend: Arc<Backend>,
    next: Arc<DynEngine<PreprocessedRequest, LLMEngineOutput>>,
) -> Arc<DynEngine<PreprocessedRequest, BackendOutput>> {
    engine_for_dynamo_stages::<_, _, _, Boxed<PreprocessedRequest>>(around::<
        _,
        PreprocessedRequest,
        _,
    >(backend, sink_stage(next)))
}

/// The chat or completions preprocessor over `next`, as the frontend composes it: an around
/// node of the hop to `next`, with `next` also as the engine a speculative prefill goes to.
pub(crate) fn preprocessor<Req, Resp>(
    preprocessor: Arc<OpenAIPreprocessor>,
    next: Arc<DynEngine<PreprocessedRequest, BackendOutput>>,
) -> Arc<DynEngine<Req, Resp>>
where
    Req: Data,
    Resp: Data,
    PreprocessorNode<Req>: gdl::Around<
            gdl::EndedForm,
            UpReq = Boxed<Req>,
            UpItem = Annotated<Resp>,
            DownReq = Boxed<PreprocessedRequest>,
            DownItem = Annotated<BackendOutput>,
        >,
{
    let (forward, backward) = gdl::around(Arc::new(PreprocessorNode::<Req>::new(
        preprocessor,
        next.clone(),
    )));
    engine_for_dynamo_stages::<_, Req, Resp, Boxed<Req>>(gdl::Hop::new(
        (forward, ()),
        sink_stage::<_, _, Boxed<PreprocessedRequest>>(next),
        (backward, ()),
    ))
}

/// The embeddings preprocessor over `next`, as the frontend composes it: a native gdl operator
/// over the hop to `next`, carrying `SingleIn` itself.
pub(crate) fn embeddings_preprocessor(
    preprocessor: Arc<OpenAIPreprocessor>,
    next: Arc<DynEngine<PreprocessedEmbeddingRequest, EmbeddingsEngineOutput>>,
) -> Arc<DynEngine<NvCreateEmbeddingRequest, NvCreateEmbeddingResponse>> {
    type Req = NvCreateEmbeddingRequest;
    engine_for_dynamo_stages::<_, Req, NvCreateEmbeddingResponse, SingleIn<Req>>(
        native::<_, Req, _>(
            preprocessor,
            sink::<_, _, SingleIn<PreprocessedEmbeddingRequest>>(next),
        ),
    )
}

/// The downstream a native operator's body gets on gdl under `cx`, over `engine`. A test
/// derives `cx` from the request's context as the frontend's routed pipelines do
/// (`CxBridge::cx`: a stop stops it and a kill cancels it).
pub(crate) fn downstream<Req: Data, T: Data>(
    cx: gdl::Cx,
    engine: Arc<DynEngine<Req, T>>,
) -> EngineDownstream<Req, T> {
    GdlDownstream::new(cx, sink(engine))
}

/// A native operator body's response read as a Dynamo stream, by the rule the frontend's
/// `PipelineEngine` reads it with. `root` is the `Cx` the body ran under and `context` the
/// request's.
pub(crate) fn read<T: Data>(
    items: gdl::Ended<Annotated<T>>,
    root: gdl::Cx,
    context: Arc<dyn AsyncEngineContext>,
) -> ManyOut<Annotated<T>> {
    ResponseStream::new(
        Box::pin(EndedItems::new(items, root, context.clone())),
        context,
    )
}

/// The `DynamoError` an operator body's eager error on gdl stands for. The body sees a stage's
/// error as an `anyhow::Error` holding the `gdl::Error` that carries it; the frontend's
/// `PipelineEngine` takes it out again on the way up. A test of the body alone finds it on the
/// error's chain, which goes on through the `anyhow::Error` that entered gdl.
pub(crate) fn dynamo_error(error: &anyhow::Error) -> Option<&DynamoError> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<DynamoError>())
}
