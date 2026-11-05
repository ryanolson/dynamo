// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native gdl operators as Dynamo engines, for unit tests. A test composes its operator over
//! its own Dynamo engine as the frontend's gdl local pipeline composes it
//! (`entrypoint::input::common`), calls the result as a Dynamo engine, and reads what comes
//! out as a Dynamo stream. So a test of an operator checks the node the frontend runs.

use std::sync::Arc;

use dynamo_gdl_adapters::{Boxed, Carried, EngineSink, PipelineEngine};
use dynamo_runtime::engine::{AsyncEngine, Data};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

use super::GdlOperator;
use crate::backend::Backend;
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest};

/// A Dynamo engine of `Req` and `Annotated<T>`.
pub(crate) type DynEngine<Req, T> = dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>;

/// A test's engine as a gdl sink, taking requests in the carrier `R`.
pub(crate) type EngineStage<Req, T, R = Boxed<Req>> =
    gdl::SinkStage<EngineSink<DynEngine<Req, T>, Req, T, R>>;

/// `engine` as the inner end of a gdl hop, taking requests in the carrier `R`.
pub(crate) fn sink_stage<Req: Data, T: Data, R: Carried<Req>>(
    engine: Arc<DynEngine<Req, T>>,
) -> EngineStage<Req, T, R> {
    gdl::SinkStage::new(Arc::new(EngineSink::new(engine)))
}

/// A hop to `I` with `GdlOperator<O, K>` as its one around node.
pub(crate) type AroundHop<O, K, I> = gdl::Hop<
    (gdl::ForwardHalf<GdlOperator<O, K>>, ()),
    I,
    (gdl::BackwardHalf<GdlOperator<O, K>>, ()),
>;

/// A hop to `inner` with `op`, keyed by `K`, as its one around node.
pub(crate) fn around<O, K, I>(op: Arc<O>, inner: I) -> AroundHop<O, K, I>
where
    GdlOperator<O, K>: gdl::Around,
{
    let (forward, backward) = gdl::around(Arc::new(GdlOperator::new(op)));
    gdl::Hop::new((forward, ()), inner, (backward, ()))
}

/// A gdl pipeline of requests in the carrier `R` as a Dynamo engine, rooted as the frontend
/// roots a pipeline over an in-process engine (Backend and the preprocessor over the engine,
/// the embeddings pipeline): a kill reaches the engine through the request's context only.
pub(crate) fn engine_for_dynamo_stages<N, Req, T, R>(next: N) -> Arc<DynEngine<Req, T>>
where
    N: gdl::Next<R, Annotated<T>>,
    Req: Data,
    T: Data,
    R: Carried<Req>,
{
    Arc::new(PipelineEngine::<_, Req, T, R>::for_dynamo_stages(next))
}

/// `backend` over `next`, as the frontend composes it: an around node of the hop to `next`.
pub(crate) fn backend(
    backend: Arc<Backend>,
    next: Arc<DynEngine<PreprocessedRequest, LLMEngineOutput>>,
) -> Arc<DynEngine<PreprocessedRequest, BackendOutput>> {
    engine_for_dynamo_stages::<_, _, _, Boxed<PreprocessedRequest>>(around::<
        _,
        PreprocessedRequest,
        _,
    >(backend, sink_stage(next)))
}
