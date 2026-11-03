// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Dynamo operator as a gdl stage.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngine, Data};
use dynamo_runtime::pipeline::{ManyOut, Operator, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use gdl::{Cx, Inner, Items};

use crate::engine::PipelineEngine;
use crate::error::enter;
use crate::sink::items;

/// The downstream engine a Dynamo operator calls.
type Downstream<Req, T> = Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, anyhow::Error>>;

/// A Dynamo [`Operator`] as the inner end of a gdl [`Hop`](gdl::Hop), as gdl's
/// own [`OperatorStage`](gdl::OperatorStage) is for a gdl operator.
///
/// A Dynamo operator takes its downstream as an `Arc<dyn AsyncEngine>`, so the
/// stage wraps its downstream hop in a [`PipelineEngine`] once, when it is
/// built, and hands each call a clone of that `Arc`. The operator handles a
/// kill itself, so a kill does not cancel the hop
/// ([`PipelineEngine::for_dynamo_stages`]). Each call into the hop
/// derives a fresh `Cx` from the context the operator passes down: Migration,
/// for one, builds a new context for every attempt.
///
/// The stage's own `Cx` does not reach the operator, for the reason
/// [`EngineSink`](crate::EngineSink) gives: the request's Dynamo context is
/// the source of truth.
///
/// Transition: remove when Migration and the encoder and prefill routers run as native gdl stages.
pub struct DynOperatorStage<O: ?Sized, UpReq, UpT, DownReq: Data, DownT: Data> {
    op: Arc<O>,
    next: Downstream<DownReq, DownT>,
    types: PhantomData<fn(UpReq) -> UpT>,
}

impl<O: ?Sized, UpReq, UpT, DownReq: Data, DownT: Data>
    DynOperatorStage<O, UpReq, UpT, DownReq, DownT>
{
    /// Wrap an operator and the hop it calls.
    pub fn new<H>(op: Arc<O>, next: H) -> Self
    where
        H: gdl::Next<SingleIn<DownReq>, Annotated<DownT>>,
    {
        Self {
            op,
            next: Arc::new(PipelineEngine::for_dynamo_stages(next)),
            types: PhantomData,
        }
    }
}

impl<O: ?Sized, UpReq, UpT, DownReq: Data, DownT: Data> Clone
    for DynOperatorStage<O, UpReq, UpT, DownReq, DownT>
{
    fn clone(&self) -> Self {
        Self {
            op: Arc::clone(&self.op),
            next: Arc::clone(&self.next),
            types: PhantomData,
        }
    }
}

impl<O, UpReq, UpT, DownReq, DownT> Inner<SingleIn<UpReq>>
    for DynOperatorStage<O, UpReq, UpT, DownReq, DownT>
where
    O: Operator<
            SingleIn<UpReq>,
            ManyOut<Annotated<UpT>>,
            SingleIn<DownReq>,
            ManyOut<Annotated<DownT>>,
        > + ?Sized,
    UpReq: Data,
    UpT: Data,
    DownReq: Data,
    DownT: Data,
{
    type Item = Annotated<UpT>;

    fn call(
        &self,
        _cx: Cx,
        req: SingleIn<UpReq>,
    ) -> impl Future<Output = gdl::Result<Items<Annotated<UpT>>>> + Send {
        let next = Arc::clone(&self.next);
        async move {
            match self.op.generate(req, next).await {
                Ok(stream) => Ok(items(stream)),
                Err(error) => Err(enter(error)),
            }
        }
    }
}
