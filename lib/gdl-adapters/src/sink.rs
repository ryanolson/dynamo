// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Dynamo engine as a gdl sink.

use std::marker::PhantomData;
use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngine, Data};
use dynamo_runtime::pipeline::{ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use gdl::{Cx, Items};

use crate::error::enter;

/// A Dynamo engine as a gdl [`Sink`](gdl::Sink): Python engines, the KV
/// router's `RoutingHost`, the mocker, or any other
/// `AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>>`.
///
/// Items stay `Annotated<T>`, so an error item is an `Ok` item to gdl and does
/// not end the stream. An eager error enters gdl with the original
/// `anyhow::Error` as its source.
///
/// The request's Dynamo context is the source of truth for stop and kill, and
/// the engine sees it as it is. The sink's `Cx` does not reach the engine:
/// gdl cancels it whenever its attempt ends, also when it ends normally, and
/// passing that on would kill the request after each successful stream. A gdl
/// caller whose stop must reach the engine builds the request with
/// [`CxBridge::context`](crate::CxBridge::context).
pub struct EngineSink<E: ?Sized, Req, T> {
    engine: Arc<E>,
    types: PhantomData<fn(Req) -> T>,
}

impl<E: ?Sized, Req, T> EngineSink<E, Req, T> {
    /// Wrap an engine.
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            types: PhantomData,
        }
    }
}

impl<E: ?Sized, Req, T> Clone for EngineSink<E, Req, T> {
    fn clone(&self) -> Self {
        Self::new(Arc::clone(&self.engine))
    }
}

impl<E, Req, T> gdl::Sink for EngineSink<E, Req, T>
where
    E: AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, anyhow::Error> + ?Sized + 'static,
    Req: Data,
    T: Data,
{
    type Req = SingleIn<Req>;
    type Item = Annotated<T>;

    async fn generate(&self, _cx: Cx, req: SingleIn<Req>) -> gdl::Result<Items<Annotated<T>>> {
        match self.engine.generate(req).await {
            Ok(stream) => Ok(items(stream)),
            Err(error) => Err(enter(error)),
        }
    }
}

/// A Dynamo response stream as gdl items. The stream's context is dropped
/// here; the adapter that leaves gdl puts the request's back.
pub(crate) fn items<T: Data>(stream: ManyOut<T>) -> Items<T> {
    gdl::items(stream.map(Ok))
}
