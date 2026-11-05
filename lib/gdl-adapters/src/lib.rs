// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bridge between Dynamo's engines and gdl's local pipeline, the only
//! local pipeline the frontend composes.
//!
//! These adapters let a Dynamo `AsyncEngine` run as the sink of a gdl local
//! pipeline, and a gdl pipeline run as a Dynamo `AsyncEngine`:
//!
//! - [`CxBridge`]: a gdl `Cx` from a Dynamo request context, and back.
//! - [`EngineSink`]: a Dynamo engine as a gdl `Sink`.
//! - [`PipelineEngine`]: a gdl `Next` as a Dynamo `AsyncEngine`.
//! - [`Carried`] and [`Boxed`]: how a gdl pipeline carries a Dynamo request,
//!   as `SingleIn<T>` itself or, for a large request, boxed.
//! - [`AnnotatedItems`], [`CheckedItems`] and [`enter`]: what a native gdl
//!   operator over Dynamo items uses to read its downstream and to raise an
//!   eager error, so it behaves at its edges as the adapters do.
//!   `dynamo_llm::operator` builds Backend, the preprocessor, Migration and the
//!   routers this way; the stages that edit items in place read through
//!   `CheckedItems`, which leaves each item in its gdl `Result`.
//! - [`metrics`]: a gdl node's metrics on a Dynamo endpoint.
//! - [`config`]: how a gdl node in Dynamo gets its configuration.
//!
//! A gdl pipeline of native operators over a Dynamo engine:
//!
//! ```text
//! HTTP --Context--> PipelineEngine --Cx, Context--> Hop
//!                                                    \-> OperatorStage(op)
//!                                    op --Cx, Context--> Hop
//!                                                         \-> SinkStage(EngineSink(engine))
//! ```
//!
//! The request stays Dynamo's `SingleIn<T>`, boxed for a large one
//! ([`Boxed`]), and its context is the source of truth: `PipelineEngine`
//! derives a fresh `Cx` from the context it receives, and nothing is stored
//! in the context's registry, because
//! Migration rebuilds the context of every attempt. Items stay `Annotated<T>`,
//! so a Dynamo error item does not end a stream, while a gdl `Err` does.
//!
//! gdl enters Dynamo only here and in `dynamo-llm`'s gdl compositions and
//! native operators, and both take the `gdl` crate alone: gdl's runtime and
//! code generator link velo from git, which Dynamo already takes from
//! crates.io. `tests/dependencies.rs` keeps them out of this crate.

pub mod config;
pub mod metrics;

mod bridge;
mod engine;
mod error;
mod request;
mod sink;

pub use bridge::{CxBridge, REQUEST_ID};
pub use engine::{AnnotatedItems, CheckedItems, PipelineEngine};
pub use error::enter;
pub use request::{Boxed, Carried, SpareBox};
pub use sink::EngineSink;
