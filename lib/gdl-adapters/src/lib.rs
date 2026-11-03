// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bridge between Dynamo's local pipelines and gdl's.
//!
//! These adapters let a Dynamo `AsyncEngine` or `Operator` run as a stage of a
//! gdl local pipeline, and a gdl pipeline run as a Dynamo `AsyncEngine`:
//!
//! - [`CxBridge`]: a gdl `Cx` from a Dynamo request context, and back.
//! - [`EngineSink`]: a Dynamo engine as a gdl `Sink`.
//! - [`PipelineEngine`]: a gdl `Next` as a Dynamo `AsyncEngine`.
//! - [`DynOperatorStage`]: a Dynamo `Operator` as the inner end of a gdl `Hop`.
//! - [`AnnotatedItems`] and [`enter`]: what a native gdl operator over Dynamo
//!   items uses to read its downstream and to raise an eager error, so it
//!   behaves at its edges as the adapters do. `dynamo_llm::operator` builds
//!   Backend and the preprocessor this way.
//! - [`metrics`]: a gdl node's metrics on a Dynamo endpoint.
//! - [`config`]: how a gdl node in Dynamo gets its configuration.
//!
//! A pipeline of Dynamo operators composed with gdl:
//!
//! ```text
//! HTTP --Context--> PipelineEngine --Cx, Context--> Hop
//!                                                    \-> DynOperatorStage(op)
//!                        op --Context--> PipelineEngine --Cx, Context--> Hop
//!                                                                         \-> SinkStage(EngineSink(engine))
//! ```
//!
//! The request stays Dynamo's `SingleIn<T>` and its context is the source of
//! truth: each `PipelineEngine` derives a fresh `Cx` from the context it
//! receives, and nothing is stored in the context's registry, because
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
mod sink;
mod stage;

pub use bridge::{CxBridge, REQUEST_ID};
pub use engine::{AnnotatedItems, PipelineEngine};
pub use error::enter;
pub use sink::EngineSink;
pub use stage::DynOperatorStage;
