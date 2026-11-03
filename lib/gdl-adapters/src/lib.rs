// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bridge between Dynamo's local pipelines and gdl's.
//!
//! This crate will hold the adapters that let a Dynamo `AsyncEngine` or
//! `Operator` run as a stage of a gdl local pipeline, and a gdl pipeline run
//! as a Dynamo `AsyncEngine`. It is the only Dynamo crate that depends on
//! gdl, and it takes the `gdl` crate alone: gdl's runtime and code generator
//! link velo from git, which Dynamo already takes from crates.io.
//! `tests/dependencies.rs` keeps them out.

use gdl as _;
