// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a gdl node in Dynamo gets its configuration.
//!
//! A node's [`Build::Config`](gdl::Build::Config) is Dynamo's existing serde
//! type for that component: `KvRouterConfig` for a router, `MockerConfig` for
//! a mocker. Dynamo loads it as it does today, and the host passes the value
//! to [`Build::build`](gdl::Build::build):
//!
//! ```text
//! let config = try_kv_router_config_from_dynamo_env()?; // DYN_ROUTER_*
//! let node = RouterNode::build(config, metrics).await?;
//! ```
//!
//! So a node keeps every flag, environment variable and file it has now, and
//! a deployment that sets them does not change.
//!
//! gdl's own loader is not used. It lives in gdl's runtime, which this crate
//! does not link (`tests/dependencies.rs`), and it reads `GDL_*` variables
//! and `[node.X]` sections, which mean nothing to Dynamo's operator. For the
//! same reason a node's config does not derive `gdl::Config`: that derive
//! only describes the config to gdl's composer.
//!
//! The one requirement is gdl's bound, `DeserializeOwned + Send`. Dynamo's
//! configs meet it as they are; `tests/config.rs` builds a node from each of
//! the two above, loaded by Dynamo's loader.
