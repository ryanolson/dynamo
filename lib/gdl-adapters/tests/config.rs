// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl node takes Dynamo's own config type, loaded by Dynamo's loader.
//!
//! Each node below names a Dynamo config as its `Build::Config`, so this file
//! compiles only while that type meets gdl's bound. Each test then loads the
//! config with Dynamo's loader and checks the node gets the loaded value, not
//! a default.
//!
//! Its own test binary: the router test sets a `DYN_*` variable, which is
//! process-wide.

use dynamo_kv_router::config::{KvRouterConfig, try_kv_router_config_from_dynamo_env};
use dynamo_mocker::common::protocols::MockerConfig;
use gdl::Build;

struct RouterNode {
    config: KvRouterConfig,
}

impl Build for RouterNode {
    type Config = KvRouterConfig;
    type Metrics = ();

    async fn build(config: KvRouterConfig, (): ()) -> gdl::Result<Self> {
        Ok(Self { config })
    }
}

struct MockerNode {
    config: MockerConfig,
}

impl Build for MockerNode {
    type Config = MockerConfig;
    type Metrics = ();

    async fn build(config: MockerConfig, (): ()) -> gdl::Result<Self> {
        Ok(Self { config })
    }
}

#[tokio::test]
async fn a_router_node_takes_the_config_dynamos_environment_sets() {
    assert!(!KvRouterConfig::default().router_replica_sync);
    let config = temp_env::with_var("DYN_ROUTER_REPLICA_SYNC", Some("true"), || {
        try_kv_router_config_from_dynamo_env()
    })
    .expect("router config");

    let node = RouterNode::build(config, ()).await.expect("built");
    assert!(node.config.router_replica_sync);
}

#[tokio::test]
async fn a_mocker_node_takes_the_config_dynamos_file_sets() {
    let default = MockerConfig::default();
    assert_ne!(default.runtime.handoff_session_timeout_ms, 1234);
    assert_ne!(default.engine.engine.block_size, 8);
    let config = MockerConfig::from_json_str(
        r#"{"engine": {"block_size": 8}, "dynamo": {"handoff_session_timeout_ms": 1234}}"#,
    )
    .expect("mocker config");

    let node = MockerNode::build(config, ()).await.expect("built");
    assert_eq!(node.config.runtime.handoff_session_timeout_ms, 1234);
    assert_eq!(node.config.engine.engine.block_size, 8);
}
