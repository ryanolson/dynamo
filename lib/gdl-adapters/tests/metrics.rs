// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl node's metrics on a Dynamo endpoint, as `/metrics` serves them: with
//! Dynamo's prefix, and each identity label once, with the value Dynamo's own
//! metrics on that endpoint carry.
//!
//! Its own test binary: the scrape test sets `DYN_SYSTEM_PORT`, which is
//! process-wide.

use dynamo_gdl_adapters::metrics;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::{DistributedRuntime, MetricsRegistry, Runtime};
use gdl::Metrics as _;
use gdl::prometheus::Registry;

/// The labels Dynamo adds to each series of an endpoint.
const IDENTITY: [&str; 4] = [
    "dynamo_namespace",
    "dynamo_component",
    "dynamo_endpoint",
    "worker_id",
];

#[derive(gdl::Metrics)]
struct NodeMetrics {
    /// Prompts the node has seen.
    prompts_total: gdl::prometheus::IntCounter,
    /// Prompts refused, by reason.
    #[gdl(labels = ["reason"])]
    refused_total: gdl::prometheus::IntCounterVec,
    /// Time to the first token.
    #[gdl(buckets = [0.5, 1.5])]
    first_token_seconds: gdl::prometheus::Histogram,
}

impl NodeMetrics {
    /// Give each field a series, so that each shows in the text.
    fn record(&self) {
        self.prompts_total.inc();
        self.refused_total.with_label_values(&["full"]).inc();
        self.first_token_seconds.observe(1.0);
    }
}

/// The names the bridge must serve `NodeMetrics` under.
const FAMILIES: [&str; 3] = [
    "dynamo_component_prompts_total",
    "dynamo_component_refused_total",
    "dynamo_component_first_token_seconds",
];

async fn runtime() -> DistributedRuntime {
    let rt = Runtime::from_current().expect("runtime");
    DistributedRuntime::new(rt, DistributedConfig::process_local())
        .await
        .expect("distributed runtime")
}

/// The sample lines of `text` whose metric name is `family` or one of its
/// histogram series.
fn samples<'a>(text: &'a str, family: &str) -> Vec<&'a str> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| {
            let name = line.split(['{', ' ']).next().unwrap_or_default();
            name == family
                || ["_bucket", "_sum", "_count"]
                    .iter()
                    .any(|suffix| name.strip_prefix(family) == Some(suffix))
        })
        .collect()
}

/// The label pairs of one sample line, in order.
fn labels(line: &str) -> Vec<(String, String)> {
    let Some((_, rest)) = line.split_once('{') else {
        return Vec::new();
    };
    let (body, _) = rest.split_once('}').expect("closing brace");
    body.split(',')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (name, value) = pair.split_once('=').expect("name=value");
            (name.to_string(), value.trim_matches('"').to_string())
        })
        .collect()
}

/// `line`'s labels named in `IDENTITY`, sorted.
fn identity(line: &str) -> Vec<(String, String)> {
    let mut pairs: Vec<_> = labels(line)
        .into_iter()
        .filter(|(name, _)| IDENTITY.contains(&name.as_str()))
        .collect();
    pairs.sort();
    pairs
}

/// How often `line` names `label`.
fn count(line: &str, label: &str) -> usize {
    labels(line)
        .iter()
        .filter(|(name, _)| name == label)
        .count()
}

async fn scrape(drt: &DistributedRuntime) -> String {
    let address = drt
        .system_status_server_info()
        .expect("system status server")
        .socket_addr;
    let response = reqwest::get(format!("http://{address}/metrics"))
        .await
        .expect("scrape");
    assert_eq!(response.status(), 200);
    response.text().await.expect("body")
}

#[tokio::test]
async fn metrics_scrape_with_dynamos_prefix_and_identity() {
    temp_env::async_with_vars([("DYN_SYSTEM_PORT", Some("0"))], async {
        let drt = runtime().await;
        let endpoint = drt
            .namespace("gdl-ns")
            .and_then(|ns| ns.component("router"))
            .expect("component")
            .endpoint("generate");
        metrics::register::<NodeMetrics>(&endpoint)
            .expect("registered")
            .record();

        let text = scrape(&drt).await;
        let worker_id = format!("{:x}", drt.connection_id());
        let expected: Vec<(String, String)> = {
            let mut pairs = vec![
                ("dynamo_namespace".to_string(), "gdl_ns".to_string()),
                ("dynamo_component".to_string(), "router".to_string()),
                ("dynamo_endpoint".to_string(), "generate".to_string()),
                ("worker_id".to_string(), worker_id),
            ];
            pairs.sort();
            pairs
        };
        // A sample each for the counter and the counter vector, and the 5
        // series of the histogram: 2 buckets, +Inf, sum and count.
        let lines: Vec<&str> = FAMILIES.iter().flat_map(|f| samples(&text, f)).collect();
        assert_eq!(lines.len(), 7, "{text}");
        for line in lines {
            for label in IDENTITY {
                assert_eq!(count(line, label), 1, "{label} in {line}");
            }
            assert_eq!(identity(line), expected, "{line}");
        }
        // Nothing under the node's own, unprefixed names.
        for family in ["prompts_total", "refused_total", "first_token_seconds"] {
            assert!(samples(&text, family).is_empty(), "{family} in {text}");
        }
        drt.shutdown();
    })
    .await;
}

/// The control: the same scrape keeps a label a registry adds twice, so the
/// count above can fail. gdl's `register` adds `dynamo_endpoint` itself, and
/// a registry with Dynamo's identity labels adds it again.
#[tokio::test]
async fn a_doubled_label_shows_in_the_scrape() {
    temp_env::async_with_vars([("DYN_SYSTEM_PORT", Some("0"))], async {
        let drt = runtime().await;
        let endpoint = drt
            .namespace("control")
            .and_then(|ns| ns.component("router"))
            .expect("component")
            .endpoint("generate");
        let labels = IDENTITY.map(|name| (name.to_string(), "host".to_string()));
        let child = MetricsRegistry::new();
        *child.prometheus_registry.write().unwrap() =
            Registry::new_custom(Some("dynamo_component".to_string()), Some(labels.into()))
                .expect("registry");
        NodeMetrics::register(&child.prometheus_registry.read().unwrap(), "Node")
            .expect("registered")
            .record();
        endpoint.get_metrics_registry().add_child_registry(&child);

        let text = scrape(&drt).await;
        let lines = samples(&text, FAMILIES[0]);
        assert_eq!(lines.len(), 1, "{text}");
        assert_eq!(count(lines[0], "dynamo_endpoint"), 2, "{}", lines[0]);
        drt.shutdown();
    })
    .await;
}

/// The identity labels of `endpoint`'s series: one Dynamo creates with
/// `create_metric`, and one through the bridge.
async fn dynamo_and_bridge(
    drt: &DistributedRuntime,
    endpoint: &Endpoint,
) -> [Vec<(String, String)>; 2] {
    endpoint
        .metrics()
        .create_intcounter("dynamo_side_total", "A Dynamo metric.", &[])
        .expect("dynamo metric")
        .inc();
    metrics::register::<NodeMetrics>(endpoint)
        .expect("registered")
        .record();
    let text = drt
        .get_metrics_registry()
        .prometheus_expfmt_combined()
        .expect("text");
    let dynamo = samples(&text, "dynamo_component_dynamo_side_total");
    let bridge = samples(&text, FAMILIES[0]);
    assert_eq!((dynamo.len(), bridge.len()), (1, 1), "{text}");
    [identity(dynamo[0]), identity(bridge[0])]
}

/// The bridge labels a series as `create_metric` does, including names it
/// sanitizes.
#[tokio::test]
async fn identity_matches_dynamos_own_metrics() {
    let drt = runtime().await;
    let endpoint = drt
        .namespace("my-ns.v1")
        .and_then(|ns| ns.component("kv-router"))
        .expect("component")
        .endpoint("generate.v2");
    let [dynamo, bridge] = dynamo_and_bridge(&drt, &endpoint).await;
    assert_eq!(dynamo.len(), 4, "{dynamo:?}");
    assert_eq!(bridge, dynamo);
    drt.shutdown();
}

/// Under a nested namespace Dynamo labels by position in the hierarchy, not
/// by kind. The bridge follows it, so both sides of one endpoint agree.
#[tokio::test]
async fn identity_matches_under_a_nested_namespace() {
    let drt = runtime().await;
    let endpoint = drt
        .namespace("outer")
        .and_then(|ns| ns.namespace("inner"))
        .and_then(|ns| ns.component("router"))
        .expect("component")
        .endpoint("generate");
    let [dynamo, bridge] = dynamo_and_bridge(&drt, &endpoint).await;
    assert_eq!(bridge, dynamo);
    drt.shutdown();
}
