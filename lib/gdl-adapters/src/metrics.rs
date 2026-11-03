// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A gdl node's metrics on a Dynamo endpoint.
//!
//! [`register`] serves a `derive(gdl::Metrics)` struct as Dynamo serves its
//! own endpoint metrics: each name has Dynamo's `dynamo_component_` prefix,
//! and each series has `dynamo_namespace`, `dynamo_component`,
//! `dynamo_endpoint` and `worker_id` once each, with the values
//! [`create_metric`](dynamo_runtime::metrics::create_metric) gives them for
//! the same endpoint. `dynamo_endpoint` is the endpoint, as in Dynamo, not
//! the gdl node.
//!
//! Only new node metrics use this. Dynamo's existing metrics stay where they
//! are, on Dynamo's registries under their own names, so no dashboard,
//! planner query or scrape path changes. gdl's derive could not take many
//! of them anyway: at compile time it refuses Dynamo's work-handler names
//! (`requests_total` and 61 more, gdl's `RUNTIME_METRICS`), a counter not
//! ending in `_total`, a time not ending in `_seconds`, and a field label
//! named like an identity label.
//!
//! The metrics live in a registry of their own, made with
//! `Registry::new_custom` so that it adds the prefix and the identity labels
//! to every family it gathers, and attached to the endpoint's registry as a
//! child, as Dynamo attaches each endpoint's registry to its component's.
//! Dynamo merges child registries at each scrape, for `/metrics` and for
//! OTLP alike. Two smaller-looking bridges are worse:
//!
//! - Registering on the endpoint's own registry with the labels passed to
//!   `register_with` gives the labels but not the prefix: only a registry
//!   prefixes a gdl name.
//! - An exposition-text callback reaches `/metrics` but not OTLP.
//!
//! Registration happens once, at startup, and the labels and prefix are
//! applied when a scrape gathers: nothing here runs per request.
//!
//! Two things fail at scrape time, not at registration, because each call
//! makes its own registry:
//!
//! - A field named like another family in the same scrape, with a different
//!   help text or type, fails the whole `/metrics` render.
//! - Registering the same type twice on one endpoint gives identical series;
//!   the scrape keeps the first and logs a warning for each later one.

use dynamo_runtime::MetricsRegistry;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{labels, name_prefix, sanitize_prometheus_label};
use gdl::prometheus::Registry;

/// Register `M` on `endpoint`, with Dynamo's metric prefix and identity
/// labels.
///
/// # Errors
///
/// When Prometheus refuses the registry or one of `M`'s metrics. Nothing is
/// then attached to `endpoint`. The derive already refuses, at compile time,
/// the names and labels that would clash inside the registry.
pub fn register<M: gdl::Metrics>(endpoint: &Endpoint) -> anyhow::Result<M> {
    let registry = Registry::new_custom(
        Some(name_prefix::COMPONENT.to_string()),
        Some(identity_labels(endpoint)?.into_iter().collect()),
    )?;
    // The identity labels come from the registry, so the node adds none.
    let metrics = M::register_with(&registry, &[])?;
    let child = MetricsRegistry::new();
    *child
        .prometheus_registry
        .write()
        .expect("a new registry's lock is not poisoned") = registry;
    endpoint.get_metrics_registry().add_child_registry(&child);
    Ok(metrics)
}

/// The identity labels `create_metric` gives a series on `endpoint`.
///
/// Dynamo takes them by position in the hierarchy (runtime, namespace,
/// component, endpoint), not by kind, so under a nested namespace
/// `dynamo_component` is the inner namespace. This follows it, so that a
/// node's series and Dynamo's own on one endpoint agree.
fn identity_labels(endpoint: &Endpoint) -> anyhow::Result<Vec<(String, String)>> {
    let mut names: Vec<String> = endpoint
        .parent_hierarchies()
        .iter()
        .map(|parent| parent.basename())
        .collect();
    names.push(endpoint.basename());

    let mut pairs = Vec::with_capacity(4);
    for (label, name) in [labels::NAMESPACE, labels::COMPONENT, labels::ENDPOINT]
        .into_iter()
        .zip(names.iter().skip(1))
    {
        if name.is_empty() {
            continue;
        }
        let value = sanitize_prometheus_label(name)?;
        if !value.is_empty() {
            pairs.push((label.to_string(), value));
        }
    }
    if let Some(id) = endpoint.connection_id() {
        pairs.push((labels::WORKER_ID.to_string(), format!("{id:x}")));
    }
    Ok(pairs)
}
