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
//! the gdl node. Dynamo assigns the three names by position in the hierarchy,
//! so under a nested namespace `dynamo_component` is the inner namespace and
//! `dynamo_endpoint` the component; the bridge does the same, so that a
//! node's series and Dynamo's own on one endpoint agree.
//!
//! Only new node metrics use this. Dynamo's existing metrics stay where they
//! are, on Dynamo's registries under their own names, so no dashboard,
//! planner query or scrape path changes. gdl's derive could not take many
//! of them anyway: at compile time it refuses the names in gdl's
//! `RUNTIME_METRICS` (`requests_total` and 61 more), a counter not ending in
//! `_total`, a time not ending in `_seconds`, and a field label named like an
//! identity label. That list does not hold all of Dynamo's work-handler
//! names: the derive takes `request_bytes_total` or `queue_depth`, for
//! example, and registration refuses them instead.
//!
//! The metrics live in a registry of their own, made with
//! `Registry::new_custom` so that it adds the prefix and the identity labels
//! to every family it gathers. A collector serves that registry from the
//! endpoint's own registry, the one `create_metric` registers in, and
//! describes each metric there under its served name and labels. So the
//! endpoint's registry refuses a clash when the second of two registrations
//! is made, as it does between two of Dynamo's own metrics:
//!
//! - Registering the same type twice on one endpoint fails the second time.
//!   Register once per endpoint and keep the handles.
//! - A node metric and a Dynamo metric with one name on one endpoint fail
//!   whichever registers second. When it is Dynamo's work-handler metrics,
//!   the endpoint fails to start.
//!
//! Two other bridges are worse. A child registry attached with
//! `add_child_registry` is merged only when a scrape runs, so both clashes
//! above would register without error and appear only at scrape time: the
//! second registration's handles would never be served, and two help texts
//! for one name fail the whole `/metrics` render. An exposition-text callback
//! reaches `/metrics` but not OTLP.
//!
//! A refused registration is not undone in full: Prometheus records the
//! help and labels of each name it checked before the clash, for the life
//! of the registry, so a later metric with one of those names on that
//! endpoint must keep them. Treat a refusal as a startup error.
//!
//! What registration cannot see is a clash with a metric on another
//! endpoint or another level of the hierarchy: a scrape merges them all, and
//! one name with two help texts fails it. Dynamo's own metrics share that
//! limit.
//!
//! Registration happens once, at startup, and the labels and prefix are
//! applied when a scrape gathers: nothing here runs per request.

use std::collections::HashMap;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{labels, name_prefix, sanitize_prometheus_label};
use gdl::prometheus::Registry;
use gdl::prometheus::core::{Collector, Desc};
use gdl::prometheus::proto::MetricFamily;

/// Register `M` on `endpoint`, with Dynamo's metric prefix and identity
/// labels.
///
/// # Errors
///
/// When the endpoint's registry already holds a metric with one of `M`'s
/// served names and the same labels, or with a different help text or label
/// names: `M` registered on this endpoint before, or Dynamo uses the name.
/// Nothing is then registered.
pub fn register<M: gdl::Metrics>(endpoint: &Endpoint) -> anyhow::Result<M> {
    let identity: HashMap<String, String> = identity_labels(endpoint)?.into_iter().collect();
    let registry = Registry::new_custom(
        Some(name_prefix::COMPONENT.to_string()),
        Some(identity.clone()),
    )?;
    // The identity labels come from the registry, so the node adds none.
    let metrics = M::register_with(&registry, &[])?;
    let descs = M::definitions()
        .into_iter()
        .map(|field| {
            Desc::new(
                format!("{}_{}", name_prefix::COMPONENT, field.name),
                field.help,
                field.labels,
                identity.clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A collector with no descriptors has the id 0, so a second one on the
    // same registry would be refused. A node without metrics serves nothing.
    if !descs.is_empty() {
        endpoint
            .get_metrics_registry()
            .add_metric(Box::new(NodeCollector { descs, registry }))?;
    }
    Ok(metrics)
}

/// A node's registry, served from the endpoint's registry under the names
/// and labels it adds.
struct NodeCollector {
    descs: Vec<Desc>,
    registry: Registry,
}

impl Collector for NodeCollector {
    fn desc(&self) -> Vec<&Desc> {
        self.descs.iter().collect()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        self.registry.gather()
    }
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
