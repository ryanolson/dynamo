// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Python and KVBM bindings are workspaces of their own, each with its own lockfile. When the
//! workspace moves its gdl pin, their lockfiles must move with it, or their `--locked` builds
//! fail. This compares the lockfiles' text, so it needs no network and no access to gdl's
//! repository.

use std::collections::BTreeSet;
use std::path::Path;

/// The `source` of every gdl package in the lockfile at `path`.
fn gdl_sources(path: &Path) -> BTreeSet<String> {
    let lock = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut sources = BTreeSet::new();
    let mut gdl = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            gdl = false;
        } else if let Some(name) = line.strip_prefix("name = ") {
            gdl = name == "\"gdl\"" || name.starts_with("\"gdl-");
        } else if let Some(source) = line.strip_prefix("source = ")
            && gdl
        {
            sources.insert(source.to_string());
        }
    }
    sources
}

#[test]
fn binding_lockfiles_resolve_the_workspace_gdl_pin() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = gdl_sources(&root.join("Cargo.lock"));
    assert_eq!(
        workspace.len(),
        1,
        "one gdl source in the workspace: {workspace:?}"
    );
    for binding in ["python", "kvbm"] {
        let path = root.join("lib/bindings").join(binding).join("Cargo.lock");
        assert_eq!(gdl_sources(&path), workspace, "{binding}'s lockfile");
    }
}
