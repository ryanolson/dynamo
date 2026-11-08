// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::Command;

#[test]
#[ignore = "edge-endings item 2: both binding lockfiles still pin the public gdl source"]
fn binding_lockfiles_resolve_the_workspace_gdl_pin() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bindings");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut failures = Vec::new();
    for binding in ["python", "kvbm"] {
        let output = Command::new(&cargo)
            .args(["metadata", "--locked", "--format-version", "1"])
            .current_dir(root.join(binding))
            .output()
            .expect("run cargo metadata");
        if !output.status.success() {
            failures.push(format!(
                "{binding}: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "binding lockfiles do not resolve the workspace pin:\n{}",
        failures.join("\n")
    );
}
