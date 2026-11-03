// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What this crate links.
//!
//! Dynamo takes velo from crates.io. gdl's runtime and code generator take
//! velo from a git commit, and two velo sources in one build are two
//! incompatible crates. So this crate may depend on the `gdl` crate, which
//! links no velo, and nothing else from gdl that brings velo.

use std::process::Command;

/// Packages this crate must never build, by name, from any source.
const FORBIDDEN: &[&str] = &["gdl-runtime", "gdl-gen"];

/// Whether one line of `cargo tree --prefix none` names a package this crate
/// must not build: velo or a `velo-*` crate from a git source, or a package
/// in `FORBIDDEN`. A git source prints as `(https://...#<sha>)`; crates.io
/// prints no source and a path prints a path, neither with `://`.
fn forbidden(line: &str) -> bool {
    let mut fields = line.split_whitespace();
    let Some(name) = fields.next() else {
        return false;
    };
    if FORBIDDEN.contains(&name) {
        return true;
    }
    let is_velo = name == "velo" || name.starts_with("velo-");
    is_velo && fields.any(|field| field.contains("://"))
}

/// Normal and build edges, on every target: a generator enters through
/// `[build-dependencies]` and a transport may sit behind a `cfg`.
#[test]
fn links_no_git_velo_and_no_gdl_runtime() {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args([
            "tree",
            "--package",
            "dynamo-gdl-adapters",
            "--edges",
            "no-dev",
            "--target",
            "all",
            "--prefix",
            "none",
            "--locked",
            "--offline",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run cargo tree");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8(output.stdout).expect("utf-8");
    // An empty or truncated tree would pass the check below.
    assert!(
        tree.lines().any(|line| line.starts_with("gdl v")),
        "cargo tree does not show gdl:\n{tree}"
    );
    let found: Vec<&str> = tree.lines().filter(|line| forbidden(line)).collect();
    assert!(found.is_empty(), "dynamo-gdl-adapters links {found:?}");
}

#[test]
fn forbidden_tells_sources_apart() {
    // Allowed: velo from crates.io, gdl from git, a path crate.
    assert!(!forbidden("velo v0.21.0"));
    assert!(!forbidden("velo-hub v0.21.0 (*)"));
    assert!(!forbidden(
        "gdl v0.1.0 (https://github.com/ryanolson/gdl?rev=37864fbb#37864fbb)"
    ));
    assert!(!forbidden(
        "dynamo-runtime v1.7.0 (/src/dynamo/lib/runtime)"
    ));
    assert!(!forbidden("velocity v1.0.0 (https://example.com/v#abc)"));
    assert!(!forbidden(""));

    // Refused: velo or velo-* from git, gdl-runtime and gdl-gen from anywhere.
    assert!(forbidden(
        "velo v0.21.0 (https://github.com/ai-dynamo/velo?rev=e892f87e#e892f87e)"
    ));
    assert!(forbidden(
        "velo-hub v0.21.0 (https://github.com/ai-dynamo/velo?rev=e892f87e#e892f87e) (*)"
    ));
    assert!(forbidden(
        "gdl-runtime v0.1.0 (https://github.com/ryanolson/gdl?rev=37864fbb#37864fbb)"
    ));
    assert!(forbidden("gdl-gen v0.1.0"));
}
