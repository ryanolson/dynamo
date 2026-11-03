#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Instructions per request through the frontend pipelines, counted by callgrind. It runs the
# `allocations` test binary, which builds the same pipelines, once with N requests and once
# with 2N, and reports (Ir(2N) - Ir(N)) / N. Setup, warm-up and teardown
# cancel out in the difference.
#
# Usage: lib/llm/benches/callgrind.sh [ROW ITEMS]...
#   With no arguments it measures every row at 1 and 16 items.
#   ROW is engine, a, b, b-gdl, c, c-gdl, embed-engine or embed (see tests/allocations.rs).
#   CALLGRIND_REQUESTS sets N (default 20). CALLGRIND_PROFILE sets the cargo profile
#   (default profiling). CARGO_TARGET_DIR is honored.
#
# Record counts from the `profiling` profile (release with symbols); debug counts mean little.
# Callgrind runs all threads on one, so the totals include tokio's blocking pool, where the
# preprocessor tokenizes.

set -euo pipefail

if ! command -v valgrind >/dev/null 2>&1; then
    echo "callgrind.sh: valgrind is not installed" >&2
    exit 2
fi

requests=${CALLGRIND_REQUESTS:-20}
profile=${CALLGRIND_PROFILE:-profiling}
crate_dir=$(cd "$(dirname "$0")/.." && pwd)
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/callgrind.XXXXXX")

bin=$(cd "$crate_dir" && cargo test --locked --profile "$profile" -p dynamo-llm --test allocations \
    --no-run --message-format=json \
    | jq -r 'select(.reason == "compiler-artifact" and .target.name == "allocations") | .executable')
if [[ -z "$bin" ]]; then
    echo "callgrind.sh: could not find the allocations test binary" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- engine 1 engine 16 a 1 a 16 b 1 b 16 b-gdl 1 b-gdl 16 c 1 c 16 c-gdl 1 c-gdl 16 \
        embed-engine 1 embed-engine 16 embed 1 embed 16
fi
if (( $# % 2 != 0 )); then
    echo "callgrind.sh: arguments must be ROW ITEMS pairs" >&2
    exit 1
fi

# Total instructions (Ir) for one run of the binary.
instructions() {
    local row=$1 items=$2 n=$3
    local out="$out_dir/$row-$items-$n.out"
    valgrind --tool=callgrind --callgrind-out-file="$out" \
        "$bin" row "$row" "$items" "$n" >/dev/null 2>"$out.log"
    awk '/^(summary|totals):/ { print $2; exit }' "$out"
}

printf '%-14s %6s %14s\n' row items instructions/request
while (( $# > 0 )); do
    row=$1 items=$2
    shift 2
    one=$(instructions "$row" "$items" "$requests")
    two=$(instructions "$row" "$items" $((requests * 2)))
    printf '%-14s %6s %14s\n' "$row" "$items" $(((two - one) / requests))
done
echo "callgrind output: $out_dir"
