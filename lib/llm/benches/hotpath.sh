#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The hot-path table: allocations and instructions per request, for each pipeline in both
# forms, the dynamo form (`DYN_LOCAL_PIPELINE` unset, the default path) and the gdl form
# (`DYN_LOCAL_PIPELINE=gdl`). It runs callgrind.sh for every row at 1 and 16 items.
#
# Usage: lib/llm/benches/hotpath.sh [--base FILE]
#   FILE is an earlier output of this script. With it, each row also shows its change, and the
#   script marks
#     RISE   a default-path row (form dynamo, or a stand-in engine) that got worse,
#     WORSE  a gdl row that got worse,
#   and exits 1 if it marked any row. "Worse" means any rise in allocations, or a rise in
#   instructions above the row's tolerance (below).
#   The environment variables of callgrind.sh apply, and HOTPATH_RUNS (below).
#
# Columns, per request:
#   allocs, instr          at 1 item
#   allocs/item, instr/item  each extra item: (count at 16 items - count at 1 item) / 15
# For the embeddings rows, an item is an input; their stream has one item for any number.
#
# Instructions per request vary a little between runs of one binary (see the README). A rise
# counts only above the row's tolerance: the larger of a share of the base (the table below,
# per request and per item) and a floor, HOTPATH_FLOOR instructions per request (default 300)
# and HOTPATH_ITEM_FLOOR per item (default 50). Allocations do not vary and have no tolerance.
#
# HOTPATH_RUNS runs every row that many times (default 1) and keeps the lowest instruction
# count. The variation only adds instructions (a thread that spins or parks once more), so the
# lowest of a few runs is the closest to the request's own cost.

set -euo pipefail

base=
while (( $# > 0 )); do
    case $1 in
        --base)
            base=${2:?--base needs a file}
            shift 2
            ;;
        *)
            echo "hotpath.sh: unknown argument $1" >&2
            exit 1
            ;;
    esac
done
if [[ -n $base && ! -r $base ]]; then
    echo "hotpath.sh: cannot read the base table $base" >&2
    exit 1
fi

# pipeline form row, and the tolerance in percent per request and per item, in table order.
# The embeddings rows' batch encode runs on rayon, whose idle threads spin a varying number of
# times; their tolerance is wider.
table_rows="engine - engine 1 2
A dynamo a 1 2
A gdl a-gdl 1 2
B dynamo b 1 2
B gdl b-gdl 1 2
C dynamo c 1 2
C gdl c-gdl 1 2
embed-engine - embed-engine 1 2
embeddings dynamo embed 3 6
embeddings gdl embeddings-gdl 3 6
ingress dynamo ingress 1 2"

runs=${HOTPATH_RUNS:-1}
args=()
for ((run = 0; run < runs; run++)); do
    while read -r _ _ row _; do
        args+=("$row" 1 "$row" 16)
    done <<<"$table_rows"
done

counts=$(mktemp "${TMPDIR:-/tmp}/hotpath.XXXXXX")
trap 'rm -f "$counts"' EXIT
"$(dirname "$0")/callgrind.sh" "${args[@]}" >"$counts"

awk -v base="$base" -v table_rows="$table_rows" \
    -v floor="${HOTPATH_FLOOR:-300}" -v item_floor="${HOTPATH_ITEM_FLOOR:-50}" '
function tolerance(b, pct, least) { t = b * pct / 100; return t > least ? t : least }
function change(now, was) { d = now - was; return (d > 0 ? "+" : "") d }
BEGIN {
    if (base != "") {
        while ((getline line < base) > 0) {
            n = split(line, f)
            if (n >= 6 && f[3] ~ /^[0-9]+$/) {
                key = f[1] " " f[2]
                b_allocs[key] = f[3]; b_allocs_item[key] = f[4]
                b_instr[key] = f[5]; b_instr_item[key] = f[6]
            }
        }
    }
}
# Repeated runs: the highest allocations (they must not vary) and the lowest instructions.
NR > 1 {
    if (!(($1, $2) in instr) || $4 < instr[$1, $2]) instr[$1, $2] = $4
    if (!(($1, $2) in allocs) || $3 > allocs[$1, $2]) allocs[$1, $2] = $3
}
END {
    if (base == "")
        printf "%-12s %-6s %8s %11s %10s %10s\n", "pipeline", "form", "allocs", "allocs/item", "instr", "instr/item"
    else
        printf "%-12s %-6s %8s %11s %10s %10s  %6s %11s %8s %10s  %s\n", "pipeline", "form", "allocs", "allocs/item", "instr", "instr/item", "change", "allocs/item", "instr", "instr/item", "mark"
    nrows = split(table_rows, lines, "\n")
    marked = 0
    for (i = 1; i <= nrows; i++) {
        split(lines[i], f, " ")
        pipeline = f[1]; form = f[2]; row = f[3]; key = pipeline " " form
        pct = f[4]; item_pct = f[5]
        if (!((row, 1) in allocs) || !((row, 16) in allocs)) {
            print "hotpath.sh: no counts for row " row > "/dev/stderr"
            missing = 1
            continue
        }
        a = allocs[row, 1]; ai = sprintf("%.1f", (allocs[row, 16] - a) / 15)
        r = instr[row, 1]; ri = sprintf("%.0f", (instr[row, 16] - r) / 15)
        if (base == "") {
            printf "%-12s %-6s %8d %11s %10d %10s\n", pipeline, form, a, ai, r, ri
            continue
        }
        if (!(key in b_allocs)) {
            printf "%-12s %-6s %8d %11s %10d %10s  %6s %11s %8s %10s  %s\n", pipeline, form, a, ai, r, ri, "-", "-", "-", "-", "new"
            continue
        }
        worse = a > b_allocs[key] || ai + 0 > b_allocs_item[key] + 0 \
            || r - b_instr[key] > tolerance(b_instr[key], pct, floor) \
            || ri - b_instr_item[key] > tolerance(b_instr_item[key], item_pct, item_floor)
        mark = ""
        if (worse) { mark = form == "gdl" ? "WORSE" : "RISE"; marked++ }
        printf "%-12s %-6s %8d %11s %10d %10s  %6s %11s %8s %10s  %s\n", pipeline, form, a, ai, r, ri,
            change(a, b_allocs[key]), sprintf("%+.1f", ai - b_allocs_item[key]),
            change(r, b_instr[key]), change(ri, b_instr_item[key]), mark
    }
    if (missing) exit 2
    if (marked) exit 1
}' "$counts"
