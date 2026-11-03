#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The hot-path table: allocations and instructions per request, for each pipeline in both
# forms, the dynamo form (`DYN_LOCAL_PIPELINE` unset, the default path) and the gdl form
# (`DYN_LOCAL_PIPELINE=gdl`). It runs callgrind.sh for every row at 1 and 16 items.
#
# Usage: lib/llm/benches/hotpath.sh [--base FILE]
#   FILE is an earlier output of this script; a file with no rows in that format fails before
#   any run. With it, each row also shows its change, and the
#   script marks
#     RISE   a default-path row (form dynamo, or a stand-in engine) that got worse,
#     WORSE  a gdl row that got worse,
#   and exits 1 if it marked any row. "Worse" means any rise in allocations, or a rise in
#   instructions above the row's tolerance (below).
#   The environment variables of callgrind.sh apply, and HOTPATH_RUNS (below).
#
# Columns, per request:
#   allocs, instr            at 1 item
#   allocs/item, instr/item  each extra item: (count at 16 items - count at 1 item) / 15
#   allocs@16, instr@16      at 16 items; the comparison uses these, not the rounded per-item
#   range                    the lowest and highest allocations of a single request, at 1 and
#                            at 16 items
# For the embeddings rows, an item is an input; their stream has one item for any number.
# A dynamo row's allocations are its typical request's. A gdl row's are its highest request's:
# a gdl request can lose an allocation or two from its window, so its typical count moves
# between runs.
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
# A base with no rows in this script's format would make every row "new" and pass any rise.
# Check it before the runs, which take minutes.
if [[ -n $base ]] && ! awk 'NF >= 9 && $3 ~ /^[0-9]+$/ && $7 ~ /^[0-9]+$/ { found = 1 }
        END { exit !found }' "$base"; then
    echo "hotpath.sh: $base has no rows in this script's table format" >&2
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
function lo(range) { split(range, p, "[.][.]"); return p[1] + 0 }
function hi(range) { split(range, p, "[.][.]"); return p[2] + 0 }
BEGIN {
    if (base != "") {
        while ((getline line < base) > 0) {
            n = split(line, f)
            if (n >= 9 && f[3] ~ /^[0-9]+$/ && f[7] ~ /^[0-9]+$/) {
                key = f[1] " " f[2]
                b_allocs[key] = f[3]; b_instr[key] = f[5]
                b_allocs16[key] = f[7]; b_instr16[key] = f[8]
            }
        }
    }
}
# Over repeated runs: the highest typical allocations, the lowest and highest single request,
# and the lowest instructions.
NR > 1 {
    k = $1 SUBSEP $2
    if (!(k in instr) || $4 < instr[k]) instr[k] = $4
    if (!(k in allocs) || $3 > allocs[k]) allocs[k] = $3
    if (!(k in low) || lo($5) < low[k]) low[k] = lo($5)
    if (!(k in high) || hi($5) > high[k]) high[k] = hi($5)
}
END {
    header = sprintf("%-12s %-6s %7s %11s %9s %10s %9s %10s  %-19s", "pipeline", "form", "allocs", "allocs/item", "instr", "instr/item", "allocs@16", "instr@16", "range")
    if (base != "")
        header = header sprintf("  %6s %9s %8s %10s  %s", "change", "allocs@16", "instr", "instr/item", "mark")
    print header
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
        # A gdl row can lose an allocation or two from some windows, so it reports its highest
        # request. The dynamo rows report their typical request.
        if (form == "gdl") {
            a = high[row, 1]; a16 = high[row, 16]
        } else {
            a = allocs[row, 1]; a16 = allocs[row, 16]
        }
        r = instr[row, 1]; r16 = instr[row, 16]
        range = low[row, 1] ".." high[row, 1] "," low[row, 16] ".." high[row, 16]
        line = sprintf("%-12s %-6s %7d %11.2f %9d %10.0f %9d %10d  %-19s", pipeline, form,
            a, (a16 - a) / 15, r, (r16 - r) / 15, a16, r16, range)
        if (base == "") {
            print line
            continue
        }
        if (!(key in b_allocs)) {
            print line sprintf("  %6s %9s %8s %10s  %s", "-", "-", "-", "-", "new")
            continue
        }
        # Compare raw counts: a per-item figure is rounded for display.
        ri = (r16 - r) / 15; b_ri = (b_instr16[key] - b_instr[key]) / 15
        worse = a > b_allocs[key] || a16 > b_allocs16[key] \
            || r - b_instr[key] > tolerance(b_instr[key], pct, floor) \
            || ri - b_ri > tolerance(b_ri, item_pct, item_floor)
        mark = ""
        if (worse) { mark = form == "gdl" ? "WORSE" : "RISE"; marked++ }
        print line sprintf("  %6s %9s %8s %10s  %s", change(a, b_allocs[key]),
            change(a16, b_allocs16[key]), change(r, b_instr[key]),
            sprintf("%+.0f", ri - b_ri), mark)
    }
    if (missing) exit 2
    if (marked) exit 1
}' "$counts"
