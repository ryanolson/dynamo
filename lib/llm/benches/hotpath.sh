#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The hot-path table: allocations and instructions per request, for each pipeline in both
# forms, the dynamo form (`DYN_LOCAL_PIPELINE` unset, the default path) and the gdl form
# (`DYN_LOCAL_PIPELINE=gdl`). It runs callgrind.sh for every row at 1 and 16 items.
#
# Usage: lib/llm/benches/hotpath.sh [--base FILE]
#   FILE is an earlier output of this script, with or without its change columns. With it, each
#   row also shows its change, and the script marks
#     RISE   a default-path row (form dynamo, or a stand-in engine) that got worse,
#     WORSE  a gdl row that got worse,
#   and exits 1 if it marked any row. "Worse" means any rise in allocations or memcpy calls, or
#   a rise in instructions above the row's tolerance (below).
#   The environment variables of callgrind.sh apply, and HOTPATH_RUNS (below).
#
# The table's first line is its provenance: "# hotpath format=N arch=A gdl=REV commit=REV".
# One strict parser reads the base and the new table. Each line must be the provenance, the
# header, a blank line or a row that matches the row pattern in full; each row's pipeline and
# form must be one of the table's, once each, with none missing. The base must have the same
# format and arch as this run. The script checks the base before the runs and fails on any
# difference: a base it cannot compare would let a rise pass. Commit and gdl rev are printed,
# not checked.
#
# Columns, per request:
#   allocs, instr, memcpy    at 1 item
#   allocs/item, instr/item, memcpy/item
#                            each extra item: (count at 16 items - count at 1 item) / 15
#   allocs@16, instr@16, memcpy@16
#                            at 16 items; the comparison uses these, not the rounded per-item
#   range                    the lowest and highest allocations of a single request, at 1 and
#                            at 16 items
# For the embeddings rows, an item is an input; their stream has one item for any number.
# A dynamo row's allocations are its typical request's. A gdl row's are its highest request's:
# a gdl request can lose an allocation or two from its window, so its typical count moves
# between runs.
#
# memcpy is the calls to memcpy and memmove (see callgrind.sh): each one is a move of a large
# value, such as a stream item, that the compiler did not elide. One more move per item costs
# fewer instructions than the per-item tolerance, so this column is the guard against it.
#
# Instructions per request vary a little between runs of one binary (see the README). A rise
# counts only above the row's tolerance: the larger of a share of the base (the table below,
# per request and per item) and a floor, HOTPATH_FLOOR instructions per request (default 300)
# and HOTPATH_ITEM_FLOOR per item (default 50). Allocations and memcpy calls do not vary and
# have no tolerance.
#
# HOTPATH_RUNS runs every row that many times (default 1) and keeps the lowest instruction
# count, and the highest memcpy count. The variation only adds instructions (a thread that spins
# or parks once more), so the lowest of a few runs is the closest to the request's own cost.

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

# Provenance of this run. Bump `format` whenever a column changes.
format=3
repo=$(cd "$(dirname "$0")/../../.." && pwd)
arch=$(uname -m)
commit=$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null) || commit=unknown
if [[ $commit != unknown && -n $(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null) ]]; then
    commit=$commit-dirty
fi
gdl=$(awk '/^\[\[package\]\]$/ { in_gdl = 0 }
    $0 == "name = \"gdl\"" { in_gdl = 1 }
    in_gdl && /^source = "git\+.*#[0-9a-f]+"$/ { sub(/.*#/, ""); sub(/"$/, ""); print substr($0, 1, 12); exit }' \
    "$repo/Cargo.lock" 2>/dev/null) || true
[[ -n $gdl ]] || gdl=unknown
provenance="# hotpath format=$format arch=$arch gdl=$gdl commit=$commit"

work=$(mktemp -d "${TMPDIR:-/tmp}/hotpath.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Reads one table strictly. Prints "prov FORMAT ARCH GDL COMMIT", then one
# "row PIPELINE FORM ALLOCS ALLOCS@16 INSTR INSTR@16 MEMCPY MEMCPY@16" per row. Fails, naming the file and line,
# on anything else. Every count it prints matched a whole-number pattern, so no count comes from
# awk's reading of a number at the start of a field.
parse_table() {
    awk -v file="$1" -v table_rows="$table_rows" '
    function bad(msg) {
        printf "hotpath.sh: %s:%d: %s\n", file, FNR, msg > "/dev/stderr"
        failed = 1
        exit 1
    }
    BEGIN {
        n = split(table_rows, lines, "\n")
        for (i = 1; i <= n; i++) {
            split(lines[i], f, " ")
            want[f[1] " " f[2]] = 1
        }
        num = "[0-9]+"; dec = "-?[0-9]+[.][0-9][0-9]"; whole = "-?[0-9]+"; chg = "[+-]?[0-9]+"
        range = num "[.][.]" num "," num "[.][.]" num
        row = "^[^ ]+ +[^ ]+ +" num " +" dec " +" num " +" whole " +" num " +" dec " +" num " +" num
        row = row " +" num " +" range
        row = row "( +" chg " +" chg " +" chg " +" chg " +" chg " +" chg "( +(RISE|WORSE))?)? *$"
        prov_re = "^# hotpath format=[0-9]+ arch=[^ ]+ gdl=[^ ]+ commit=[^ ]+$"
        base_re = "^# base format=[0-9]+ arch=[^ ]+ gdl=[^ ]+ commit=[^ ]+$"
        header_re = "^pipeline +form +allocs +allocs/item +instr +instr/item +memcpy +memcpy/item"
        header_re = header_re " +allocs@16 +instr@16 +memcpy@16 +range"
        header_re = header_re "( +change +allocs@16 +instr +instr/item +memcpy +memcpy@16 +mark)? *$"
    }
    /^$/ { next }
    $0 ~ prov_re {
        if (prov || header) bad("the provenance line must come once, first")
        prov = 1
        split($0, p, /[ =]/)
        print "prov", p[4], p[6], p[8], p[10]
        next
    }
    $0 ~ base_re {
        if (!prov || header || base_line) bad("the base line must come once, after the provenance")
        base_line = 1
        next
    }
    $0 ~ header_re {
        if (!prov || header) bad("the header must come once, after the provenance")
        header = 1
        next
    }
    $0 ~ row {
        if (!header) bad("a row before the header")
        key = $1 " " $2
        if (!(key in want)) bad("no such row in the table: " key)
        if (key in seen) bad("a second row for " key)
        seen[key] = 1
        print "row", $1, $2, $3, $9, $5, $10, $7, $11
        next
    }
    { bad("not a line of the table: " $0) }
    END {
        if (failed) exit 1
        if (!prov) { printf "hotpath.sh: %s: no provenance line\n", file > "/dev/stderr"; exit 1 }
        if (!header) { printf "hotpath.sh: %s: no header\n", file > "/dev/stderr"; exit 1 }
        for (key in want)
            if (!(key in seen)) {
                printf "hotpath.sh: %s: no row for %s\n", file, key > "/dev/stderr"
                missing = 1
            }
        if (missing) exit 1
    }' "$1"
}

# Checks the base before the runs, which take minutes.
if [[ -n $base ]]; then
    parse_table "$base" >"$work/base.parsed" || exit 1
    read -r _ base_format base_arch base_gdl base_commit < <(grep '^prov ' "$work/base.parsed")
    if [[ $base_format != "$format" ]]; then
        echo "hotpath.sh: $base is table format $base_format; this script writes format $format" >&2
        exit 1
    fi
    if [[ $base_arch != "$arch" ]]; then
        echo "hotpath.sh: $base was measured on $base_arch; this host is $arch" >&2
        exit 1
    fi
fi

args=()
for ((run = 0; run < runs; run++)); do
    while read -r _ _ row _; do
        args+=("$row" 1 "$row" 16)
    done <<<"$table_rows"
done
"$(dirname "$0")/callgrind.sh" "${args[@]}" >"$work/counts"

# The new table, from callgrind.sh's counts.
{
    echo "$provenance"
    awk -v table_rows="$table_rows" '
    function bad(msg) { print "hotpath.sh: callgrind.sh output: " msg > "/dev/stderr"; failed = 1; exit 1 }
    function lo(r) { split(r, p, "[.][.]"); return p[1] + 0 }
    function hi(r) { split(r, p, "[.][.]"); return p[2] + 0 }
    NR == 1 { next }
    # Over repeated runs: the highest typical allocations, the lowest and highest single
    # request, the lowest instructions and the highest memcpy calls.
    {
        if ($0 !~ /^[^ ]+ +(1|16) +[0-9]+ +[0-9]+ +[0-9]+ +[0-9]+[.][.][0-9]+$/) bad("not a count line: " $0)
        k = $1 SUBSEP $2
        if (!(k in instr) || $4 + 0 < instr[k]) instr[k] = $4 + 0
        if (!(k in allocs) || $3 + 0 > allocs[k]) allocs[k] = $3 + 0
        if (!(k in copies) || $5 + 0 > copies[k]) copies[k] = $5 + 0
        if (!(k in low) || lo($6) < low[k]) low[k] = lo($6)
        if (!(k in high) || hi($6) > high[k]) high[k] = hi($6)
    }
    END {
        if (failed) exit 1
        printf "%-12s %-6s %7s %11s %9s %10s %6s %11s %9s %10s %9s  %-19s\n", "pipeline", "form", "allocs", "allocs/item", "instr", "instr/item", "memcpy", "memcpy/item", "allocs@16", "instr@16", "memcpy@16", "range"
        nrows = split(table_rows, lines, "\n")
        for (i = 1; i <= nrows; i++) {
            split(lines[i], f, " ")
            pipeline = f[1]; form = f[2]; row = f[3]
            if (!((row, 1) in allocs) || !((row, 16) in allocs)) {
                print "hotpath.sh: no counts for row " row > "/dev/stderr"
                missing = 1
                continue
            }
            # A gdl row can lose an allocation or two from some windows, so it reports its
            # highest request. The dynamo rows report their typical request.
            if (form == "gdl") {
                a = high[row, 1]; a16 = high[row, 16]
            } else {
                a = allocs[row, 1]; a16 = allocs[row, 16]
            }
            r = instr[row, 1]; r16 = instr[row, 16]
            m = copies[row, 1]; m16 = copies[row, 16]
            printf "%-12s %-6s %7d %11.2f %9d %10.0f %6d %11.2f %9d %10d %9d  %-19s\n", pipeline, form,
                a, (a16 - a) / 15, r, (r16 - r) / 15, m, (m16 - m) / 15, a16, r16, m16,
                low[row, 1] ".." high[row, 1] "," low[row, 16] ".." high[row, 16]
        }
        if (missing) exit 2
    }' "$work/counts"
} >"$work/head"
# The new table goes through the same parser as the base.
parse_table "$work/head" >"$work/head.parsed" || exit 1

if [[ -z $base ]]; then
    cat "$work/head"
    exit 0
fi

# Compare the parsed tables in whole numbers, and print the new table with its changes.
awk -v table_rows="$table_rows" -v base_prov="# base format=$base_format arch=$base_arch gdl=$base_gdl commit=$base_commit" \
    -v floor="${HOTPATH_FLOOR:-300}" -v item_floor="${HOTPATH_ITEM_FLOOR:-50}" '
# 100 times the tolerance of a base count b, so the comparison stays in whole numbers.
function tolerance100(b, pct, least) { t = b * pct; return t > least * 100 ? t : least * 100 }
# `diff` is a parameter so that it stays local: awk makes every other name global, and a global
# here would overwrite the per-item delta `d` of the row being printed.
function change(now, was,    diff) { diff = now - was; return (diff > 0 ? "+" : "") diff }
BEGIN {
    n = split(table_rows, lines, "\n")
    for (i = 1; i <= n; i++) {
        split(lines[i], f, " ")
        pct[f[1] " " f[2]] = f[4]; item_pct[f[1] " " f[2]] = f[5]
    }
}
FILENAME == ARGV[1] && $1 == "row" {
    key = $2 " " $3; ba[key] = $4; ba16[key] = $5; br[key] = $6; br16[key] = $7
    bm[key] = $8; bm16[key] = $9; next
}
FILENAME == ARGV[1] { next }
FILENAME == ARGV[2] && $1 == "row" {
    key = $2 " " $3; ha[key] = $4; ha16[key] = $5; hr[key] = $6; hr16[key] = $7
    hm[key] = $8; hm16[key] = $9; next
}
FILENAME == ARGV[2] { next }
/^# hotpath / { print; print base_prov; next }
/^pipeline / { printf "%s  %6s %9s %8s %10s %6s %9s  %s\n", $0, "change", "allocs@16", "instr", "instr/item", "memcpy", "memcpy@16", "mark"; next }
/^$/ { next }
{
    key = $1 " " $2
    a = ha[key]; a16 = ha16[key]; r = hr[key]; r16 = hr16[key]; m = hm[key]; m16 = hm16[key]
    # Per item in fifteenths: the per-item column is rounded for display.
    d = r16 - r; b_d = br16[key] - br[key]
    worse = a > ba[key] || a16 > ba16[key] || m > bm[key] || m16 > bm16[key] \
        || (r - br[key]) * 100 > tolerance100(br[key], pct[key], floor) \
        || (d - b_d) * 100 > tolerance100(b_d, item_pct[key], item_floor * 15)
    mark = ""
    if (worse) { mark = $2 == "gdl" ? "WORSE" : "RISE"; marked++ }
    printf "%s  %6s %9s %8s %10s %6s %9s  %s\n", $0, change(a, ba[key]), change(a16, ba16[key]),
        change(r, br[key]), sprintf("%+.0f", (d - b_d) / 15), change(m, bm[key]),
        change(m16, bm16[key]), mark
}
END { if (marked) exit 1 }' "$work/base.parsed" "$work/head.parsed" "$work/head"
