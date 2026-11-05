#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Allocations and instructions per request through the frontend pipelines and the worker's
# Ingress, one run of a test binary per row under callgrind.
#
# The pipeline rows run the `allocations` test binary of dynamo-llm; the `ingress` row runs the
# `ingress_allocations` test binary of dynamo-runtime. Each binary marks its measured requests
# with two callgrind client requests: one dump after setup and warm-up, one after the measured
# requests. The script reads the second dump, the window, and reports its instructions (all
# threads) divided by N. Setup, warm-up and teardown are outside it. Allocations per request are
# the binary's own count, from the same run.
#
# Usage: lib/llm/benches/callgrind.sh [ROW ITEMS]...
#   With no arguments it measures every row at 1 and 16 items.
#   ROW is engine, a-gdl, b-gdl, c-gdl, embed-engine, embeddings-gdl (see
#   tests/allocations.rs) or ingress (see lib/runtime/tests/ingress_allocations.rs).
#   CALLGRIND_REQUESTS sets N (default 200). CALLGRIND_JOBS sets how many rows run at once
#   (default: the number of CPUs). CALLGRIND_PROFILE sets the cargo profile (default
#   profiling). CARGO_TARGET_DIR is honored.
#
# Output: a header, then one line per argument pair,
# "ROW ITEMS ALLOCATIONS INSTRUCTIONS MEMCPY RANGE", in argument order. ALLOCATIONS is the
# binary's typical request; MEMCPY is the calls to memcpy and memmove in the window, per
# request; RANGE is the lowest and highest allocations of its measured requests, "LOW..HIGH". A
# pair may repeat, for repeated runs. A row that fails prints no line, and the script exits
# non-zero.
#
# MEMCPY counts what the instructions cannot show: each move of a large value (a stream item of
# about 1 KB) that the compiler does not elide is one call, of 100 to 200 instructions, below
# the instruction tolerance of hotpath.sh. It counts calls from any function that is not itself
# a memcpy or memmove, so `__memcpy_chk` calling `memcpy` counts once. The names match glibc's
# on aarch64 (`__GI_memcpy`) and x86_64 (`__memmove_avx_unaligned_erms` and the like).
#
# Record counts from the `profiling` profile (release with symbols); debug counts mean little.
# Callgrind runs all threads on one, so the totals include tokio's blocking pool, where the
# preprocessor tokenizes. The Ingress row's stand-in frontend turns its thread's counting off.
#
# The script fails, naming the row and the valgrind log, when valgrind exits non-zero, when its
# log says the program died by a signal, or when there is no window dump. A run that dies still
# writes its counts, and a binary whose client requests did nothing writes only the count of the
# whole run, so a number from either would be wrong.
#
# The pipeline rows die with SIGSEGV under valgrind unless built with `--cfg rustix_use_libc`; see
# the "callgrind.sh and hotpath.sh" section of benches/README.md for the build.

set -euo pipefail

if ! command -v valgrind >/dev/null 2>&1; then
    echo "callgrind.sh: valgrind is not installed" >&2
    exit 2
fi

requests=${CALLGRIND_REQUESTS:-200}
profile=${CALLGRIND_PROFILE:-profiling}
crate_dir=$(cd "$(dirname "$0")/.." && pwd)
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/callgrind.XXXXXX")

if [[ $# -eq 0 ]]; then
    set -- engine 1 engine 16 a-gdl 1 a-gdl 16 b-gdl 1 b-gdl 16 c-gdl 1 c-gdl 16 \
        embed-engine 1 embed-engine 16 embeddings-gdl 1 embeddings-gdl 16 ingress 1 ingress 16
fi
if (( $# % 2 != 0 )); then
    echo "callgrind.sh: arguments must be ROW ITEMS pairs" >&2
    exit 1
fi

# The test binary of PACKAGE named TEST.
test_binary() {
    local package=$1 test=$2 bin
    bin=$(cd "$crate_dir" && cargo test --locked --profile "$profile" -p "$package" --test "$test" \
        --no-run --message-format=json \
        | jq -r --arg test "$test" \
            'select(.reason == "compiler-artifact" and .target.name == $test) | .executable')
    if [[ -z "$bin" ]]; then
        echo "callgrind.sh: could not find the $test test binary" >&2
        return 1
    fi
    echo "$bin"
}

# Build and find each binary once, before the runs start.
llm_bin= ingress_bin=
for ((i = 1; i <= $#; i += 2)); do
    if [[ ${!i} == ingress ]]; then
        [[ -n $ingress_bin ]] || ingress_bin=$(test_binary dynamo-runtime ingress_allocations)
    else
        [[ -n $llm_bin ]] || llm_bin=$(test_binary dynamo-llm allocations)
    fi
done

# Calls to memcpy and memmove in the callgrind dump FILE, from functions that are not themselves
# one. Function names are compressed: "(ID) NAME" the first time, "(ID)" after.
memcpy_calls() {
    awk '
    function name(spec,    id, rest) {
        if (!match(spec, /^\([0-9]+\)/)) return spec
        id = substr(spec, 2, RLENGTH - 2)
        rest = substr(spec, RLENGTH + 2)
        if (rest != "") names[id] = rest
        return id in names ? names[id] : spec
    }
    BEGIN { copy = "^(__GI_)?_*mem(cpy|move)" }
    /^fn=/ { fn = name(substr($0, 4)); next }
    /^cfn=/ { cfn = name(substr($0, 5)); next }
    /^calls=/ {
        split(substr($0, 7), c, " ")
        if (cfn ~ copy && fn !~ copy) total += c[1]
    }
    END { print total + 0 }' "$1"
}

# "ALLOCATIONS INSTRUCTIONS MEMCPY RANGE" per request for one run of a row. Check the exit
# status, the log and the window dump before reading a count.
measure() {
    local bin=$1 row=$2 items=$3 n=$4 i=$5
    local out="$out_dir/$i.$row-$items.out"
    local what="row $row, $items items"
    local status=0
    valgrind --tool=callgrind --callgrind-out-file="$out" \
        "$bin" row "$row" "$items" "$n" >"$out.stdout" 2>"$out.log" || status=$?
    if (( status != 0 )); then
        echo "callgrind.sh: $what: valgrind exited with status $status; see $out.log" >&2
        return 1
    fi
    if grep -q 'Process terminating with default action of signal' "$out.log"; then
        echo "callgrind.sh: $what: the program died by a signal; see $out.log" >&2
        return 1
    fi
    local window
    window=$(grep -l '^desc: Trigger: Client Request: measured requests$' "$out".* 2>/dev/null \
        | head -n 1) || true
    if [[ -z $window ]]; then
        echo "callgrind.sh: $what: no window dump (\"measured requests\") beside $out; see $out.log" >&2
        return 1
    fi
    local ir
    ir=$(awk '/^(summary|totals):/ { print $2; exit }' "$window" 2>/dev/null) || true
    if [[ ! $ir =~ ^[0-9]+$ ]]; then
        echo "callgrind.sh: $what: no instruction total in $window; see $out.log" >&2
        return 1
    fi
    local memcpy
    memcpy=$(memcpy_calls "$window") || true
    if [[ ! $memcpy =~ ^[0-9]+$ ]]; then
        echo "callgrind.sh: $what: could not count memcpy calls in $window; see $out.log" >&2
        return 1
    fi
    # Every request of a row makes the same copies, so the window's count divides by N. A
    # remainder is a copy that not every request makes; dividing would round it away, and a
    # rise hidden that way would pass the zero tolerance of hotpath.sh.
    if (( memcpy % n != 0 )); then
        echo "callgrind.sh: $what: $memcpy memcpy calls in $n requests is not a whole number per request; see $window" >&2
        return 1
    fi
    local allocs range
    allocs=$(grep -o 'allocs=[0-9]*' "$out.stdout" | head -n 1 | cut -d= -f2) || true
    if [[ ! $allocs =~ ^[0-9]+$ ]]; then
        echo "callgrind.sh: $what: no allocs= in the program's output $out.stdout" >&2
        return 1
    fi
    # The ingress row prints no range: it fails unless every request allocates the same.
    range=$(grep -o 'allocs_range=[0-9]*\.\.[0-9]*' "$out.stdout" | head -n 1 | cut -d= -f2) || true
    [[ -n $range ]] || range=$allocs..$allocs
    echo "$allocs $((ir / n)) $((memcpy / n)) $range"
}

# Rows run in parallel, CALLGRIND_JOBS at a time (default: the number of CPUs). Each run is one
# valgrind process on one CPU, and the window's count does not depend on the host's load.
jobs=${CALLGRIND_JOBS:-$(nproc)}
rows=()
while (( $# > 0 )); do
    rows+=("$1 $2")
    shift 2
done
for i in "${!rows[@]}"; do
    read -r row items <<<"${rows[$i]}"
    bin=$llm_bin
    [[ $row == ingress ]] && bin=$ingress_bin
    while (( $(jobs -rp | wc -l) >= jobs )); do
        wait -n || true
    done
    (measure "$bin" "$row" "$items" "$requests" "$i" >"$out_dir/result.$i") &
done
# A run's exit status may already be reaped by `wait -n`; its result file says whether it
# succeeded, so read that instead.
wait
failed=0

printf '%-14s %6s %12s %14s %8s %12s\n' row items allocations instructions memcpy range
for i in "${!rows[@]}"; do
    read -r row items <<<"${rows[$i]}"
    counts=$(cat "$out_dir/result.$i" 2>/dev/null) || true
    if [[ ! $counts =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\.\.[0-9]+$ ]]; then
        failed=1
        continue
    fi
    read -r allocs ir memcpy range <<<"$counts"
    printf '%-14s %6s %12s %14s %8s %12s\n' "$row" "$items" "$allocs" "$ir" "$memcpy" "$range"
done
echo "callgrind output: $out_dir" >&2
exit "$failed"
