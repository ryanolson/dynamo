#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for callgrind.sh and hotpath.sh, run with stand-ins for cargo and valgrind on PATH, so
# they need neither a build nor valgrind. Needs jq.
#
# Usage: lib/llm/benches/callgrind-test.sh
#
# A valgrind run that dies still writes its counts, and a binary whose client requests did
# nothing writes only the count of the whole run. A script that reads a count without checking
# the run prints a number for either. The stand-in valgrind writes a count for the whole run and
# a window dump, as the real one does, then fails as asked by FAKE_VALGRIND_MODE:
#   ok         exits 0
#   segv       logs "Process terminating with default action of signal 11" and dies by SIGSEGV
#   logonly    logs that line but exits 0
#   nosummary  writes no count in the window dump and exits 0
#   nowindow   writes no window dump, as when the client requests are no-ops, and exits 0
#   noallocs   prints no allocs= and exits 0
#
# Counts: a row of name length L at I items makes L + I allocations and 1000 * L + 10 * I
# instructions per request, plus FAKE_RISE_ALLOCS and FAKE_RISE_INSTR for the row named
# FAKE_RISE_ROW. The window of N requests holds N times the per-request instructions; the whole
# run holds more, so reading it gives a wrong count. With FAKE_JITTER, a run after the first of
# a row (its output file's index is 22 or more) costs 100 instructions more per request.

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/callgrind-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

mkdir "$work/bin" "$work/tmp"

cat >"$work/bin/cargo" <<'EOF'
#!/usr/bin/env bash
test=
while (( $# > 0 )); do
    [[ $1 == --test ]] && test=$2
    shift
done
echo "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"$test\"},\"executable\":\"/nonexistent/$test\"}"
EOF

cat >"$work/bin/valgrind" <<'EOF'
#!/usr/bin/env bash
out=
for arg in "$@"; do
    case $arg in --callgrind-out-file=*) out=${arg#*=} ;; esac
done
# ... BIN row ROW ITEMS N
n=${!#}
args=("$@")
row=${args[$# - 3]} items=${args[$# - 2]}
allocs=$((${#row} + items))
per=$((1000 * ${#row} + 10 * items))
if [[ $row == "${FAKE_RISE_ROW:-}" ]] \
    && [[ -z ${FAKE_RISE_ITEMS:-} || $items == "$FAKE_RISE_ITEMS" ]]; then
    allocs=$((allocs + ${FAKE_RISE_ALLOCS:-0}))
    per=$((per + ${FAKE_RISE_INSTR:-0}))
fi
index=$(basename "$out")
index=${index%%.*}
if [[ -n ${FAKE_JITTER:-} ]] && (( index >= 22 )); then
    per=$((per + 100))
fi
mode=${FAKE_VALGRIND_MODE:-ok}
echo "summary: $((5000000 + per * n))" >"$out"
if [[ $mode != nowindow ]]; then
    {
        echo "desc: Trigger: Client Request: setup and warm-up"
        echo "summary: 5000000"
    } >"$out.1"
    {
        echo "desc: Trigger: Client Request: measured requests"
        [[ $mode == nosummary ]] || echo "summary: $((per * n))"
    } >"$out.2"
fi
[[ $mode == noallocs ]] || echo "$row items=$items allocs=$allocs bytes=1"
case $mode in
    segv | logonly)
        echo "==1== Process terminating with default action of signal 11 (SIGSEGV)" >&2
        ;;
esac
if [[ $mode == segv ]]; then
    kill -SEGV $$
fi
exit 0
EOF
chmod +x "$work/bin/cargo" "$work/bin/valgrind"

failures=0

# Runs a script with the stand-ins; sets status, and writes stdout and stderr to $work.
run() {
    status=0
    PATH="$work/bin:$PATH" TMPDIR="$work/tmp" CALLGRIND_REQUESTS=4 CALLGRIND_JOBS=3 \
        "$@" >"$work/stdout" 2>"$work/stderr" || status=$?
}

fail() {
    echo "FAIL $1: $2" >&2
    sed 's/^/  stdout: /' "$work/stdout" >&2
    sed 's/^/  stderr: /' "$work/stderr" >&2
    failures=$((failures + 1))
}

# callgrind.sh reads the window, not the whole run, and the allocations of the same run.
# b-gdl: 5 + 1 allocations, 5000 + 10 instructions; ingress at 16: 23, 7160.
FAKE_VALGRIND_MODE=ok run "$here/callgrind.sh" b-gdl 1 ingress 16
if [[ $status -ne 0 ]]; then
    fail ok "exit status $status, want 0"
elif ! grep -Eq '^b-gdl +1 +6 +5010$' "$work/stdout"; then
    fail ok "no 'b-gdl 1 6 5010' row"
elif ! grep -Eq '^ingress +16 +23 +7160$' "$work/stdout"; then
    fail ok "no 'ingress 16 23 7160' row"
elif ! grep -q '"executable":"/nonexistent/ingress_allocations"' <(PATH="$work/bin:$PATH" cargo test --test ingress_allocations); then
    fail ok "the cargo stand-in does not name the ingress binary"
else
    echo "ok callgrind.sh ok"
fi

for mode in segv logonly nosummary nowindow noallocs; do
    FAKE_VALGRIND_MODE=$mode run "$here/callgrind.sh" b-gdl 1
    if [[ $status -eq 0 ]]; then
        fail "$mode" "exit status 0, want non-zero"
    elif grep -Eq '^b-gdl +1 ' "$work/stdout"; then
        fail "$mode" "printed a count for a failed run"
    elif ! grep -q 'b-gdl' "$work/stderr" || ! grep -q "$work/tmp/" "$work/stderr"; then
        fail "$mode" "error does not name the row and its files"
    else
        echo "ok callgrind.sh $mode"
    fi
done

# Rows run in parallel but print in argument order.
FAKE_VALGRIND_MODE=ok run "$here/callgrind.sh" embeddings-gdl 16 a 1 c 1 b 16
if [[ $status -ne 0 ]] || [[ $(awk 'NR > 1 { printf "%s ", $1 }' "$work/stdout") != "embeddings-gdl a c b " ]]; then
    fail order "exit status $status, want 0 and rows in argument order"
else
    echo "ok callgrind.sh order"
fi

# hotpath.sh: the table. A dynamo: 1 + 1 = 2 allocations and 1010 instructions at 1 item,
# 17 and 1160 at 16, so 1.0 allocations and 10 instructions per extra item.
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh"
if [[ $status -ne 0 ]]; then
    fail table "exit status $status, want 0"
elif ! grep -Eq '^A +dynamo +2 +1\.0 +1010 +10$' "$work/stdout"; then
    fail table "no 'A dynamo 2 1.0 1010 10' row"
elif [[ $(grep -Ec '^(engine|A|B|C|embed-engine|embeddings|ingress) ' "$work/stdout") -ne 11 ]]; then
    fail table "want 11 rows"
else
    echo "ok hotpath.sh table"
fi
cp "$work/stdout" "$work/base"

# Against an unchanged base: no marks, exit 0.
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 0 ]] || grep -Eq 'RISE|WORSE' "$work/stdout"; then
    fail same "exit status $status and marks against an unchanged base"
else
    echo "ok hotpath.sh same"
fi

# One more allocation per request on the default path: RISE, exit 1.
FAKE_RISE_ROW=b FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 1 ]] || ! grep -Eq '^B +dynamo .* RISE$' "$work/stdout" \
    || [[ $(grep -Ec 'RISE|WORSE' "$work/stdout") -ne 1 ]]; then
    fail rise "exit status $status, want 1 and one RISE on B dynamo"
else
    echo "ok hotpath.sh rise"
fi

# A gdl row with more instructions than the tolerance: WORSE, exit 1.
FAKE_RISE_ROW=c-gdl FAKE_RISE_INSTR=2000 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 1 ]] || ! grep -Eq '^C +gdl .* WORSE$' "$work/stdout"; then
    fail worse "exit status $status, want 1 and WORSE on C gdl"
else
    echo "ok hotpath.sh worse"
fi

# An instruction rise within the tolerance (300 instructions for a row this small): shown, not
# marked.
FAKE_RISE_ROW=a FAKE_RISE_INSTR=200 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 0 ]] || grep -Eq 'RISE|WORSE' "$work/stdout" \
    || ! grep -Eq '^A +dynamo .* \+200 ' "$work/stdout"; then
    fail tolerance "exit status $status, want 0, no mark, and +200 on A dynamo"
else
    echo "ok hotpath.sh tolerance"
fi

# Repeated runs keep the lowest instruction count: the later, costlier runs do not show.
FAKE_JITTER=1 HOTPATH_RUNS=3 run "$here/hotpath.sh"
if [[ $status -ne 0 ]] || ! diff -q "$work/stdout" "$work/base" >/dev/null; then
    fail runs "exit status $status, want 0 and the base table"
else
    echo "ok hotpath.sh runs"
fi

# A failed run fails the table.
FAKE_VALGRIND_MODE=nowindow run "$here/hotpath.sh"
if [[ $status -eq 0 ]]; then
    fail table-fail "exit status 0 with no window dumps"
else
    echo "ok hotpath.sh table-fail"
fi

# A one-allocation rise at 16 items must survive the displayed slope's rounding.
FAKE_RISE_ROW=b FAKE_RISE_ITEMS=16 FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh"
if [[ $status -ne 0 ]]; then
    fail item-allocations-base "exit status $status, want 0"
else
    cp "$work/stdout" "$work/item-base"
    FAKE_RISE_ROW=b FAKE_RISE_ITEMS=16 FAKE_RISE_ALLOCS=2 \
        run "$here/hotpath.sh" --base "$work/item-base"
    if [[ $status -ne 1 ]] || ! grep -Eq '^B +dynamo .* RISE$' "$work/stdout"; then
        fail item-allocations "16-item allocations rose from 18 to 19, want exit 1 and RISE on B dynamo"
    else
        echo "ok hotpath.sh item-allocations"
    fi
fi

if ((failures > 0)); then
    echo "$failures failed" >&2
    exit 1
fi
