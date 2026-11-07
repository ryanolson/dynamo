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
# Counts: a row of name length L at I items makes L + I allocations, 1000 * L + 10 * I
# instructions and 2 * L + I memcpy calls per request, plus FAKE_RISE_ALLOCS, FAKE_RISE_INSTR and
# FAKE_RISE_MEMCPY for the row named FAKE_RISE_ROW. FAKE_RISE_WINDOW_MEMCPY adds calls once to
# that row's measured window. The window spreads the memcpy calls over
# glibc's aarch64 and x86_64 names, through `__memcpy_chk`, whose own call to memcpy is not
# another move, and next to a call to a function whose name only contains "mem". The window of N requests holds N times the per-request instructions; the whole
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
moves=$((items))
extra_moves=0
if [[ $row == "${FAKE_RISE_ROW:-}" ]] \
    && [[ -z ${FAKE_RISE_ITEMS:-} || $items == "$FAKE_RISE_ITEMS" ]]; then
    allocs=$((allocs + ${FAKE_RISE_ALLOCS:-0}))
    per=$((per + ${FAKE_RISE_INSTR:-0}))
    moves=$((moves + ${FAKE_RISE_MEMCPY:-0}))
    extra_moves=${FAKE_RISE_WINDOW_MEMCPY:-0}
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
        echo "fn=(1) caller_a"
        echo "cfn=(2) __GI_memcpy"
        echo "calls=$((${#row} * n)) 0"
        echo "0 10"
        echo "cfn=(6) core::mem::replace"
        echo "calls=$((3 * n)) 0"
        echo "0 10"
        echo "fn=(3) __memcpy_chk"
        echo "cfn=(2)"
        echo "calls=$((${#row} * n)) 0"
        echo "0 10"
        echo "fn=(4) caller_b"
        echo "cfn=(3)"
        echo "calls=$((${#row} * n)) 0"
        echo "0 10"
        echo "cfn=(5) __memmove_avx_unaligned_erms"
        echo "calls=$((moves * n + extra_moves)) 0"
        echo "0 10"
    } >"$out.2"
fi
# FAKE_FLIP_ROW: some requests make 2 allocations fewer, and they are the typical ones.
low=$allocs
if [[ $row == "${FAKE_FLIP_ROW:-}" ]]; then
    low=$((allocs - 2))
fi
[[ $mode == noallocs ]] || echo "$row items=$items allocs=$low bytes=1 allocs_range=$low..$allocs"
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
# b-gdl: 5 + 1 allocations, 5000 + 10 instructions, 10 + 1 memcpy calls; ingress at 16: 23,
# 7160, 30.
FAKE_VALGRIND_MODE=ok run "$here/callgrind.sh" b-gdl 1 ingress 16
if [[ $status -ne 0 ]]; then
    fail ok "exit status $status, want 0"
elif ! grep -Eq '^b-gdl +1 +6 +5010 +11 +6\.\.6$' "$work/stdout"; then
    fail ok "no 'b-gdl 1 6 5010 11 6..6' row"
elif ! grep -Eq '^ingress +16 +23 +7160 +30 +23\.\.23$' "$work/stdout"; then
    fail ok "no 'ingress 16 23 7160 30 23..23' row"
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
FAKE_VALGRIND_MODE=ok run "$here/callgrind.sh" embeddings-gdl 16 a-gdl 1 c-gdl 1 b-gdl 16
if [[ $status -ne 0 ]] || [[ $(awk 'NR > 1 { printf "%s ", $1 }' "$work/stdout") != "embeddings-gdl a-gdl c-gdl b-gdl " ]]; then
    fail order "exit status $status, want 0 and rows in argument order"
else
    echo "ok callgrind.sh order"
fi

# hotpath.sh: the table. A gdl (a-gdl): 5 + 1 = 6 allocations, 5010 instructions and 11 memcpy
# calls at 1 item, 21, 5160 and 26 at 16, so 1.00 allocations, 10 instructions and 1.00 memcpy
# calls per extra item. No row of Dynamo's retired local pipeline.
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh"
if [[ $status -ne 0 ]]; then
    fail table "exit status $status, want 0"
elif ! grep -Eq '^A +gdl +6 +1\.00 +5010 +10 +11 +1\.00 +21 +5160 +26 +6\.\.6,21\.\.21 *$' "$work/stdout"; then
    fail table "no 'A gdl 6 1.00 5010 10 11 1.00 21 5160 26 6..6,21..21' row"
elif [[ $(grep -Ec '^(engine|A|B|C|embed-engine|embeddings|ingress) ' "$work/stdout") -ne 7 ]]; then
    fail table "want 7 rows"
elif grep -Eq '^(A|B|C|embeddings) +dynamo ' "$work/stdout"; then
    fail table "a retired row is in the table"
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

# The change columns against an unchanged base are all zero, the per-item one included.
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/base"
if ! awk '/^pipeline / { rows = 1; next } rows && NF >= 18 && ($13 != 0 || $14 != 0 || $15 != 0 || $16 != 0 || $17 != 0 || $18 != 0) { bad = 1 }
    END { exit bad }' "$work/stdout"; then
    fail same-changes "a change column is not zero against an unchanged base"
else
    echo "ok hotpath.sh same-changes"
fi

# One more allocation per request on the default path: RISE, exit 1.
FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 1 ]] || ! grep -Eq '^B +gdl .* RISE$' "$work/stdout" \
    || [[ $(grep -Ec 'RISE|WORSE' "$work/stdout") -ne 1 ]]; then
    fail rise "exit status $status, want 1 and one RISE on B gdl"
else
    echo "ok hotpath.sh rise"
fi

# One more memcpy call per request at 16 items, a move of an item in one of 15 requests' worth
# of items: far below the instruction tolerance, but RISE, on a gdl row and on the worker's
# Ingress row alike.
for memcpy_case in "b-gdl B gdl RISE" "ingress ingress dynamo RISE"; do
    read -r rise_row pipeline form want_mark <<<"$memcpy_case"
    FAKE_RISE_ROW=$rise_row FAKE_RISE_ITEMS=16 FAKE_RISE_MEMCPY=1 FAKE_RISE_INSTR=150 \
        run "$here/hotpath.sh" --base "$work/base"
    if [[ $status -ne 1 ]] || ! grep -Eq "^$pipeline +$form .* \\+1 +$want_mark\$" "$work/stdout" \
        || [[ $(grep -Ec 'RISE|WORSE' "$work/stdout") -ne 1 ]]; then
        fail "memcpy-$rise_row" "exit status $status, want 1 and one $want_mark on $pipeline $form"
    else
        echo "ok hotpath.sh memcpy-$rise_row"
    fi
done

# A single extra copy in the measured window must survive request-count normalization.
FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 FAKE_RISE_WINDOW_MEMCPY=1 \
    run "$here/hotpath.sh" --base "$work/base"
if [[ $status -eq 0 ]]; then
    fail memcpy-window-rise "the window gained one memcpy call across four requests (104 to 105), want non-zero exit; got exit 0"
else
    echo "ok hotpath.sh memcpy-window-rise"
fi

# A gdl row with more instructions than the tolerance: RISE, exit 1.
FAKE_RISE_ROW=c-gdl FAKE_RISE_INSTR=2000 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 1 ]] || ! grep -Eq '^C +gdl .* RISE$' "$work/stdout"; then
    fail instructions "exit status $status, want 1 and RISE on C gdl"
else
    echo "ok hotpath.sh instructions"
fi

# An instruction rise within the tolerance (300 instructions for a row this small): shown, not
# marked.
FAKE_RISE_ROW=a-gdl FAKE_RISE_INSTR=200 run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 0 ]] || grep -Eq 'RISE|WORSE' "$work/stdout" \
    || ! grep -Eq '^A +gdl .* \+200 ' "$work/stdout"; then
    fail tolerance "exit status $status, want 0, no mark, and +200 on A gdl"
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

# A gdl row whose typical request makes 2 allocations fewer reports its highest request, and
# its range shows the low one: no change, no mark. Another row reports its typical request.
FAKE_FLIP_ROW=b-gdl run "$here/hotpath.sh" --base "$work/base"
if [[ $status -ne 0 ]] || ! grep -Eq '^B +gdl +6 .* 4\.\.6,19\.\.21 +0 ' "$work/stdout"; then
    fail gdl-highest "exit status $status, want 0 and B gdl at 6 with range 4..6,19..21"
else
    echo "ok hotpath.sh gdl-highest"
fi
FAKE_FLIP_ROW=ingress run "$here/hotpath.sh"
if [[ $status -ne 0 ]] || ! grep -Eq '^ingress +dynamo +6 .* 6\.\.8,21\.\.23 *$' "$work/stdout"; then
    fail other-typical "exit status $status, want 0 and ingress at its typical 6"
else
    echo "ok hotpath.sh other-typical"
fi

# A failed run fails the table.
FAKE_VALGRIND_MODE=nowindow run "$here/hotpath.sh"
if [[ $status -eq 0 ]]; then
    fail table-fail "exit status 0 with no window dumps"
else
    echo "ok hotpath.sh table-fail"
fi

# A one-allocation rise at 16 items must survive the displayed slope's rounding.
FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh"
if [[ $status -ne 0 ]]; then
    fail item-allocations-base "exit status $status, want 0"
else
    cp "$work/stdout" "$work/item-base"
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 FAKE_RISE_ALLOCS=2 \
        run "$here/hotpath.sh" --base "$work/item-base"
    if [[ $status -ne 1 ]] || ! grep -Eq '^B +gdl .* RISE$' "$work/stdout"; then
        fail item-allocations "16-item allocations rose from 22 to 23, want exit 1 and RISE on B gdl"
    else
        echo "ok hotpath.sh item-allocations"
    fi
fi

# A base without readable counts must fail rather than pass an allocation rise.
: >"$work/empty-base"
awk '{ print $1, $2, $3, $4, $5, $6 }' "$work/base" >"$work/six-column-base"
for base_case in empty six-column; do
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 \
        run "$here/hotpath.sh" --base "$work/$base_case-base"
    if [[ $status -eq 0 ]]; then
        fail "base-$base_case" "accepted a base without comparable rows despite B gdl gaining an allocation"
    else
        echo "ok hotpath.sh base-$base_case"
    fi
done

# A truncated base cannot establish whether the missing row regressed.
sed '$d' "$work/base" >"$work/truncated-base"
FAKE_RISE_ROW=ingress FAKE_RISE_ALLOCS=1 \
    run "$here/hotpath.sh" --base "$work/truncated-base"
if [[ $status -eq 0 ]]; then
    fail base-truncated "accepted a base missing ingress despite ingress gaining an allocation"
else
    echo "ok hotpath.sh base-truncated"
fi

# A rise equal to the instruction tolerance must pass, including fractional per-item counts.
for boundary_case in "floor 62 750" "percent 37450 752"; do
    read -r boundary_name base_rise allowed_rise <<<"$boundary_case"
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 FAKE_RISE_INSTR=$base_rise \
        run "$here/hotpath.sh"
    if [[ $status -ne 0 ]]; then
        fail "item-$boundary_name-base" "exit status $status, want 0"
        continue
    fi
    cp "$work/stdout" "$work/boundary-base"
    for offset in -1 0 1; do
        want=0
        ((offset > 0)) && want=1
        FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 \
            FAKE_RISE_INSTR=$((base_rise + allowed_rise + offset)) \
            run "$here/hotpath.sh" --base "$work/boundary-base"
        if [[ $status -ne $want ]]; then
            fail "item-$boundary_name-$offset" "exit status $status, want $want at tolerance offset $offset/15"
        elif [[ $want -eq 0 ]] && grep -Eq 'RISE|WORSE' "$work/stdout"; then
            fail "item-$boundary_name-$offset" "marked a rise within the tolerance"
        elif [[ $want -eq 1 ]] && ! grep -Eq '^B +gdl .* RISE$' "$work/stdout"; then
            fail "item-$boundary_name-$offset" "did not mark B gdl above the tolerance"
        else
            echo "ok hotpath.sh item-$boundary_name-$offset"
        fi
    done
done

# Conflicting duplicate rows must not hide an allocation rise behind the last row.
cp "$work/base" "$work/duplicate-base"
awk '$1 == "B" && $2 == "gdl" { $3++; $9++; $12 = $3 ".." $3 "," $9 ".." $9; print }' "$work/base" \
    >>"$work/duplicate-base"
FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 \
    run "$here/hotpath.sh" --base "$work/duplicate-base"
if [[ $status -eq 0 ]]; then
    fail base-duplicate "accepted conflicting B gdl rows and hid allocation rises from 6 to 7 and 21 to 22"
else
    echo "ok hotpath.sh base-duplicate"
fi

# A nonnumeric instruction count cannot establish the per-item tolerance.
awk '$1 == "B" && $2 == "gdl" { $10 = "999999junk" } { print }' "$work/base" \
    >"$work/nonnumeric-instructions-base"
FAKE_RISE_ROW=b-gdl FAKE_RISE_ITEMS=16 FAKE_RISE_INSTR=751 \
    run "$here/hotpath.sh" --base "$work/nonnumeric-instructions-base"
if [[ $status -eq 0 ]]; then
    fail base-nonnumeric-instructions "accepted a nonnumeric instr@16 and hid an instruction rise of 751/15 above the floor of 50 per item"
else
    echo "ok hotpath.sh base-nonnumeric-instructions"
fi

# The table starts with its provenance.
if ! head -n 1 "$work/base" | grep -Eq "^# hotpath format=[0-9]+ arch=$(uname -m) gdl=[^ ]+ commit=[^ ]+$"; then
    fail provenance "the table does not start with its provenance line"
    sed 's/^/  base: /' "$work/base" >&2
else
    echo "ok hotpath.sh provenance"
fi

# Every base the strict parser cannot read in full fails before any run, even when the head
# has a rise it would otherwise mark. Each case: name, then a sed script applied to the base.
while IFS='|' read -r case_name edit; do
    sed -e "$edit" "$work/base" >"$work/bad-base"
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh" --base "$work/bad-base"
    if [[ $status -eq 0 ]]; then
        fail "base-$case_name" "accepted the base"
    elif grep -Eq '^B +gdl ' "$work/stdout"; then
        fail "base-$case_name" "printed a table from a base it should reject"
    elif grep -q 'callgrind output:' "$work/stderr"; then
        fail "base-$case_name" "ran callgrind before rejecting the base"
    else
        echo "ok hotpath.sh base-$case_name"
    fi
done <<'CASES'
garbage-line|$a hello
unexpected-row|/^C  *gdl /{p;s/^C /D /}
renamed-row|s/^C  *gdl /D      gdl /
no-provenance|1d
second-provenance|1p
arch|1s/arch=[^ ]*/arch=riscv64/
format|1s/format=[0-9]*/format=1/
crlf|s/$/\r/
row-before-header|2{h;d};3{p;x}
CASES

# A base that is itself the output of a --base run, with its base line and change columns,
# is read like any other: unchanged counts pass, and a rise is still marked.
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/base"
cp "$work/stdout" "$work/compared-base"
if ! grep -q '^# base format=' "$work/compared-base"; then
    fail compared-base "a --base table has no base line"
fi
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/compared-base"
if [[ $status -ne 0 ]] || grep -Eq 'RISE|WORSE' "$work/stdout"; then
    fail compared-base "exit status $status against a --base table, want 0 and no marks"
else
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh" --base "$work/compared-base"
    if [[ $status -ne 1 ]] || ! grep -Eq '^B +gdl .* RISE$' "$work/stdout"; then
        fail compared-base "exit status $status, want 1 and RISE on B gdl"
    else
        echo "ok hotpath.sh compared-base"
    fi
fi

# A base from before gdl became the only local pipeline: it has the dynamo rows of A, B, C and
# embeddings, which this script no longer measures, and a gdl row marked WORSE, as tables of
# that time marked a worse gdl row. The script skips each retired row, names it on stderr,
# and compares the rest: unchanged counts pass, and a rise is still marked.
awk '
    { print }
    $1 ~ /^(A|B|C|embeddings)$/ && $2 == "gdl" { $2 = "dynamo"; print }
' "$work/compared-base" | sed -E '/^C +gdl /s/ *$/  WORSE/' >"$work/both-forms-base"
if [[ $(grep -Ec '^(A|B|C|embeddings) +dynamo ' "$work/both-forms-base") -ne 4 ]] \
    || ! grep -Eq '^C +gdl .* WORSE$' "$work/both-forms-base"; then
    fail both-forms-base "the test did not build a both-forms base"
fi
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/both-forms-base"
if [[ $status -ne 0 ]] || grep -Eq 'RISE|WORSE' "$work/stdout"; then
    fail both-forms-base "exit status $status against a both-forms base, want 0 and no marks"
elif [[ $(grep -Ec 'skipping (A|B|C|embeddings) dynamo, a retired row' "$work/stderr") -ne 4 ]]; then
    fail both-forms-base "stderr does not name each of the 4 retired rows"
elif grep -Eq '^(A|B|C|embeddings) +dynamo ' "$work/stdout"; then
    fail both-forms-base "a retired row is in the table"
else
    FAKE_RISE_ROW=b-gdl FAKE_RISE_ALLOCS=1 run "$here/hotpath.sh" --base "$work/both-forms-base"
    if [[ $status -ne 1 ]] || ! grep -Eq '^B +gdl .* RISE$' "$work/stdout" \
        || [[ $(grep -Ec 'RISE|WORSE' "$work/stdout") -ne 1 ]]; then
        fail both-forms-base "exit status $status, want 1 and one RISE on B gdl"
    else
        echo "ok hotpath.sh both-forms-base"
    fi
fi

# Only a retired row is skipped: a base row of the dynamo form for a pipeline that never had
# one is still an error.
sed -E '/^engine +- /{p;s/^engine +- /engine dynamo /}' "$work/base" >"$work/unknown-dynamo-base"
FAKE_VALGRIND_MODE=ok run "$here/hotpath.sh" --base "$work/unknown-dynamo-base"
if [[ $status -eq 0 ]] || ! grep -q 'no such row in the table: engine dynamo' "$work/stderr"; then
    fail base-unknown-dynamo "exit status $status, want non-zero and the unknown row named"
else
    echo "ok hotpath.sh base-unknown-dynamo"
fi

if ((failures > 0)); then
    echo "$failures failed" >&2
    exit 1
fi
