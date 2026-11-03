#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for callgrind.sh, run with stand-ins for cargo and valgrind on PATH, so they need
# neither a build nor valgrind. Needs jq.
#
# Usage: lib/llm/benches/callgrind-test.sh
#
# A valgrind run that dies still writes a callgrind summary, so a script that reads the
# summary without checking the run prints a count for a crashed run. The stand-in valgrind
# writes the summary first, as the real one did on the bench host, then fails as asked by
# FAKE_VALGRIND_MODE:
#   ok         exits 0
#   segv       logs "Process terminating with default action of signal 11" and dies by SIGSEGV
#   logonly    logs that line but exits 0
#   nosummary  writes no summary and exits 0

set -euo pipefail

script=$(cd "$(dirname "$0")" && pwd)/callgrind.sh
work=$(mktemp -d "${TMPDIR:-/tmp}/callgrind-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

mkdir "$work/bin" "$work/tmp"

cat >"$work/bin/cargo" <<'EOF'
#!/usr/bin/env bash
echo '{"reason":"compiler-artifact","target":{"name":"allocations"},"executable":"/nonexistent/allocations"}'
EOF

# Ir = 1000 + 7 * N, so (Ir(2N) - Ir(N)) / N is 7.
cat >"$work/bin/valgrind" <<'EOF'
#!/usr/bin/env bash
out=
for arg in "$@"; do
    case $arg in --callgrind-out-file=*) out=${arg#*=} ;; esac
done
n=${!#}
if [[ $FAKE_VALGRIND_MODE != nosummary ]]; then
    echo "summary: $((1000 + 7 * n))" >"$out"
fi
case $FAKE_VALGRIND_MODE in
    segv | logonly)
        echo "==1== Process terminating with default action of signal 11 (SIGSEGV)" >&2
        ;;
esac
if [[ $FAKE_VALGRIND_MODE == segv ]]; then
    kill -SEGV $$
fi
exit 0
EOF
chmod +x "$work/bin/cargo" "$work/bin/valgrind"

failures=0

# Runs callgrind.sh for row b-gdl at 1 item with the given stand-in mode.
run() {
    local mode=$1
    status=0
    PATH="$work/bin:$PATH" TMPDIR="$work/tmp" FAKE_VALGRIND_MODE=$mode CALLGRIND_REQUESTS=4 \
        "$script" b-gdl 1 >"$work/stdout" 2>"$work/stderr" || status=$?
}

fail() {
    echo "FAIL $1: $2" >&2
    sed 's/^/  stdout: /' "$work/stdout" >&2
    sed 's/^/  stderr: /' "$work/stderr" >&2
    failures=$((failures + 1))
}

run ok
if [[ $status -ne 0 ]]; then
    fail ok "exit status $status, want 0"
elif ! grep -Eq '^b-gdl +1 +7$' "$work/stdout"; then
    fail ok "no 'b-gdl 1 7' row"
else
    echo "ok ok"
fi

for mode in segv logonly nosummary; do
    run "$mode"
    if [[ $status -eq 0 ]]; then
        fail "$mode" "exit status 0, want non-zero"
    elif grep -Eq '^b-gdl +1 ' "$work/stdout"; then
        fail "$mode" "printed a count for a failed run"
    elif ! grep -q 'b-gdl' "$work/stderr" || ! grep -q "$work/tmp/.*\.log" "$work/stderr"; then
        fail "$mode" "error does not name the row and the log"
    else
        echo "ok $mode"
    fi
done

if ((failures > 0)); then
    echo "$failures failed" >&2
    exit 1
fi
