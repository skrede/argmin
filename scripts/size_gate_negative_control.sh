#!/usr/bin/env bash
# Prove that the size gates are able to fail, on an over-limit file or function nobody registered and on
# a debt baseline that grew.
#
# The gate measures tracked files only, so an untracked plant would be invisible to it. Every plant
# therefore goes into a throwaway repository holding a copy of the real tree, and the real working
# tree is never written.
#
# The function plants run a real clang-tidy, found through CLANG_TIDY, then clang-tidy-18, then the
# llvm18 install path; without one they are skipped visibly. ARGMIN_TIDY_EXTRA_ARG, when set, is passed
# to the compiler, e.g. --gcc-install-dir=... where clang-18 cannot parse the default libstdc++.
#
#   scripts/size_gate_negative_control.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PROBE="lib/argmin/include/argmin/size_gate_probe.h"
BASELINE="scripts/size_baseline.tsv"
WORK=""

cleanup() { if [ -n "$WORK" ]; then rm -rf "$WORK"; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

failures=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }

in_copy() { git -C "$WORK" -c user.name=control -c user.email=control@localhost \
    -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }

gate_status() {
    local rc=0
    (cd "$WORK" && "$@") >/dev/null 2>&1 || rc=$?
    echo "$rc"
}

expect() {
    local want="$1" label="$2" got
    shift 2
    got="$(gate_status "$@")"
    if [ "$got" -eq "$want" ]; then pass "$label (exit $got)"; else fail "$label: exit $got, wanted $want"; fi
}

plant_probe() { for _ in $(seq 201); do echo '// probe'; done > "$WORK/$PROBE"; in_copy add "$PROBE"; }
remove_probe() { in_copy rm -q --cached "$PROBE"; rm -f "$WORK/$PROBE"; }
restore() { in_copy checkout -q -- "$@"; }

plant_function() {
    {
        echo '#ifndef HPP_GUARD_ARGMIN_SIZE_GATE_PROBE_H'
        echo '#define HPP_GUARD_ARGMIN_SIZE_GATE_PROBE_H'
        echo 'inline void size_gate_probe()'
        echo '{'
        for _ in $(seq 29); do echo '    static_cast<void>(0);'; done
        echo '}'
        echo '#endif'
    } > "$WORK/$PROBE"
    in_copy add "$PROBE"
}

find_tidy() {
    local candidate
    for candidate in "${CLANG_TIDY:-}" clang-tidy-18 /usr/lib/llvm18/bin/clang-tidy; do
        if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
}

add_probe_row() {
    python3 - "$WORK/$BASELINE" "$PROBE" <<'PY'
import sys
path, probe = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
head = [line for line in lines if line.startswith("#")]
rows = [line.split("\t") for line in lines if line and not line.startswith("#")]
rows.append(["file", probe, "-", "201"])
rows.sort(key=lambda row: (row[0], row[1], row[2], int(row[3])))
open(path, "w").write("\n".join(head + ["\t".join(row) for row in rows]) + "\n")
PY
}

before="$(git status --porcelain)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/argmin-size-gate.XXXXXX")"
git ls-files -z -- lib scripts EXCEPTIONS.md | tar --null -T - -cf - | tar -xf - -C "$WORK"
in_copy init -q
in_copy add -A
in_copy commit -q -m "copy of the real tree"

recorded="$(awk -F'\t' '$1 == "file" { print $2; exit }' "$WORK/$BASELINE")"
FILE_GATE=(python3 scripts/check_file_size.py)
TIDY="$(find_tidy)"
FUNCTION_GATE=(python3 scripts/check_function_size.py --clang-tidy "$TIDY")
if [ -n "${ARGMIN_TIDY_EXTRA_ARG:-}" ]; then FUNCTION_GATE+=("--extra-arg=$ARGMIN_TIDY_EXTRA_ARG"); fi

echo "proving the file-size gate can fail:"

expect 0 "the untouched copy passes" "${FILE_GATE[@]}"
expect 0 "the untouched copy passes against its own commit" "${FILE_GATE[@]}" --baseline-ref HEAD

plant_probe
expect 1 "a planted lib/ header in no register is caught" "${FILE_GATE[@]}"
remove_probe

echo '// probe' >> "$WORK/$recorded"
expect 4 "one line added to a file with a baseline row is caught" "${FILE_GATE[@]}"
restore "$recorded"

awk 'BEGIN { FS = OFS = "\t" } !done && $1 == "file" { $4 += 1; done = 1 } { print }' \
    "$WORK/$BASELINE" > "$WORK/$BASELINE.raised"
mv "$WORK/$BASELINE.raised" "$WORK/$BASELINE"
expect 8 "a raised baseline count is caught against the ref" "${FILE_GATE[@]}" --baseline-ref HEAD
restore "$BASELINE"

plant_probe
add_probe_row
expect 8 "a new baseline row is caught against the ref" "${FILE_GATE[@]}" --baseline-ref HEAD
remove_probe
restore "$BASELINE"

echo "proving the function-size gate can fail:"
if [ -z "$TIDY" ]; then
    echo "  SKIP  function plants: no clang-tidy found; set CLANG_TIDY to run them"
else
    expect 0 "the untouched copy passes the function gate" "${FUNCTION_GATE[@]}" --baseline-ref HEAD
    plant_function
    expect 1 "a planted 30-line function in no register is caught by $TIDY" "${FUNCTION_GATE[@]}"
    remove_probe
    awk 'BEGIN { FS = OFS = "\t" } !done && $1 == "function" { $4 -= 1; done = 1 } { print }' \
        "$WORK/$BASELINE" > "$WORK/$BASELINE.lowered"
    mv "$WORK/$BASELINE.lowered" "$WORK/$BASELINE"
    expect 4 "a function one line past its lowered baseline row is caught" "${FUNCTION_GATE[@]}"
    restore "$BASELINE"
fi

if [ "$(git status --porcelain)" = "$before" ]; then
    pass "the real working tree is unchanged"
else
    fail "the real working tree changed while the control ran"
fi

echo
if [ "$failures" -ne 0 ]; then
    echo "the size gates are NOT proven able to fail: $failures of the control's properties do not hold"
    exit 1
fi

echo "the size gates discriminate: the untouched copy passes, and an unregistered overage, growth past"
echo "a baseline row, and a baseline that grew against the ref all go red"
