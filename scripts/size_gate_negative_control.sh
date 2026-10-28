#!/usr/bin/env bash
# Prove that the file-size gate is able to fail, on an over-limit file nobody registered and on a
# debt baseline that grew.
#
# The gate measures tracked files only, so an untracked plant would be invisible to it. Every plant
# therefore goes into a throwaway repository holding a copy of the real tree, and the real working
# tree is never written.
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
    (cd "$WORK" && python3 scripts/check_file_size.py "$@") >/dev/null 2>&1 || rc=$?
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

echo "proving the file-size gate can fail:"

expect 0 "the untouched copy passes"
expect 0 "the untouched copy passes against its own commit" --baseline-ref HEAD

plant_probe
expect 1 "a planted lib/ header in no register is caught"
remove_probe

echo '// probe' >> "$WORK/$recorded"
expect 4 "one line added to a file with a baseline row is caught"
restore "$recorded"

awk 'BEGIN { FS = OFS = "\t" } !done && $1 == "file" { $4 += 1; done = 1 } { print }' \
    "$WORK/$BASELINE" > "$WORK/$BASELINE.raised"
mv "$WORK/$BASELINE.raised" "$WORK/$BASELINE"
expect 8 "a raised baseline count is caught against the ref" --baseline-ref HEAD
restore "$BASELINE"

plant_probe
add_probe_row
expect 8 "a new baseline row is caught against the ref" --baseline-ref HEAD
remove_probe
restore "$BASELINE"

if [ "$(git status --porcelain)" = "$before" ]; then
    pass "the real working tree is unchanged"
else
    fail "the real working tree changed while the control ran"
fi

echo
if [ "$failures" -ne 0 ]; then
    echo "the file-size gate is NOT proven able to fail: $failures of the control's properties do not hold"
    exit 1
fi

echo "the file-size gate discriminates: the untouched copy passes, and an unregistered overage,"
echo "file growth past its baseline row, and a baseline that grew against the ref all go red"
