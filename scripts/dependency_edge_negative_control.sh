#!/usr/bin/env bash
# Prove that the one-way dependency-edge check is able to fail.
#
# The check that guards the invariant had, until this control existed, only ever
# run against a tree that satisfies it -- so a green run was evidence of nothing.
# A check that cannot fail, trusted as evidence, is worse than no check at all:
# it launders an unverified claim as a verified one.
#
# So the check is not assumed to discriminate. It is made to, on every run, for
# every forbidden tree and for both kinds of edge it claims to catch:
#
#   1. the untouched tree passes            -- the check is not simply red
#   2. a planted include under lib/ fails   -- the source-edge half discriminates
#   3. a planted CMake command under tests/
#      fails, written across several
#      physical lines                       -- the command half discriminates, and
#                                              it really does fold newlines
#   4. the working tree is left clean       -- restored by a trap, not by a
#                                              hopeful final line
#
#   scripts/dependency_edge_negative_control.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CHECK="scripts/dependency_edge_check.sh"
TREES=(benchmarks python)

PROBE_HEADER="lib/argmin/include/argmin/edge_probe.h"
PROBE_CMAKE_DIR="tests/edge_probe"

cleanup() { rm -f "$PROBE_HEADER"; rm -rf "$PROBE_CMAKE_DIR"; }
trap cleanup EXIT

failures=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }

check_status() {
    local rc=0
    bash "$CHECK" >/dev/null 2>&1 || rc=$?
    echo "$rc"
}

echo "proving the one-way dependency-edge check can fail:"

if [ "$(check_status)" -eq 0 ]; then
    pass "the untouched tree passes the check"
else
    fail "the untouched tree does not pass; the control cannot distinguish a violation from noise"
fi

for tree in "${TREES[@]}"; do
    cat > "$PROBE_HEADER" <<EOF
#include <${tree}/probe.h>
EOF
    if [ "$(check_status)" -ne 0 ]; then
        pass "$tree: a lib/ header including the tree is caught"
    else
        fail "$tree: a lib/ header including the tree is NOT caught"
    fi
    rm -f "$PROBE_HEADER"

    mkdir -p "$PROBE_CMAKE_DIR"
    cat > "$PROBE_CMAKE_DIR/CMakeLists.txt" <<EOF
target_link_libraries(edge_probe
    PRIVATE
    ${tree}_probe)
EOF
    if [ "$(check_status)" -ne 0 ]; then
        pass "$tree: a multi-line CMake command under tests/ naming the tree is caught"
    else
        fail "$tree: a multi-line CMake command under tests/ naming the tree is NOT caught"
    fi
    rm -rf "$PROBE_CMAKE_DIR"
done

cleanup
if git status --porcelain | grep -q 'edge_probe'; then
    fail "the control left planted files behind"
else
    pass "the working tree carries none of the planted files"
fi

echo
if [ "$failures" -ne 0 ]; then
    echo "the dependency-edge check is NOT proven able to fail: $failures of the control's properties do not hold"
    echo "until this passes, a green check is not evidence that the one-way edge holds"
    exit 1
fi

echo "the dependency-edge check discriminates: the untouched tree passes, and both"
echo "an included header and a folded CMake command go red for every forbidden tree"
