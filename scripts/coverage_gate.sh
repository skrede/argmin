#!/usr/bin/env bash
# Run the coverage gate locally, matching the CI invocation byte-for-byte.
#
# Builds the coverage preset, runs the suite to generate profile data, emits a
# gcovr JSON report over the argmin header tree, and runs the floor gate. The
# gcov executable defaults to `gcov`; set GCOV=gcov-14 to match the authoritative
# CI leg when that toolchain is installed.
#
#   scripts/coverage_gate.sh            # build + test + gate
#   scripts/coverage_gate.sh --no-build # reuse an existing build/coverage
#   GCOV=gcov-14 scripts/coverage_gate.sh
set -euo pipefail

cd "$(dirname "$0")/.."
GCOV="${GCOV:-gcov}"

if [ "${1:-}" != "--no-build" ]; then
    cmake --preset coverage
    cmake --build build/coverage -j"${JOBS:-4}"
    ctest --test-dir build/coverage --output-on-failure
fi

gcovr build/coverage \
    --root . \
    --gcov-executable "$GCOV" \
    --filter 'lib/argmin/include/argmin/' \
    --gcov-ignore-errors=source_not_found \
    --json coverage.json

python3 scripts/coverage_gate.py coverage.json scripts/coverage_floors.txt
