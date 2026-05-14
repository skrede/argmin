#!/usr/bin/env bash
# Publication-grade benchmark run wrapper.
#
# Exports single-thread BLAS / OpenMP env controls and invokes publish_bench
# with an 11-seed sweep (42..52), writing to a UTC-ISO8601-timestamped
# output directory under .planning/benchmarks/publish/. After the bench
# completes the wrapper invokes the regression_check binary to gate the
# run against the checked-in per-cell baseline, propagating its exit code.
#
# Usage:
#   bash benchmarks/scripts/run_publish_bench.sh
#   SEED_START=42 SEED_COUNT=11 BENCH_BINARY=build/bench/benchmarks/publish_bench \
#       bash benchmarks/scripts/run_publish_bench.sh
#
# Env-var overrides (all optional):
#   SEED_START               first seed (default 42)
#   SEED_COUNT               number of seeds (default 11)
#   BENCH_BINARY             path to publish_bench executable
#                            (default build/bench/benchmarks/publish_bench)
#   OUT_ROOT                 root directory for timestamped output
#                            (default .planning/benchmarks/publish)
#   RUN_REGRESSION_CHECK     1 to run the regression gate (default 1);
#                            0 to skip and emit publish_summary.csv only
#   REGRESSION_CHECK_BINARY  path to regression_check executable
#                            (default build/bench/benchmarks/regression_check)
#   REGRESSION_BASELINE      path to baseline CSV
#                            (default benchmarks/baselines/v0.3.0-regression.csv)
#
# Exit 0 on success; non-zero if the binary is missing, the bench run
# fails, or the regression gate fails. regression_check exit codes:
#   2 = expected:pass cell breached a bound; 3 = expected:shouldfail
#   cell unexpectedly converged.

set -euo pipefail

SEED_START="${SEED_START:-42}"
SEED_COUNT="${SEED_COUNT:-11}"
BENCH_BINARY="${BENCH_BINARY:-build/bench/benchmarks/publish_bench}"
OUT_ROOT="${OUT_ROOT:-.planning/benchmarks/publish}"
RUN_REGRESSION_CHECK="${RUN_REGRESSION_CHECK:-1}"
REGRESSION_CHECK_BINARY="${REGRESSION_CHECK_BINARY:-build/bench/benchmarks/regression_check}"
REGRESSION_BASELINE="${REGRESSION_BASELINE:-benchmarks/baselines/v0.3.0-regression.csv}"

# R5 thread control: single-threaded on every BLAS surface for timing stability.
# Apple Silicon caveat: VECLIB_MAXIMUM_THREADS only controls CPU cores; the AMX
# matrix co-processor scheduling is opaque to env vars. Disclosed in the
# methodology document, not masked here.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ ! -x "${BENCH_BINARY}" ]; then
    echo "error: BENCH_BINARY=${BENCH_BINARY} not found or not executable" >&2
    echo "hint: cmake --build build/bench --target publish_bench" >&2
    exit 1
fi

# UTC ISO 8601, filesystem-safe (no colons). "2026-04-22T14-30-00Z".
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_ROOT}/${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

echo "run_publish_bench: output=${OUT_DIR}"
echo "env: OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS}"
echo "seeds: ${SEED_START}..$((SEED_START + SEED_COUNT - 1))"
echo "binary: ${BENCH_BINARY}"
echo

"${BENCH_BINARY}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"

if [ "${RUN_REGRESSION_CHECK}" = "1" ]; then
    if [ ! -x "${REGRESSION_CHECK_BINARY}" ]; then
        echo "error: REGRESSION_CHECK_BINARY=${REGRESSION_CHECK_BINARY} not found or not executable" >&2
        echo "hint: cmake --build build/bench --target regression_check" >&2
        exit 1
    fi
    if [ ! -f "${REGRESSION_BASELINE}" ]; then
        echo "error: REGRESSION_BASELINE=${REGRESSION_BASELINE} not found" >&2
        exit 1
    fi
    echo
    echo "run_publish_bench: regression_check on ${OUT_DIR}/publish_summary.csv vs ${REGRESSION_BASELINE}"
    set +e
    "${REGRESSION_CHECK_BINARY}" \
        "${OUT_DIR}/publish_summary.csv" \
        "${REGRESSION_BASELINE}" \
        2>&1 | tee "${OUT_DIR}/regression_check.log"
    rc="${PIPESTATUS[0]}"
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "run_publish_bench: regression_check exited rc=${rc}" >&2
        exit "${rc}"
    fi
fi

echo
echo "run_publish_bench: done"
echo "output:"
echo "  ${OUT_DIR}/publish_summary.csv"
echo "  ${OUT_DIR}/traces/"
echo "  ${OUT_DIR}/run.log"
if [ "${RUN_REGRESSION_CHECK}" = "1" ]; then
    echo "  ${OUT_DIR}/regression_check.log"
fi
