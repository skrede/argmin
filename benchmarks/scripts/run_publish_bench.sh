#!/usr/bin/env bash
# Publication-grade benchmark run wrapper.
#
# Exports single-thread BLAS / OpenMP env controls, records run provenance,
# and invokes publish_bench with an 11-seed sweep (42..52), writing to a
# UTC-ISO8601-timestamped output directory. After the bench completes the
# wrapper invokes the regression_check binary to gate the run against the
# checked-in per-cell baseline, then invokes the independent oracle
# (oracle_check) to re-validate feasibility/KKT at the returned point from
# the returned-points sidecar, propagating any non-zero exit. The oracle gate
# re-derives correctness analytically rather than trusting the solver's
# self-reported constraint_violation, so an infeasible returned point with
# green accuracy still fails the run.
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
#                            (default /tmp/argmin-publish)
#   RUN_REGRESSION_CHECK     1 to run the regression gate (default 1);
#                            0 to skip and emit publish_summary.csv only
#   REGRESSION_CHECK_BINARY  path to regression_check executable
#                            (default build/bench/benchmarks/regression_check)
#   REGRESSION_BASELINE      path to baseline CSV
#                            (default benchmarks/baselines/publication-regression.csv)
#   RUN_ORACLE_CHECK         1 to run the independent oracle after the bench
#                            (default 1); the oracle recomputes feasibility/KKT
#                            at the returned point and fails the run on any
#                            oracle_pass=fail verdict
#   ORACLE_CHECK_BINARY      path to oracle_check executable
#                            (default build/bench/benchmarks/oracle_check)
#   EXPECT_BASELINE_ELIGIBLE 1 if this run is intended for publication baseline
#                            generation (default 0)
#   REQUIRE_RELEASE_PUBLICATION
#                            1 to reject baseline-ineligible provenance before
#                            running publish_bench (default 0)
#   BUILD_TYPE_OVERRIDE_FOR_TEST
#                            override detected build type for release-guard tests
#   PROVENANCE_FILE_NAME     sidecar filename under the run directory
#                            (default provenance.json)
#
# Exit 0 on success; non-zero if the binary is missing, the bench run
# fails, or the regression gate fails. regression_check exit codes:
#   2 = pass cell breached a bound; 3 = expected-fail cell unexpectedly
#   converged.

set -euo pipefail

SEED_START="${SEED_START:-42}"
SEED_COUNT="${SEED_COUNT:-11}"
BENCH_BINARY="${BENCH_BINARY:-build/bench/benchmarks/publish_bench}"
OUT_ROOT="${OUT_ROOT:-/tmp/argmin-publish}"
RUN_REGRESSION_CHECK="${RUN_REGRESSION_CHECK:-1}"
REGRESSION_CHECK_BINARY="${REGRESSION_CHECK_BINARY:-build/bench/benchmarks/regression_check}"
REGRESSION_BASELINE="${REGRESSION_BASELINE:-benchmarks/baselines/publication-regression.csv}"
RUN_ORACLE_CHECK="${RUN_ORACLE_CHECK:-1}"
ORACLE_CHECK_BINARY="${ORACLE_CHECK_BINARY:-build/bench/benchmarks/oracle_check}"
# Oracle pass bar for the closed-form cells: the publication accuracy/cv bar
# (1e-8 / 1e-6). The committed baseline dispositions are derived at this same
# bar, so the recurring gate re-validates each pass cell against the exact
# criterion its disposition was sourced from. Control cells use the calibrated
# KKT bound, independent of these cutoffs.
ORACLE_ACCURACY_CUTOFF="${ORACLE_ACCURACY_CUTOFF:-1e-8}"
ORACLE_CV_CUTOFF="${ORACLE_CV_CUTOFF:-1e-6}"
EXPECT_BASELINE_ELIGIBLE="${EXPECT_BASELINE_ELIGIBLE:-0}"
REQUIRE_RELEASE_PUBLICATION="${REQUIRE_RELEASE_PUBLICATION:-0}"
BUILD_TYPE_OVERRIDE_FOR_TEST="${BUILD_TYPE_OVERRIDE_FOR_TEST:-}"
PROVENANCE_FILE_NAME="${PROVENANCE_FILE_NAME:-provenance.json}"

# Dolan-More post-step opt-out. Set RUN_DM_PROFILE=0 to skip the
# performance-profile generation; the bench run + any other post-steps
# still execute. The Python interpreter and script path are also
# overridable for cross-machine portability.
RUN_DM_PROFILE="${RUN_DM_PROFILE:-1}"
DM_PROFILE_SCRIPT="${DM_PROFILE_SCRIPT:-benchmarks/scripts/dm_profile.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# R5 thread control: single-threaded on every BLAS surface for timing stability.
# Apple Silicon caveat: VECLIB_MAXIMUM_THREADS only controls CPU cores; the AMX
# matrix co-processor scheduling is opaque to env vars. Disclosed in the
# methodology document, not masked here.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

have_command() {
    command -v "$1" >/dev/null 2>&1
}

absolute_path() {
    if have_command realpath; then
        realpath "$1" 2>/dev/null && return 0
    fi
    if have_command readlink; then
        readlink -f "$1" 2>/dev/null && return 0
    fi
    printf '%s\n' "$1"
}

cache_value() {
    local key="$1"
    local line=""
    if [ -f "${CMAKE_CACHE}" ]; then
        line="$(grep -E "^${key}(:[^=]+)?=" "${CMAKE_CACHE}" | tail -n 1 || true)"
    fi
    if [ -n "${line}" ]; then
        printf '%s\n' "${line#*=}"
    else
        printf '%s\n' "unknown"
    fi
}

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_string() {
    printf '"%s"' "$(json_escape "$1")"
}

json_bool() {
    if [ "$1" = "true" ]; then
        printf 'true'
    else
        printf 'false'
    fi
}

host_identifier() {
    if have_command hostname; then
        hostname 2>/dev/null && return 0
    fi
    uname -n 2>/dev/null && return 0
    printf '%s\n' "unknown"
}

cpu_model() {
    local model=""
    if [ -r /proc/cpuinfo ]; then
        model="$(awk -F: '/model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    elif have_command sysctl; then
        model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    fi
    if [ -n "${model}" ]; then
        printf '%s\n' "${model}"
    else
        printf '%s\n' "unknown"
    fi
}

command_first_line() {
    local cmd="$1"
    local out=""
    if have_command "${cmd}"; then
        shift
        out="$("${cmd}" "$@" 2>/dev/null | sed -n '1p' || true)"
    fi
    if [ -n "${out}" ]; then
        printf '%s\n' "${out}"
    else
        printf '%s\n' "unavailable"
    fi
}

binary_hash() {
    if have_command sha256sum; then
        sha256sum "$1" | awk '{print $1}'
    elif have_command shasum; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf '%s\n' "unavailable"
    fi
}

affinity_policy() {
    local out=""
    if have_command taskset; then
        out="$(taskset -pc "$$" 2>&1 || true)"
        printf '%s\n' "${out}"
    else
        printf '%s\n' "taskset unavailable"
    fi
}

governor_policy() {
    local out=""
    if have_command cpupower; then
        out="$(cpupower frequency-info -p 2>&1 || true)"
        if [ -n "${out}" ]; then
            printf '%s\n' "${out}" | tr '\n' ';'
            printf '\n'
            return 0
        fi
    fi
    printf '%s\n' "cpupower unavailable"
}

turbo_policy() {
    if [ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        case "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" in
            0) printf '%s\n' "intel_pstate turbo enabled" ;;
            1) printf '%s\n' "intel_pstate turbo disabled" ;;
            *) printf '%s\n' "intel_pstate turbo unknown" ;;
        esac
    elif [ -r /sys/devices/system/cpu/cpufreq/boost ]; then
        case "$(cat /sys/devices/system/cpu/cpufreq/boost)" in
            0) printf '%s\n' "cpufreq boost disabled" ;;
            1) printf '%s\n' "cpufreq boost enabled" ;;
            *) printf '%s\n' "cpufreq boost unknown" ;;
        esac
    else
        printf '%s\n' "turbo policy unavailable"
    fi
}

compute_wall_gate_policy() {
    if [ "${RUN_REGRESSION_CHECK}" != "1" ]; then
        printf '%s\n' "not_run"
    elif [ "${REGRESSION_CHECK_DISABLE_WALL_GATE:-0}" = "1" ]; then
        printf '%s\n' "disabled_advisory"
    else
        printf '%s\n' "enforced"
    fi
}

write_provenance() {
    {
        printf '{\n'
        printf '  "schema": "argmin.publish_bench.provenance.v1",\n'
        printf '  "provenance_id": '; json_string "${PROVENANCE_ID}"; printf ',\n'
        printf '  "run_id": '; json_string "${PROVENANCE_ID}"; printf ',\n'
        printf '  "timestamp_utc": '; json_string "${TIMESTAMP}"; printf ',\n'
        printf '  "host_identifier": '; json_string "${HOST_IDENTIFIER}"; printf ',\n'
        printf '  "cpu_model": '; json_string "${CPU_MODEL}"; printf ',\n'
        printf '  "benchmark_binary": '; json_string "${BENCH_ABS}"; printf ',\n'
        printf '  "benchmark_binary_hash": '; json_string "${BENCH_HASH}"; printf ',\n'
        printf '  "build_type": '; json_string "${BUILD_TYPE}"; printf ',\n'
        printf '  "cmake_build_type": '; json_string "${BUILD_TYPE}"; printf ',\n'
        printf '  "build": {\n'
        printf '    "type": '; json_string "${BUILD_TYPE}"; printf ',\n'
        printf '    "cmake_cache": '; json_string "${CMAKE_CACHE}"; printf ',\n'
        printf '    "cxx_compiler": '; json_string "${CXX_COMPILER}"; printf ',\n'
        printf '    "cxx_compiler_version": '; json_string "${CXX_COMPILER_VERSION}"; printf ',\n'
        printf '    "cxx_flags": '; json_string "${CXX_FLAGS}"; printf ',\n'
        printf '    "cxx_flags_release": '; json_string "${CXX_FLAGS_RELEASE}"; printf '\n'
        printf '  },\n'
        printf '  "SEED_START": '; json_string "${SEED_START}"; printf ',\n'
        printf '  "SEED_COUNT": '; json_string "${SEED_COUNT}"; printf ',\n'
        printf '  "seeds": {\n'
        printf '    "start": %s,\n' "${SEED_START}"
        printf '    "count": %s\n' "${SEED_COUNT}"
        printf '  },\n'
        printf '  "thread_environment": {\n'
        printf '    "OMP_NUM_THREADS": '; json_string "${OMP_NUM_THREADS}"; printf ',\n'
        printf '    "OPENBLAS_NUM_THREADS": '; json_string "${OPENBLAS_NUM_THREADS}"; printf ',\n'
        printf '    "MKL_NUM_THREADS": '; json_string "${MKL_NUM_THREADS}"; printf ',\n'
        printf '    "VECLIB_MAXIMUM_THREADS": '; json_string "${VECLIB_MAXIMUM_THREADS}"; printf '\n'
        printf '  },\n'
        printf '  "optional_libraries": {\n'
        printf '    "ARGMIN_BENCH_NLOPT": '; json_string "${ARGMIN_BENCH_NLOPT_CACHE}"; printf ',\n'
        printf '    "ARGMIN_BENCH_CERES": '; json_string "${ARGMIN_BENCH_CERES_CACHE}"; printf ',\n'
        printf '    "ARGMIN_BENCH_DLIB": '; json_string "${ARGMIN_BENCH_DLIB_CACHE}"; printf ',\n'
        printf '    "ARGMIN_BENCH_LIBCMAES": '; json_string "${ARGMIN_BENCH_LIBCMAES_CACHE}"; printf ',\n'
        printf '    "ARGMIN_BENCH_IPOPT": '; json_string "${ARGMIN_BENCH_IPOPT_CACHE}"; printf '\n'
        printf '  },\n'
        printf '  "environment_controls": {\n'
        printf '    "affinity_policy": '; json_string "${AFFINITY_POLICY}"; printf ',\n'
        printf '    "governor_policy": '; json_string "${GOVERNOR_POLICY}"; printf ',\n'
        printf '    "turbo_policy": '; json_string "${TURBO_POLICY}"; printf '\n'
        printf '  },\n'
        printf '  "wall_gate_policy": '; json_string "${WALL_GATE_POLICY}"; printf ',\n'
        printf '  "run_regression_check": '; json_string "${RUN_REGRESSION_CHECK}"; printf ',\n'
        printf '  "run_dm_profile": '; json_string "${RUN_DM_PROFILE}"; printf ',\n'
        printf '  "baseline_valid": '; json_bool "${BASELINE_VALID}"; printf ',\n'
        printf '  "baseline_invalid_reason": '; json_string "${BASELINE_INVALID_REASON}"; printf '\n'
        printf '}\n'
    } > "${PROVENANCE_FILE}"
}

if [ ! -x "${BENCH_BINARY}" ]; then
    echo "error: BENCH_BINARY=${BENCH_BINARY} not found or not executable" >&2
    echo "hint: cmake --build build/bench --target publish_bench" >&2
    exit 1
fi

# UTC ISO 8601, filesystem-safe (no colons). "2026-04-22T14-30-00Z".
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_ROOT}/${TIMESTAMP}"
PROVENANCE_FILE="${OUT_DIR}/${PROVENANCE_FILE_NAME}"
mkdir -p "${OUT_DIR}"
printf '%s\n' "${OUT_DIR}" > "${OUT_ROOT}/latest-run.txt"

BENCH_ABS="$(absolute_path "${BENCH_BINARY}")"
BINARY_DIR="$(dirname "${BENCH_ABS}")"
BUILD_DIR="$(dirname "${BINARY_DIR}")"
CMAKE_CACHE="${CMAKE_CACHE:-${BUILD_DIR}/CMakeCache.txt}"
if [ ! -f "${CMAKE_CACHE}" ] && [ -f "${BINARY_DIR}/CMakeCache.txt" ]; then
    CMAKE_CACHE="${BINARY_DIR}/CMakeCache.txt"
fi

BUILD_TYPE="$(cache_value CMAKE_BUILD_TYPE)"
if [ -n "${BUILD_TYPE_OVERRIDE_FOR_TEST}" ]; then
    BUILD_TYPE="${BUILD_TYPE_OVERRIDE_FOR_TEST}"
fi
CXX_COMPILER="$(cache_value CMAKE_CXX_COMPILER)"
if [ "${CXX_COMPILER}" = "unknown" ]; then
    CXX_COMPILER="${CXX:-c++}"
fi
CXX_COMPILER_VERSION="$(command_first_line "${CXX_COMPILER}" --version)"
CXX_FLAGS="$(cache_value CMAKE_CXX_FLAGS)"
CXX_FLAGS_RELEASE="$(cache_value CMAKE_CXX_FLAGS_RELEASE)"
ARGMIN_BENCH_NLOPT_CACHE="$(cache_value ARGMIN_BENCH_NLOPT)"
ARGMIN_BENCH_CERES_CACHE="$(cache_value ARGMIN_BENCH_CERES)"
ARGMIN_BENCH_DLIB_CACHE="$(cache_value ARGMIN_BENCH_DLIB)"
ARGMIN_BENCH_LIBCMAES_CACHE="$(cache_value ARGMIN_BENCH_LIBCMAES)"
ARGMIN_BENCH_IPOPT_CACHE="$(cache_value ARGMIN_BENCH_IPOPT)"

PROVENANCE_ID="publish-${TIMESTAMP}"
HOST_IDENTIFIER="$(host_identifier)"
CPU_MODEL="$(cpu_model)"
BENCH_HASH="$(binary_hash "${BENCH_BINARY}")"
AFFINITY_POLICY="$(affinity_policy)"
GOVERNOR_POLICY="$(governor_policy)"
TURBO_POLICY="$(turbo_policy)"
WALL_GATE_POLICY="$(compute_wall_gate_policy)"
BASELINE_VALID="false"
BASELINE_INVALID_REASON="not requested for publication baseline generation"

if [ "${EXPECT_BASELINE_ELIGIBLE}" = "1" ]; then
    BASELINE_VALID="true"
    BASELINE_INVALID_REASON=""
    if [ "${BUILD_TYPE}" != "Release" ]; then
        BASELINE_VALID="false"
        BASELINE_INVALID_REASON="build_type is ${BUILD_TYPE}, expected Release"
    elif [ "${WALL_GATE_POLICY}" != "enforced" ]; then
        BASELINE_VALID="false"
        BASELINE_INVALID_REASON="wall_gate_policy is ${WALL_GATE_POLICY}, expected enforced"
    fi
fi

write_provenance

if [ "${REQUIRE_RELEASE_PUBLICATION}" = "1" ] && [ "${BASELINE_VALID}" != "true" ]; then
    echo "error: publication-baseline run is not eligible: ${BASELINE_INVALID_REASON}" >&2
    echo "provenance: ${PROVENANCE_FILE}" >&2
    exit 2
fi

echo "run_publish_bench: output=${OUT_DIR}"
echo "run_publish_bench: provenance=${PROVENANCE_FILE}"
echo "run_publish_bench: latest=${OUT_ROOT}/latest-run.txt"
echo "env: OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS}"
echo "seeds: ${SEED_START}..$((SEED_START + SEED_COUNT - 1))"
echo "binary: ${BENCH_BINARY}"
echo "build_type: ${BUILD_TYPE}"
echo "wall_gate_policy: ${WALL_GATE_POLICY}"
echo

ARGMIN_ONLY_FLAG=()
if [ "${ARGMIN_ONLY:-0}" = "1" ]; then
    ARGMIN_ONLY_FLAG=(--argmin-only)
fi

"${BENCH_BINARY}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --output-dir "${OUT_DIR}" \
    "${ARGMIN_ONLY_FLAG[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

# Independent oracle first: recompute feasibility/KKT at each returned point
# and write the per-cell verdict. regression_check then witnesses pass-cell
# correctness from this verdict (a committed pass cell breaches on an
# oracle_pass=fail or a missing verdict) instead of the solver's self-reported
# status/accuracy, so an infeasible returned point with a green self-report
# still trips the gate. The verdict is produced at the publication accuracy/cv
# bar, the same bar the committed dispositions are derived at.
if [ "${RUN_ORACLE_CHECK}" = "1" ]; then
    if [ ! -x "${ORACLE_CHECK_BINARY}" ]; then
        echo "error: ORACLE_CHECK_BINARY=${ORACLE_CHECK_BINARY} not found or not executable" >&2
        echo "hint: cmake --build build/bench --target oracle_check" >&2
        exit 1
    fi
    if [ ! -f "${OUT_DIR}/publish_returned_points.csv" ]; then
        echo "error: returned-points sidecar ${OUT_DIR}/publish_returned_points.csv not found" >&2
        echo "the oracle re-validates the returned point; a run without the sidecar cannot be certified" >&2
        exit 1
    fi
    echo
    echo "run_publish_bench: oracle_check on ${OUT_DIR}/publish_summary.csv + publish_returned_points.csv"
    set +e
    "${ORACLE_CHECK_BINARY}" \
        "${OUT_DIR}/publish_summary.csv" \
        "${OUT_DIR}/publish_returned_points.csv" \
        "${OUT_DIR}/oracle_verdict.csv" \
        --accuracy-cutoff "${ORACLE_ACCURACY_CUTOFF}" \
        --cv-cutoff "${ORACLE_CV_CUTOFF}" \
        2>&1 | tee "${OUT_DIR}/oracle_check.log"
    oracle_rc="${PIPESTATUS[0]}"
    set -e
    if [ "${oracle_rc}" -ne 0 ]; then
        echo "run_publish_bench: oracle_check exited rc=${oracle_rc} (structural fail-closed)" >&2
        exit "${oracle_rc}"
    fi
fi

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
    if [ "${RUN_ORACLE_CHECK}" = "1" ]; then
        # Oracle-witnessed pass correctness: a committed pass cell breaches on an
        # oracle_pass=fail or a missing verdict, never on self-reported status.
        "${REGRESSION_CHECK_BINARY}" \
            "${OUT_DIR}/publish_summary.csv" \
            "${REGRESSION_BASELINE}" \
            --oracle-verdict "${OUT_DIR}/oracle_verdict.csv" \
            2>&1 | tee "${OUT_DIR}/regression_check.log"
    else
        # Legacy self-reported-status correctness gate (no verdict supplied).
        "${REGRESSION_CHECK_BINARY}" \
            "${OUT_DIR}/publish_summary.csv" \
            "${REGRESSION_BASELINE}" \
            2>&1 | tee "${OUT_DIR}/regression_check.log"
    fi
    rc="${PIPESTATUS[0]}"
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "run_publish_bench: regression_check exited rc=${rc}" >&2
        exit "${rc}"
    fi
fi

if [ "${RUN_DM_PROFILE}" = "1" ]; then
    if [ ! -f "${DM_PROFILE_SCRIPT}" ]; then
        echo "error: DM_PROFILE_SCRIPT=${DM_PROFILE_SCRIPT} not found" >&2
        exit 1
    fi
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "error: PYTHON_BIN=${PYTHON_BIN} not found on PATH" >&2
        exit 1
    fi
    mkdir -p "${OUT_DIR}/profiles"
    echo
    echo "run_publish_bench: dm_profile on ${OUT_DIR} -> ${OUT_DIR}/profiles/"
    set +e
    "${PYTHON_BIN}" "${DM_PROFILE_SCRIPT}" \
        --run-dir "${OUT_DIR}" \
        --output-dir "${OUT_DIR}/profiles" \
        2>&1 | tee "${OUT_DIR}/dm_profile.log"
    dm_rc=${PIPESTATUS[0]}
    set -e
    if [ "${dm_rc}" -ne 0 ]; then
        echo "run_publish_bench: dm_profile exited rc=${dm_rc}" >&2
        exit "${dm_rc}"
    fi
fi

echo
echo "run_publish_bench: done"
echo "output:"
echo "  ${OUT_DIR}/publish_summary.csv"
echo "  ${OUT_DIR}/traces/"
echo "  ${OUT_DIR}/${PROVENANCE_FILE_NAME}"
echo "  ${OUT_DIR}/run.log"
if [ "${RUN_REGRESSION_CHECK}" = "1" ]; then
    echo "  ${OUT_DIR}/regression_check.log"
fi
if [ "${RUN_ORACLE_CHECK}" = "1" ]; then
    echo "  ${OUT_DIR}/oracle_verdict.csv"
    echo "  ${OUT_DIR}/oracle_check.log"
fi
if [ "${RUN_DM_PROFILE}" = "1" ]; then
    echo "  ${OUT_DIR}/dm_profile.log"
    echo "  ${OUT_DIR}/profiles/"
fi
