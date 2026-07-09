#!/usr/bin/env bash
set -euo pipefail

old_rev="HEAD~1"
new_rev="HEAD"
build_root="build/nw-sqp-profile-compare"
output=""
cases="double_integrator_nx4_n20,double_integrator_nx8_n20"
repetitions="1"
build_type="Release"
dry_run_harness="0"

usage() {
    cat <<'USAGE'
usage: profile_nw_sqp_cost_jump.sh [options]

Options:
  --old-rev REV          revision used as the left side
  --new-rev REV          revision used as the right side
  --build-root DIR       directory for generated harness builds
  --output PATH          merged CSV output path
  --cases LIST           comma-separated case list, or all
  --repetitions N        harness repetitions per case and driver
  --build-type TYPE      CMake build type for external harnesses
  --dry-run-harness      write provenance-only rows without building revisions
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --old-rev)
            old_rev="$2"
            shift 2
            ;;
        --new-rev)
            new_rev="$2"
            shift 2
            ;;
        --build-root)
            build_root="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --cases)
            cases="$2"
            shift 2
            ;;
        --repetitions)
            repetitions="$2"
            shift 2
            ;;
        --build-type)
            build_type="$2"
            shift 2
            ;;
        --dry-run-harness)
            dry_run_harness="1"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$output" ]]; then
    output="$build_root/nw_sqp_budget_profile.csv"
fi

repo_root="$(git rev-parse --show-toplevel)"
harness_source="$repo_root/benchmarks/nw_sqp_budget_profile.cpp"
harness_project="$build_root/external-harness"

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf '%s\n' unknown
    fi
}

csv_text() {
    printf '"%s"' "$(printf '%s' "$1" | sed 's/"/""/g')"
}

emit_dry_row() {
    local rev="$1"
    local driver="$2"
    csv_text "$rev"
    printf ','
    csv_text "$harness_hash"
    printf ','
    csv_text "${CXX:-c++}"
    printf ','
    csv_text "$build_type"
    printf ','
    csv_text "double_integrator_nx4_n20"
    printf ','
    csv_text "$driver"
    printf ',0,0,0,0,0,0,'
    csv_text "dry_run"
    printf ','
    csv_text "external_harness;dry_run"
    printf '\n'
}

harness_hash="$(hash_file "$harness_source")"
mkdir -p "$build_root" "$(dirname "$output")"

if [[ "$dry_run_harness" == "1" ]]; then
    {
        printf 'library_rev,harness_hash,compiler,build_type,case,driver_path,repetition,iterations,solve_wall_us,per_iteration_us,objective,constraint_violation,status,notes\n'
        emit_dry_row "$old_rev" "step_budget_solver"
        emit_dry_row "$new_rev" "time_budget_solver"
    } > "$output"
    echo "wrote dry-run harness CSV: $output"
    exit 0
fi

write_external_project() {
    mkdir -p "$harness_project"
    cat > "$harness_project/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.28)
project(argmin_nw_sqp_external_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ARGMIN_SOURCE_DIR)
    message(FATAL_ERROR "ARGMIN_SOURCE_DIR is required")
endif()
if(NOT DEFINED HARNESS_SOURCE)
    message(FATAL_ERROR "HARNESS_SOURCE is required")
endif()

find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 was not found")
    endif()
endif()

get_filename_component(HARNESS_BENCH_DIR "${HARNESS_SOURCE}" DIRECTORY)
add_executable(nw_sqp_budget_profile "${HARNESS_SOURCE}")
target_include_directories(nw_sqp_budget_profile PRIVATE
    "${HARNESS_BENCH_DIR}"
    "${ARGMIN_SOURCE_DIR}/lib/argmin/include"
)
target_compile_definitions(nw_sqp_budget_profile PRIVATE
    EIGEN_STACK_ALLOCATION_LIMIT=1048576
)
if(Eigen3_FOUND)
    target_link_libraries(nw_sqp_budget_profile PRIVATE Eigen3::Eigen)
else()
    target_include_directories(nw_sqp_budget_profile PRIVATE "${EIGEN3_INCLUDE_DIR}")
endif()
target_compile_options(nw_sqp_budget_profile PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native -fno-math-errno -fno-trapping-math
    >
)
CMAKE
}

sanitize_label() {
    printf '%s' "$1" | tr -c '[:alnum:]_.-' '_'
}

run_revision() {
    local rev="$1"
    local label
    local worktree
    local build_dir
    local row_csv

    label="$(sanitize_label "$rev")"
    worktree="$build_root/worktree-$label"
    build_dir="$build_root/build-$label"
    row_csv="$build_root/rows-$label.csv"

    if [[ -e "$worktree" || -e "$build_dir" || -e "$row_csv" ]]; then
        echo "refusing to overwrite existing generated path for $rev" >&2
        exit 3
    fi

    git worktree add --detach "$worktree" "$rev" >&2
    cmake -S "$harness_project" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DARGMIN_SOURCE_DIR="$worktree" \
        -DHARNESS_SOURCE="$harness_source" >&2
    cmake --build "$build_dir" --target nw_sqp_budget_profile >&2
    "$build_dir/nw_sqp_budget_profile" \
        --cases "$cases" \
        --repetitions "$repetitions" \
        --output "$row_csv" \
        --library-rev "$rev" \
        --harness-hash "$harness_hash" \
        --compiler "${CXX:-c++}" \
        --build-type "$build_type" \
        --notes "external_harness"

    printf '%s\n' "$row_csv"
}

write_external_project
old_csv="$(run_revision "$old_rev")"
new_csv="$(run_revision "$new_rev")"

head -n 1 "$old_csv" > "$output"
tail -n +2 "$old_csv" >> "$output"
tail -n +2 "$new_csv" >> "$output"
echo "wrote revision comparison CSV: $output"
