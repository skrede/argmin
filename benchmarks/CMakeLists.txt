cmake_minimum_required(VERSION 3.28)
project(argmin_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Google Benchmark via FetchContent (for parametric sweep harness on (policy, mode, k)).
include(FetchContent)
if(ARGMIN_CMAKE_FETCH_DEPS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
        EXCLUDE_FROM_ALL
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
else()
    find_package(benchmark CONFIG REQUIRED)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ARGMIN_LTO_SUPPORTED)

# Allocation-gate target factory. Each gate target pairs a micro_* driver with
# the shared alloc_trace_main.cpp, which owns the counting eigen_assert, the
# atomic counters, the strong-symbol C-allocator interposition, and main().
# Targets build in the shipping Release config (-O3 via the Release flags plus
# -march=native) with EIGEN_RUNTIME_NO_MALLOC so Eigen's runtime malloc gate
# routes into the report-and-continue counter rather than aborting. This gate
# is the load-bearing embeddability instrument: it observes Eigen's
# aligned_malloc traffic, which the retired global-operator-new counter could
# not see. LTO is intentionally left off here so cross-TU optimization cannot
# reason around the interposed allocator or the report-and-continue gate.
function(argmin_add_alloc_gate _target _source _test_name)
    add_executable(${_target} ${_source} alloc_trace_main.cpp)
    target_link_libraries(${_target} PRIVATE argmin::argmin)
    if(NLopt_FOUND)
        target_link_libraries(${_target} PRIVATE NLopt::nlopt)
    endif()
    target_compile_definitions(${_target} PRIVATE
        ARGMIN_BENCH_TRACE_ALLOC=1
        EIGEN_RUNTIME_NO_MALLOC
        ${ARGN})
    target_compile_options(${_target} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    add_test(NAME ${_test_name} COMMAND ${_target})
    set_tests_properties(${_test_name} PROPERTIES LABELS "alloc-gate")
endfunction()

# Main benchmark executable (always builds).
add_executable(argmin_bench
    benchmark_driver.cpp
)
target_link_libraries(argmin_bench PRIVATE argmin::argmin benchmark::benchmark)

target_compile_options(argmin_bench PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(argmin_bench PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# publish_bench -- publication-grade Dolan-More measurement driver.
add_executable(publish_bench
    publish_driver.cpp
)
target_link_libraries(publish_bench PRIVATE argmin::argmin benchmark::benchmark)

target_compile_options(publish_bench PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(publish_bench PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Strided active-set multiplier-reest sweep harness. Parametric on
# (policy x mode x problem x k); emits Google Benchmark native JSON
# via --benchmark_out for the per-mode default selection of
# default_multiplier_reest_every_k across the four line-search SQP
# policies on HS026 / HS028 / HS043 / HS071.
add_executable(multiplier_reest_sweep multiplier_reest_sweep.cpp)
target_link_libraries(multiplier_reest_sweep PRIVATE argmin::argmin benchmark::benchmark)

target_compile_options(multiplier_reest_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(multiplier_reest_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# ISRES seed-stability sweep harness. Drives the production isres_policy
# across seeds 1..100 on the `simple constrained 2D` regression test
# problem (replicates tests/unit/isres_test.cpp). Emits a JSON record
# with per-seed terminal objective_value / cv / x_max_coeff /
# converged-flag / status; selects the lowest seed where
# `objective_value < 1.0` (the test's CHECK condition). Non-ctest;
# one-shot; consumes nothing the regular bench targets do not already
# pull in.
add_executable(isres_seed_sweep isres_seed_sweep.cpp)
target_link_libraries(isres_seed_sweep PRIVATE argmin::argmin)

target_compile_options(isres_seed_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(isres_seed_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# MMA subproblem-regularizer sweep harness. Drives mma_policy across the
# raai (distance-scaled p/q regularizer) grid {0, 1e-6, 1e-5, 1e-4} and
# the asymptote_min_fraction (ASYMIN) grid {1e-4, 1e-3, 1e-2} on the
# MMA-compatible HS024/HS035/HS043/HS076 cells plus a small-gradient-
# terminal quartic. Records the terminal KKT-residual floor, terminal
# wandering, and oscillation-onset counts. Emits a JSON record. Non-ctest;
# one-shot.
add_executable(mma_regularizer_sweep mma_regularizer_sweep.cpp)
target_link_libraries(mma_regularizer_sweep PRIVATE argmin::argmin)

target_compile_options(mma_regularizer_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(mma_regularizer_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# GCMMA raa-augmented penalty-schedule sweep. Compares the Svanberg 2002
# eqs. 3.6-3.9 schedule (gradient-scaled raa floor recomputed per outer +
# 1.1*(raa+delta) growth) against the pre-fix fixed-floor + raa+2*delta
# schedule on HS024/HS035/HS076 at objective scales {1, 1e3, 1e6}; records
# outer iterations to a relative-objective target and the relative-error
# floor so the shipped schedule is scale-robust and does not regress
# convergence. Non-ctest, one-shot.
add_executable(gcmma_raa_schedule_sweep gcmma_raa_schedule_sweep.cpp)
target_link_libraries(gcmma_raa_schedule_sweep PRIVATE argmin::argmin)

target_compile_options(gcmma_raa_schedule_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(gcmma_raa_schedule_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Bounded-dual elastics scale sweep for the MMA/GCMMA/CCSA policies.
# Drives dual_bound_scale over {0.1, 1, 10, 1000, 10000, +inf} on
# inequality-infeasible-start cells (HS024 from an infeasible interior
# point plus two synthetic half-plane / half-space cells); records the
# feasibility-gated metric (outer iterations to feasibility, final
# objective error, max multiplier magnitude, max saturation ratio) so the
# shipped scale reaches feasibility with a bounded, well-conditioned dual
# and never clips the true multiplier. Non-ctest, one-shot.
add_executable(mma_elastic_dual_sweep mma_elastic_dual_sweep.cpp)
target_link_libraries(mma_elastic_dual_sweep PRIVATE argmin::argmin)

target_compile_options(mma_elastic_dual_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(mma_elastic_dual_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Adaptive-penalty growth-factor sweep harness for tr_sqp_policy. Drives
# the policy across penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} on the
# existing HS-suite cells (HS026/HS028/HS043/HS071/HS076) under both
# accurate and fast modes; emits a JSON record per (penalty_factor,
# cell, mode) plus a selection block identifying the largest non-
# regressing default per the existing tr_sqp_test.cpp acceptance bars.
# Non-ctest, one-shot.
add_executable(penalty_factor_sweep penalty_factor_sweep.cpp)
target_link_libraries(penalty_factor_sweep PRIVATE argmin::argmin)

target_compile_options(penalty_factor_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(penalty_factor_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Joint (penalty_factor, soc_max_iterations) sweep harness for
# tr_sqp_policy. Drives the policy across the cross product of
# penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} and soc_max_iterations
# in {0, 1, 2, 4} on the HS-suite cells (reference and closure-target
# subsets, both sqp_mode variants); emits a JSON record per
# (pf, soc_iter, cell, mode) plus a selection block recording the joint
# disposition (closure-count improvement vs the no-op baseline).
# Non-ctest, one-shot.
add_executable(soc_budget_sweep soc_budget_sweep.cpp)
target_link_libraries(soc_budget_sweep PRIVATE argmin::argmin)

target_compile_options(soc_budget_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(soc_budget_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Optional comparison library hooks (NLopt LD_SLSQP and similar).

option(ARGMIN_BENCH_NLOPT "Build NLopt comparison benchmarks" ON)
if(ARGMIN_BENCH_NLOPT)
    find_package(NLopt CONFIG QUIET)
    if(NLopt_FOUND)
        target_sources(argmin_bench PRIVATE bench_nlopt.cpp)
        target_link_libraries(argmin_bench PRIVATE NLopt::nlopt)
        target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_NLOPT)

        target_sources(publish_bench PRIVATE bench_nlopt.cpp)
        target_link_libraries(publish_bench PRIVATE NLopt::nlopt)
        target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_NLOPT)
    else()
        message(STATUS "NLopt not found, skipping NLopt benchmarks")
    endif()
endif()

# Optional kthohr/optim comparator (FetchContent; no system package).
option(ARGMIN_BENCH_OPTIM "Build kthohr/optim comparison benchmarks" ON)
if(ARGMIN_BENCH_OPTIM)
    if(ARGMIN_CMAKE_FETCH_DEPS)
        include(FetchContent)
        FetchContent_Declare(
            kthohr_optim
            GIT_REPOSITORY https://github.com/kthohr/optim.git
            GIT_TAG        5453f48aca695e6fef123677f35cb4d22e356e73
        )
        FetchContent_MakeAvailable(kthohr_optim)
        if(NOT TARGET optim::optim)
            add_library(argmin_optim_header_only INTERFACE)
            target_include_directories(argmin_optim_header_only INTERFACE
                "${kthohr_optim_SOURCE_DIR}/include"
            )
            target_compile_definitions(argmin_optim_header_only INTERFACE
                OPTIM_ENABLE_EIGEN_WRAPPERS
            )
            add_library(optim::optim ALIAS argmin_optim_header_only)
        endif()
        # Upstream ships the algorithm bodies as .cpp files alongside the
        # public headers (configure-time `--header-only-version` would
        # inline them, but FetchContent grabs the standard layout). Pull
        # them in directly so the optim symbols link.
        set(ARGMIN_BENCH_OPTIM_SOURCES
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/bfgs.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/cg.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/de.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/de_prmm.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/gd.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/lbfgs.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/newton.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/nm.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/pso.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/unconstrained/pso_dv.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/constrained/sumt.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/line_search/more_thuente.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/zeros/broyden.cpp"
            "${kthohr_optim_SOURCE_DIR}/src/zeros/broyden_df.cpp"
        )
        target_sources(argmin_bench PRIVATE bench_optim.cpp ${ARGMIN_BENCH_OPTIM_SOURCES})
        target_link_libraries(argmin_bench PRIVATE optim::optim)
        target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_OPTIM)

        target_sources(publish_bench PRIVATE bench_optim.cpp ${ARGMIN_BENCH_OPTIM_SOURCES})
        target_link_libraries(publish_bench PRIVATE optim::optim)
        target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_OPTIM)
    else()
        message(STATUS "kthohr/optim requires ARGMIN_CMAKE_FETCH_DEPS=ON; skipping optim benchmarks")
    endif()
endif()

# Micro-benchmarks for single-solver profiling.
if(NLopt_FOUND)
    add_executable(micro_lbfgsb micro_lbfgsb.cpp)
    target_link_libraries(micro_lbfgsb PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_lbfgsb PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_lbfgsb PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_bobyqa micro_bobyqa.cpp)
    target_link_libraries(micro_bobyqa PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_bobyqa PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_bobyqa PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_cmaes micro_cmaes.cpp)
    target_link_libraries(micro_cmaes PRIVATE argmin::argmin)
    target_compile_options(micro_cmaes PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_cmaes PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_kraft_slsqp micro_kraft_slsqp.cpp)
    target_link_libraries(micro_kraft_slsqp PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_kraft_slsqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_kraft_slsqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_cobyla micro_cobyla.cpp)
    target_link_libraries(micro_cobyla PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_cobyla PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_cobyla PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_mma micro_mma.cpp)
    target_link_libraries(micro_mma PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_mma PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_mma PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_nw_sqp micro_nw_sqp.cpp)
    target_link_libraries(micro_nw_sqp PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_nw_sqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_nw_sqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_filter_slsqp micro_filter_slsqp.cpp)
    target_link_libraries(micro_filter_slsqp PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_filter_slsqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_filter_slsqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_filter_nw_sqp micro_filter_nw_sqp.cpp)
    target_link_libraries(micro_filter_nw_sqp PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_filter_nw_sqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_filter_nw_sqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    # filter-trust-region SQP per-iter wall-time micro bench. Pairs
    # argmin's filter_trsqp_policy (Byrd-Omojokun composite step inside
    # a Fletcher-Leyffer filter envelope) against NLopt's LD_SLSQP on
    # the closed-cell workload. Mode-parametric on sqp_mode (accurate +
    # fast); covers HS028 / HS026 on both modes and HS071 on fast only.
    add_executable(micro_filter_trsqp micro_filter_trsqp.cpp)
    target_link_libraries(micro_filter_trsqp PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_filter_trsqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_filter_trsqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    # Allocation-gate targets for every real-time-claimed policy plus their
    # reset() paths. Each pairs a micro_* driver's argmin_alloc_trace_probe()
    # with the shared alloc_trace_main.cpp (see argmin_add_alloc_gate above).
    # The known-allocating constrained solvers run as pre-fix witnesses: the
    # gate must observe the reviewer-measured per-step allocation traffic,
    # proving it is no longer blind. The already-clean L-BFGS-B family runs as
    # a zero-allocation gate. tr_sqp reuses the filter_trsqp driver with a
    # policy-select define so both trust-region policies are covered.
    argmin_add_alloc_gate(micro_kraft_slsqp_alloc_trace
        micro_kraft_slsqp.cpp sqp_alloc_gate_kraft)
    argmin_add_alloc_gate(micro_nw_sqp_alloc_trace
        micro_nw_sqp.cpp sqp_alloc_gate_nw)
    argmin_add_alloc_gate(micro_filter_slsqp_alloc_trace
        micro_filter_slsqp.cpp sqp_alloc_gate_filter_slsqp)
    argmin_add_alloc_gate(micro_filter_nw_sqp_alloc_trace
        micro_filter_nw_sqp.cpp sqp_alloc_gate_filter_nw)
    argmin_add_alloc_gate(micro_filter_trsqp_alloc_trace
        micro_filter_trsqp.cpp sqp_alloc_gate_filter_trsqp)
    argmin_add_alloc_gate(micro_tr_sqp_alloc_trace
        micro_filter_trsqp.cpp sqp_alloc_gate_tr_sqp ARGMIN_ALLOC_TRACE_TR_SQP)
    argmin_add_alloc_gate(micro_byrd_lbfgsb_alloc_trace
        micro_byrd_lbfgsb.cpp alloc_gate_byrd_lbfgsb)
    argmin_add_alloc_gate(micro_lbfgsb_alloc_trace
        micro_lbfgsb.cpp alloc_gate_lbfgsb)

    add_executable(micro_augmented_lagrangian micro_augmented_lagrangian.cpp)
    target_link_libraries(micro_augmented_lagrangian PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_augmented_lagrangian PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_augmented_lagrangian PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_isres micro_isres.cpp)
    target_link_libraries(micro_isres PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_isres PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_isres PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_projected_gn micro_projected_gn.cpp)
    target_link_libraries(micro_projected_gn PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_projected_gn PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_projected_gn PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_projected_gradient_gn micro_projected_gradient_gn.cpp)
    target_link_libraries(micro_projected_gradient_gn PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_projected_gradient_gn PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_projected_gradient_gn PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_byrd_lbfgsb micro_byrd_lbfgsb.cpp)
    target_link_libraries(micro_byrd_lbfgsb PRIVATE argmin::argmin NLopt::nlopt)
    target_compile_options(micro_byrd_lbfgsb PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(ARGMIN_LTO_SUPPORTED)
        set_target_properties(micro_byrd_lbfgsb PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

# Levenberg-Marquardt micro-benchmark (argmin-only: unconstrained
# least-squares, no direct NLopt equivalent).
add_executable(micro_lm micro_lm.cpp)
target_link_libraries(micro_lm PRIVATE argmin::argmin)
target_compile_options(micro_lm PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native -fno-math-errno -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(micro_lm PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Levenberg-Marquardt allocation gate. Independent of NLopt, so it lives
# outside the NLopt_FOUND block. Registered under the alloc-gate label.
argmin_add_alloc_gate(micro_lm_alloc_trace micro_lm.cpp alloc_gate_lm)

# Active-set QP micro-benchmark (argmin-only, no external dependencies).
add_executable(micro_active_set_qp micro_active_set_qp.cpp)
target_link_libraries(micro_active_set_qp PRIVATE argmin::argmin)
target_compile_options(micro_active_set_qp PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(micro_active_set_qp PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Filter envelope sweep harness (argmin-only). Sweeps the asymmetric
# (gamma_f, gamma_h) envelope on HS043 with HS024 + HS076 regression
# guards across both filter_slsqp and filter_nw_sqp; prints a
# selectable accurate-mode default per policy.
add_executable(filter_envelope_sweep filter_envelope_sweep.cpp)
target_link_libraries(filter_envelope_sweep PRIVATE argmin::argmin)
target_compile_options(filter_envelope_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(filter_envelope_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Filter envelope + switching-condition + initial-trust-radius +
# restoration sweep harness for filter_trsqp_policy. Drives the policy
# across the cross product of (gamma_f, gamma_h, reject_mode, kappa, s,
# initial_trust_radius, restoration_max_iter, sqp_mode, problem, seed)
# on the HS-suite cells consumed by the empirical-default-selection
# workflow; emits a JSON record per (config, cell, seed) plus an MD
# analysis. argv-driven; sweep_kind dispatches the grid. Non-ctest,
# one-shot.
add_executable(filter_trsqp_sweep filter_trsqp_sweep.cpp)
target_link_libraries(filter_trsqp_sweep PRIVATE argmin::argmin)
target_compile_options(filter_trsqp_sweep PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(filter_trsqp_sweep PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Diagnostic KKT-breakdown harness (throwaway; emits per-iter CSV across
# the four SQP premature-ftol regressions, the byrd_lbfgsb null-step
# termination gap, and the hs071 diagnostic byproduct case). Opt-in
# build via EXCLUDE_FROM_ALL so it does not enter the default target
# set; invoke explicitly with `cmake --build <dir> --target
# diagnostic_kkt_breakdown`.
add_executable(diagnostic_kkt_breakdown EXCLUDE_FROM_ALL
    diagnostic_kkt_breakdown.cpp)
target_link_libraries(diagnostic_kkt_breakdown PRIVATE argmin::argmin)
target_compile_options(diagnostic_kkt_breakdown PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(diagnostic_kkt_breakdown PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

option(ARGMIN_BENCH_CERES "Build Ceres comparison benchmarks" ON)
if(ARGMIN_BENCH_CERES)
    find_package(Ceres CONFIG QUIET)
    if(Ceres_FOUND)
        target_sources(argmin_bench PRIVATE bench_ceres.cpp)
        target_link_libraries(argmin_bench PRIVATE Ceres::ceres)
        target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_CERES)

        target_sources(publish_bench PRIVATE bench_ceres.cpp)
        target_link_libraries(publish_bench PRIVATE Ceres::ceres)
        target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_CERES)
    else()
        message(STATUS "Ceres not found, skipping Ceres benchmarks")
    endif()
endif()

option(ARGMIN_BENCH_DLIB "Build dlib comparison benchmarks" ON)
if(ARGMIN_BENCH_DLIB)
    find_package(dlib CONFIG QUIET)
    if(dlib_FOUND)
        target_sources(argmin_bench PRIVATE bench_dlib.cpp)
        target_link_libraries(argmin_bench PRIVATE dlib::dlib)
        target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_DLIB)

        target_sources(publish_bench PRIVATE bench_dlib.cpp)
        target_link_libraries(publish_bench PRIVATE dlib::dlib)
        target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_DLIB)
    else()
        message(STATUS "dlib not found, skipping dlib benchmarks")
    endif()
endif()

option(ARGMIN_BENCH_LIBCMAES "Build libcmaes comparison benchmarks" ON)
if(ARGMIN_BENCH_LIBCMAES)
    if(ARGMIN_CMAKE_FETCH_DEPS)
        include(FetchContent)
        # libcmaes v0.10 ships unconditional install(EXPORT ...) and bare
        # export(TARGETS cmaes ...) commands at top level which fail when
        # used as a FetchContent subdirectory because argmin's
        # eigen_headers (the alias backing Eigen3::Eigen under FETCH_DEPS)
        # is not in any export set. We do not need the export at all --
        # we only consume the cmaes target from a sibling target. Wrap
        # the MakeAvailable in a block() and shadow install / export
        # with no-op functions inside the block so the override does not
        # leak to the parent scope.
        #
        # The cmake_minimum_required(VERSION 3.1) inside libcmaes is
        # rejected by CMake 4+; CMAKE_POLICY_VERSION_MINIMUM is the
        # documented escape hatch.
        block()
            function(install)
            endfunction()
            function(export)
            endfunction()
            set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
            set(LIBCMAES_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
            set(LIBCMAES_BUILD_PYTHON      OFF CACHE BOOL "" FORCE)
            set(LIBCMAES_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
            set(LIBCMAES_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
            set(LIBCMAES_USE_OPENMP        OFF CACHE BOOL "" FORCE)
            set(LIBCMAES_ENABLE_SURROG     OFF CACHE BOOL "" FORCE)
            # Upstream libcmaes/src/CMakeLists.txt:81 generates
            # cmaes_export.h under ${CMAKE_BINARY_DIR}/include/libcmaes/
            # but the cmaes target's include path uses
            # ${PROJECT_BINARY_DIR}/include, which only coincide when
            # libcmaes is the top-level project. Under FetchContent the
            # generated header lands in the parent project's build dir
            # and is not on cmaes's include path. Patch the upstream
            # call to use PROJECT_BINARY_DIR so the header lands where
            # the include path expects it.
            # Portable in-place text substitution. The previous form used
            # `sed -i` which differs between GNU sed (Linux: `sed -i ...`)
            # and BSD sed (macOS / Apple toolchain: `sed -i '' ...` requires
            # an explicit backup-suffix arg). Driving the patch through a
            # cmake -P script sidesteps the platform split entirely; the
            # script reads, regex-replaces, and writes back atomically.
            set(_libcmaes_patch_script
                "${CMAKE_CURRENT_BINARY_DIR}/libcmaes_patch.cmake")
            file(WRITE "${_libcmaes_patch_script}" [=[
file(READ "${TARGET}" _content)
string(REPLACE
    "\${CMAKE_BINARY_DIR}/include/libcmaes/cmaes_export.h"
    "\${PROJECT_BINARY_DIR}/include/libcmaes/cmaes_export.h"
    _content "${_content}")
file(WRITE "${TARGET}" "${_content}")
]=])
            FetchContent_Declare(
                libcmaes
                GIT_REPOSITORY https://github.com/CMA-ES/libcmaes.git
                GIT_TAG        v0.10
                GIT_SHALLOW    TRUE
                EXCLUDE_FROM_ALL
                PATCH_COMMAND  ${CMAKE_COMMAND}
                    -DTARGET=<SOURCE_DIR>/src/CMakeLists.txt
                    -P ${_libcmaes_patch_script}
                UPDATE_DISCONNECTED TRUE
            )
            FetchContent_MakeAvailable(libcmaes)
        endblock()
        unset(CMAKE_POLICY_VERSION_MINIMUM CACHE)

        # Eigen ABI alignment: argmin's bench targets compile with
        # -march=native, which under AVX gives Eigen 32-byte-aligned
        # fixed-size types. libcmaes built with default flags keeps
        # 16-byte alignment. Mixing Eigen-allocated objects across
        # the two ABI boundaries triggers heap corruption (double-free
        # at frame teardown). Force the cmaes target to compile with
        # the same vectorization flags so allocation/deallocation
        # alignments match across the boundary.
        target_compile_options(cmaes PRIVATE
            $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
                -march=native
                -fno-math-errno
                -fno-trapping-math
            >
        )

        target_sources(argmin_bench PRIVATE bench_libcmaes.cpp)
        target_link_libraries(argmin_bench PRIVATE libcmaes::cmaes)
        target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_LIBCMAES)

        target_sources(publish_bench PRIVATE bench_libcmaes.cpp)
        target_link_libraries(publish_bench PRIVATE libcmaes::cmaes)
        target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_LIBCMAES)

        # micro_cmaes uses libcmaes CMAES_DEFAULT as its same-class
        # comparator; keep ABI alignment via the same -march=native
        # already wired on cmaes above.
        if(TARGET micro_cmaes)
            target_link_libraries(micro_cmaes PRIVATE libcmaes::cmaes)
            target_compile_definitions(micro_cmaes PRIVATE ARGMIN_HAS_LIBCMAES)
        endif()
    else()
        message(STATUS "libcmaes requires ARGMIN_CMAKE_FETCH_DEPS=ON; skipping libcmaes benchmarks")
    endif()
endif()

option(ARGMIN_BENCH_IPOPT "Build IPOPT comparison benchmarks" ON)
if(ARGMIN_BENCH_IPOPT)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(IPOPT IMPORTED_TARGET QUIET ipopt)
        if(IPOPT_FOUND)
            target_sources(argmin_bench PRIVATE bench_ipopt.cpp)
            target_link_libraries(argmin_bench PRIVATE PkgConfig::IPOPT)
            target_compile_definitions(argmin_bench PRIVATE ARGMIN_HAS_IPOPT)

            target_sources(publish_bench PRIVATE bench_ipopt.cpp)
            target_link_libraries(publish_bench PRIVATE PkgConfig::IPOPT)
            target_compile_definitions(publish_bench PRIVATE ARGMIN_HAS_IPOPT)
        else()
            message(STATUS "IPOPT not found via pkg-config, skipping IPOPT benchmarks")
        endif()
    else()
        message(STATUS "PkgConfig not available, skipping IPOPT benchmarks")
    endif()
endif()

# Standalone regression gate. Consumes publish_summary.csv and a checked-in
# per-cell baseline (benchmarks/baselines/v0.3.1-regression.csv); exits
# non-zero on any expected:pass cell breaching a numeric bound or any
# expected:shouldfail cell unexpectedly converging. C++23 std-only --
# no Eigen, no argmin, no Catch2, no NLopt; pure CSV bound comparison.
# Invoked post-bench from run_publish_bench.sh.
add_executable(regression_check regression_check.cpp)
target_compile_options(regression_check PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -fno-math-errno
        -fno-trapping-math
    >
)

# Fast-CI regression gate. Drives run_publish_bench.sh at SEED_COUNT=3
# under the build's binary paths and a temporary output root inside the
# build tree; the script propagates regression_check's exit code so any
# bound breach fails the test cell. Registered only when testing is
# enabled in the active build configuration (i.e., the build dir was
# configured with both ARGMIN_BUILD_TESTS=ON and ARGMIN_BUILD_BENCHMARKS=ON).
if(CMAKE_TESTING_ENABLED)
    add_test(
        NAME publish_bench_regression_gate
        COMMAND
            ${CMAKE_COMMAND} -E env
                "SEED_START=42"
                "SEED_COUNT=3"
                "OUT_ROOT=${CMAKE_BINARY_DIR}/regression_gate_runs"
                "BENCH_BINARY=$<TARGET_FILE:publish_bench>"
                "REGRESSION_CHECK_BINARY=$<TARGET_FILE:regression_check>"
                "REGRESSION_BASELINE=${CMAKE_SOURCE_DIR}/benchmarks/baselines/v0.3.1-regression.csv"
                # Per-step wall variance dominates at SEED_COUNT=3; the
                # outer-iter, accuracy, status, and cv gates stay active.
                # Full 11-seed sweep via run_publish_bench.sh enforces the
                # wall gate.
                "REGRESSION_CHECK_DISABLE_WALL_GATE=1"
                bash "${CMAKE_SOURCE_DIR}/benchmarks/scripts/run_publish_bench.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(publish_bench_regression_gate PROPERTIES
        TIMEOUT 600
        LABELS "regression;bench"
    )
endif()
