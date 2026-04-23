cmake_minimum_required(VERSION 3.28)
project(nablapp_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Google Benchmark via FetchContent (per D-03/BNCH-01).
include(FetchContent)
if(NABLAPP_CMAKE_FETCH_DEPS)
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
check_ipo_supported(RESULT NABLAPP_LTO_SUPPORTED)

# Main benchmark executable (always builds).
add_executable(nablapp_bench
    benchmark_driver.cpp
)
target_link_libraries(nablapp_bench PRIVATE nablapp::nablapp benchmark::benchmark)

target_compile_options(nablapp_bench PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(nablapp_bench PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Optional comparison library hooks (per D-03).

option(NABLAPP_BENCH_NLOPT "Build NLopt comparison benchmarks" ON)
if(NABLAPP_BENCH_NLOPT)
    find_package(NLopt CONFIG QUIET)
    if(NLopt_FOUND)
        target_sources(nablapp_bench PRIVATE bench_nlopt.cpp)
        target_link_libraries(nablapp_bench PRIVATE NLopt::nlopt)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_NLOPT)
    else()
        message(STATUS "NLopt not found, skipping NLopt benchmarks")
    endif()
endif()

# Optional kthohr/optim comparator (FetchContent; no system package).
option(NABLAPP_BENCH_OPTIM "Build kthohr/optim comparison benchmarks" ON)
if(NABLAPP_BENCH_OPTIM)
    if(NABLAPP_CMAKE_FETCH_DEPS)
        include(FetchContent)
        FetchContent_Declare(
            kthohr_optim
            GIT_REPOSITORY https://github.com/kthohr/optim.git
            GIT_TAG        5453f48aca695e6fef123677f35cb4d22e356e73
        )
        FetchContent_MakeAvailable(kthohr_optim)
        if(NOT TARGET optim::optim)
            add_library(nablapp_optim_header_only INTERFACE)
            target_include_directories(nablapp_optim_header_only INTERFACE
                "${kthohr_optim_SOURCE_DIR}/include"
            )
            target_compile_definitions(nablapp_optim_header_only INTERFACE
                OPTIM_ENABLE_EIGEN_WRAPPERS
            )
            add_library(optim::optim ALIAS nablapp_optim_header_only)
        endif()
        # Upstream ships the algorithm bodies as .cpp files alongside the
        # public headers (configure-time `--header-only-version` would
        # inline them, but FetchContent grabs the standard layout). Pull
        # them in directly so the optim symbols link.
        set(NABLAPP_BENCH_OPTIM_SOURCES
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
        target_sources(nablapp_bench PRIVATE bench_optim.cpp ${NABLAPP_BENCH_OPTIM_SOURCES})
        target_link_libraries(nablapp_bench PRIVATE optim::optim)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_OPTIM)
    else()
        message(STATUS "kthohr/optim requires NABLAPP_CMAKE_FETCH_DEPS=ON; skipping optim benchmarks")
    endif()
endif()

# Micro-benchmarks for single-solver profiling.
if(NLopt_FOUND)
    add_executable(micro_lbfgsb micro_lbfgsb.cpp)
    target_link_libraries(micro_lbfgsb PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_lbfgsb PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_lbfgsb PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_bobyqa micro_bobyqa.cpp)
    target_link_libraries(micro_bobyqa PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_bobyqa PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_bobyqa PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_cmaes micro_cmaes.cpp)
    target_link_libraries(micro_cmaes PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_cmaes PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_cmaes PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_kraft_slsqp micro_kraft_slsqp.cpp)
    target_link_libraries(micro_kraft_slsqp PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_kraft_slsqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native
            -fno-math-errno
            -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_kraft_slsqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_cobyla micro_cobyla.cpp)
    target_link_libraries(micro_cobyla PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_cobyla PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_cobyla PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_mma micro_mma.cpp)
    target_link_libraries(micro_mma PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_mma PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_mma PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_gcmma micro_gcmma.cpp)
    target_link_libraries(micro_gcmma PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_gcmma PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_gcmma PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_nw_sqp micro_nw_sqp.cpp)
    target_link_libraries(micro_nw_sqp PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_nw_sqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_nw_sqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_filter_slsqp micro_filter_slsqp.cpp)
    target_link_libraries(micro_filter_slsqp PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_filter_slsqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_filter_slsqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_filter_nw_sqp micro_filter_nw_sqp.cpp)
    target_link_libraries(micro_filter_nw_sqp PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_filter_nw_sqp PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_filter_nw_sqp PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_augmented_lagrangian micro_augmented_lagrangian.cpp)
    target_link_libraries(micro_augmented_lagrangian PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_augmented_lagrangian PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_augmented_lagrangian PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_isres micro_isres.cpp)
    target_link_libraries(micro_isres PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_isres PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_isres PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_projected_gn micro_projected_gn.cpp)
    target_link_libraries(micro_projected_gn PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_projected_gn PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_projected_gn PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_projected_gradient_gn micro_projected_gradient_gn.cpp)
    target_link_libraries(micro_projected_gradient_gn PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_projected_gradient_gn PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_projected_gradient_gn PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(micro_byrd_lbfgsb micro_byrd_lbfgsb.cpp)
    target_link_libraries(micro_byrd_lbfgsb PRIVATE nablapp::nablapp NLopt::nlopt)
    target_compile_options(micro_byrd_lbfgsb PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -march=native -fno-math-errno -fno-trapping-math
        >
    )
    if(NABLAPP_LTO_SUPPORTED)
        set_target_properties(micro_byrd_lbfgsb PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

# Levenberg-Marquardt micro-benchmark (nablapp-only: unconstrained
# least-squares, no direct NLopt equivalent).
add_executable(micro_lm micro_lm.cpp)
target_link_libraries(micro_lm PRIVATE nablapp::nablapp)
target_compile_options(micro_lm PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native -fno-math-errno -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(micro_lm PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Active-set QP micro-benchmark (nablapp-only, no external dependencies).
add_executable(micro_active_set_qp micro_active_set_qp.cpp)
target_link_libraries(micro_active_set_qp PRIVATE nablapp::nablapp)
target_compile_options(micro_active_set_qp PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(micro_active_set_qp PROPERTIES
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
target_link_libraries(diagnostic_kkt_breakdown PRIVATE nablapp::nablapp)
target_compile_options(diagnostic_kkt_breakdown PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(diagnostic_kkt_breakdown PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Diagnostic MMA/GCMMA-breakdown harness (throwaway; emits per-iter CSV
# for MMA on HS043, GCMMA on HS035, GCMMA on HS043, plus a one-time
# sign-convention probe on HS076 MMA). EXCLUDE_FROM_ALL so it does not
# enter the default target set; invoke with
# `cmake --build <dir> --target diagnostic_mma_breakdown`.
add_executable(diagnostic_mma_breakdown EXCLUDE_FROM_ALL
    diagnostic_mma_breakdown.cpp)
target_link_libraries(diagnostic_mma_breakdown PRIVATE nablapp::nablapp)
target_compile_options(diagnostic_mma_breakdown PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(diagnostic_mma_breakdown PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Diagnostic MMA/GCMMA asymptote-oscillation harness (throwaway; emits
# per-outer-iteration CSV for mma_policy on HS043/HS076, gcmma_policy
# on HS076, HS035 control, an mma_policy HS043 move_limit A/B sweep at
# 0.5/0.7/0.9, a GCMMA HS076 K_saturation sweep, and row-matched NLopt
# LD_MMA reference trajectories on three inequality-constrained HS
# cells. EXCLUDE_FROM_ALL so it does not enter the default target set;
# invoke with
# `cmake --build <dir> --target diagnostic_mma_asymptote_trace`.
add_executable(diagnostic_mma_asymptote_trace EXCLUDE_FROM_ALL
    diagnostic_mma_asymptote_trace.cpp)
target_link_libraries(diagnostic_mma_asymptote_trace PRIVATE nablapp::nablapp)
if(NLopt_FOUND)
    target_link_libraries(diagnostic_mma_asymptote_trace PRIVATE NLopt::nlopt)
    target_compile_definitions(diagnostic_mma_asymptote_trace PRIVATE NABLAPP_HAS_NLOPT)
endif()
target_compile_options(diagnostic_mma_asymptote_trace PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(diagnostic_mma_asymptote_trace PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

option(NABLAPP_BENCH_CERES "Build Ceres comparison benchmarks" ON)
if(NABLAPP_BENCH_CERES)
    find_package(Ceres CONFIG QUIET)
    if(Ceres_FOUND)
        target_sources(nablapp_bench PRIVATE bench_ceres.cpp)
        target_link_libraries(nablapp_bench PRIVATE Ceres::ceres)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_CERES)
    else()
        message(STATUS "Ceres not found, skipping Ceres benchmarks")
    endif()
endif()

option(NABLAPP_BENCH_DLIB "Build dlib comparison benchmarks" ON)
if(NABLAPP_BENCH_DLIB)
    find_package(dlib CONFIG QUIET)
    if(dlib_FOUND)
        target_sources(nablapp_bench PRIVATE bench_dlib.cpp)
        target_link_libraries(nablapp_bench PRIVATE dlib::dlib)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_DLIB)
    else()
        message(STATUS "dlib not found, skipping dlib benchmarks")
    endif()
endif()

option(NABLAPP_BENCH_IPOPT "Build IPOPT comparison benchmarks" ON)
if(NABLAPP_BENCH_IPOPT)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(IPOPT IMPORTED_TARGET QUIET ipopt)
        if(IPOPT_FOUND)
            target_sources(nablapp_bench PRIVATE bench_ipopt.cpp)
            target_link_libraries(nablapp_bench PRIVATE PkgConfig::IPOPT)
            target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_IPOPT)
        else()
            message(STATUS "IPOPT not found via pkg-config, skipping IPOPT benchmarks")
        endif()
    else()
        message(STATUS "PkgConfig not available, skipping IPOPT benchmarks")
    endif()
endif()
