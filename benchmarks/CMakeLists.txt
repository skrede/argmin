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
