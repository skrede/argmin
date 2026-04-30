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

# publish_bench -- publication-grade Dolan-More measurement driver.
add_executable(publish_bench
    publish_driver.cpp
)
target_link_libraries(publish_bench PRIVATE nablapp::nablapp benchmark::benchmark)

target_compile_options(publish_bench PRIVATE
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(publish_bench PROPERTIES
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

        target_sources(publish_bench PRIVATE bench_nlopt.cpp)
        target_link_libraries(publish_bench PRIVATE NLopt::nlopt)
        target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_NLOPT)
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

        target_sources(publish_bench PRIVATE bench_optim.cpp ${NABLAPP_BENCH_OPTIM_SOURCES})
        target_link_libraries(publish_bench PRIVATE optim::optim)
        target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_OPTIM)
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

option(NABLAPP_BENCH_CERES "Build Ceres comparison benchmarks" ON)
if(NABLAPP_BENCH_CERES)
    find_package(Ceres CONFIG QUIET)
    if(Ceres_FOUND)
        target_sources(nablapp_bench PRIVATE bench_ceres.cpp)
        target_link_libraries(nablapp_bench PRIVATE Ceres::ceres)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_CERES)

        target_sources(publish_bench PRIVATE bench_ceres.cpp)
        target_link_libraries(publish_bench PRIVATE Ceres::ceres)
        target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_CERES)
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

        target_sources(publish_bench PRIVATE bench_dlib.cpp)
        target_link_libraries(publish_bench PRIVATE dlib::dlib)
        target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_DLIB)
    else()
        message(STATUS "dlib not found, skipping dlib benchmarks")
    endif()
endif()

option(NABLAPP_BENCH_LIBCMAES "Build libcmaes comparison benchmarks" ON)
if(NABLAPP_BENCH_LIBCMAES)
    if(NABLAPP_CMAKE_FETCH_DEPS)
        include(FetchContent)
        # libcmaes v0.10 ships unconditional install(EXPORT ...) and bare
        # export(TARGETS cmaes ...) commands at top level which fail when
        # used as a FetchContent subdirectory because nablapp's
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
            FetchContent_Declare(
                libcmaes
                GIT_REPOSITORY https://github.com/CMA-ES/libcmaes.git
                GIT_TAG        v0.10
                GIT_SHALLOW    TRUE
                EXCLUDE_FROM_ALL
                PATCH_COMMAND  sed -i
                    "s|\\\${CMAKE_BINARY_DIR}/include/libcmaes/cmaes_export.h|\\\${PROJECT_BINARY_DIR}/include/libcmaes/cmaes_export.h|g"
                    "<SOURCE_DIR>/src/CMakeLists.txt"
                UPDATE_DISCONNECTED TRUE
            )
            FetchContent_MakeAvailable(libcmaes)
        endblock()
        unset(CMAKE_POLICY_VERSION_MINIMUM CACHE)

        # Eigen ABI alignment: nablapp's bench targets compile with
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

        target_sources(nablapp_bench PRIVATE bench_libcmaes.cpp)
        target_link_libraries(nablapp_bench PRIVATE libcmaes::cmaes)
        target_compile_definitions(nablapp_bench PRIVATE NABLAPP_HAS_LIBCMAES)

        target_sources(publish_bench PRIVATE bench_libcmaes.cpp)
        target_link_libraries(publish_bench PRIVATE libcmaes::cmaes)
        target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_LIBCMAES)
    else()
        message(STATUS "libcmaes requires NABLAPP_CMAKE_FETCH_DEPS=ON; skipping libcmaes benchmarks")
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

            target_sources(publish_bench PRIVATE bench_ipopt.cpp)
            target_link_libraries(publish_bench PRIVATE PkgConfig::IPOPT)
            target_compile_definitions(publish_bench PRIVATE NABLAPP_HAS_IPOPT)
        else()
            message(STATUS "IPOPT not found via pkg-config, skipping IPOPT benchmarks")
        endif()
    else()
        message(STATUS "PkgConfig not available, skipping IPOPT benchmarks")
    endif()
endif()
