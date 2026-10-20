cmake_minimum_required(VERSION 3.28)
project(argmin VERSION 0.3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

# gcc before 15 reports Eigen's AVX paths at tiny fixed sizes as out of bounds, and cannot relate a
# runtime loop bound to fixed-size storage. Both are false positives; MSan checks real uninitialized reads.
# Under LTO these warnings fire at link time, so LTO targets pass the set to the linker as well.
set(ARGMIN_GCC_FALSE_POSITIVE_FLAGS
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,15>>:
        -Wno-array-bounds
        -Wno-stringop-overflow
        -Wno-stringop-overread
        -Wno-maybe-uninitialized
        -Wno-aggressive-loop-optimizations
    >
)

set(ARGMIN_WARNING_FLAGS
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /utf-8>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wold-style-cast
        -Wcast-align
        -Woverloaded-virtual
        -Wnon-virtual-dtor
        -Wdouble-promotion
        -Wimplicit-fallthrough
        -Wformat=2
    >
    ${ARGMIN_GCC_FALSE_POSITIVE_FLAGS}
)

option(ARGMIN_BUILD_TESTS "Build unit and compile tests" OFF)
option(ARGMIN_BUILD_EXAMPLES "Build examples" OFF)
option(ARGMIN_BUILD_FUZZ_TESTS "Build fuzz test targets" OFF)
option(ARGMIN_BUILD_PROPERTY_TESTS "Build property-based test targets (requires RapidCheck)" OFF)
option(ARGMIN_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ARGMIN_BUILD_PYTHON "Build the Python extension module and its test suite" OFF)
option(ARGMIN_CMAKE_FETCH_DEPS "Use CMake FetchContent to download and install dependencies" OFF)

# On the fetch path Eigen3::Eigen aliases a local, uninstalled interface target that belongs to no
# export set, so argmin's own export cannot generate there; the default is therefore not merely top-level.
if(PROJECT_IS_TOP_LEVEL AND NOT ARGMIN_CMAKE_FETCH_DEPS)
    set(_argmin_install_default ON)
else()
    set(_argmin_install_default OFF)
endif()
option(ARGMIN_INSTALL "Install argmin into the install prefix" ${_argmin_install_default})

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# MSVC caps the number of sections per object file; the heavily-templated
# instantiation TUs (unit tests, no-exceptions probe) exceed it without this.
if(MSVC)
    add_compile_options(/bigobj)
endif()

add_subdirectory(lib)

if(ARGMIN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ARGMIN_BUILD_TESTS OR ARGMIN_BUILD_FUZZ_TESTS OR ARGMIN_BUILD_PROPERTY_TESTS OR ARGMIN_BUILD_BENCHMARKS OR ARGMIN_BUILD_PYTHON)
    enable_testing()
endif()

if(ARGMIN_BUILD_TESTS OR ARGMIN_BUILD_FUZZ_TESTS OR ARGMIN_BUILD_PROPERTY_TESTS)
    add_subdirectory(tests)
endif()

if(ARGMIN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ARGMIN_BUILD_PYTHON)
    add_subdirectory(python)
endif()
