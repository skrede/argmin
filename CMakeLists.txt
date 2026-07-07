cmake_minimum_required(VERSION 3.28)
project(argmin VERSION 0.2.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

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
)

option(ARGMIN_BUILD_TESTS "Build unit and compile tests" OFF)
option(ARGMIN_BUILD_EXAMPLES "Build examples" OFF)
option(ARGMIN_BUILD_FUZZ_TESTS "Build fuzz test targets" OFF)
option(ARGMIN_BUILD_PROPERTY_TESTS "Build property-based test targets (requires RapidCheck)" OFF)
option(ARGMIN_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ARGMIN_CMAKE_FETCH_DEPS "Use CMake FetchContent to download and install dependencies" OFF)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(lib)

if(ARGMIN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ARGMIN_BUILD_TESTS OR ARGMIN_BUILD_FUZZ_TESTS OR ARGMIN_BUILD_PROPERTY_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ARGMIN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
