set(ARGMIN_TEST_RELEASE_FLAGS
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        -march=native
        -fno-math-errno
        -fno-trapping-math
    >
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ARGMIN_LTO_SUPPORTED)

# The property suites share the unit tests' Catch2 and support headers, so they build only beside them.
if(ARGMIN_BUILD_PROPERTY_TESTS AND NOT ARGMIN_BUILD_TESTS)
    message(FATAL_ERROR "ARGMIN_BUILD_PROPERTY_TESTS=ON requires ARGMIN_BUILD_TESTS=ON")
endif()

if(ARGMIN_BUILD_TESTS)
    if(ARGMIN_CMAKE_FETCH_DEPS)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.13.0
            EXCLUDE_FROM_ALL
            SYSTEM
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        find_package(Catch2 CONFIG REQUIRED)
    endif()

    if(ARGMIN_BUILD_PROPERTY_TESTS AND ARGMIN_CMAKE_FETCH_DEPS)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG b96a4e626ef4c7348dcd16c500353c2f997a9f3f
            EXCLUDE_FROM_ALL
            SYSTEM
        )
        FetchContent_MakeAvailable(rapidcheck)
    elseif(ARGMIN_BUILD_PROPERTY_TESTS)
        find_package(rapidcheck CONFIG)
        if(NOT TARGET rapidcheck)
            message(FATAL_ERROR "ARGMIN_BUILD_PROPERTY_TESTS needs an installed RapidCheck CMake package, or fetching with ARGMIN_CMAKE_FETCH_DEPS=ON")
        endif()
    endif()

    if(ARGMIN_BUILD_PROPERTY_TESTS)
        add_subdirectory(property)
    endif()

    add_subdirectory(unit)
    add_subdirectory(result_oracle)
    add_subdirectory(defects)
    add_subdirectory(golden)
    add_subdirectory(compile)
    add_subdirectory(regression)

    if(NOT ARGMIN_CMAKE_FETCH_DEPS)
        add_subdirectory(integration)
    endif()
endif()

if(ARGMIN_BUILD_FUZZ_TESTS)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Fuzz targets require Clang with libFuzzer support")
    endif()
    add_subdirectory(fuzz)
endif()
