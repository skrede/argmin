add_executable(concept_satisfaction_test concept_satisfaction_test.cpp)
target_link_libraries(concept_satisfaction_test PRIVATE argmin::argmin)
target_compile_options(concept_satisfaction_test PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME concept_satisfaction_test COMMAND concept_satisfaction_test)

add_executable(include_test include_test.cpp)
target_link_libraries(include_test PRIVATE argmin::argmin)
target_compile_options(include_test PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME include_test COMMAND include_test)

# Real-time umbrella header. Previously included by no translation unit, so its
# re-exports and definition-site concept boundary never compiled; this TU forces
# that and registers the header with the build.
add_executable(rt_header_test rt_header_test.cpp)
target_link_libraries(rt_header_test PRIVATE argmin::argmin)
target_compile_options(rt_header_test PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME rt_header_test COMMAND rt_header_test)

add_executable(sqp_common_compile_test sqp_common_compile_test.cpp)
target_link_libraries(sqp_common_compile_test PRIVATE argmin::argmin)
target_compile_options(sqp_common_compile_test PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME sqp_common_compile_test COMMAND sqp_common_compile_test)

add_executable(sqp_mode_racing_test sqp_mode_racing_test.cpp)
target_link_libraries(sqp_mode_racing_test PRIVATE argmin::argmin)
target_compile_options(sqp_mode_racing_test PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME sqp_mode_racing_test COMMAND sqp_mode_racing_test)

# Exceptions-off instantiation probe. Built and linked (never syntax-only)
# with -fno-exceptions -fno-rtti so that any surviving throw in an
# instantiated solver-policy path fails the compile. Kept in the default
# build so local ctest exercises the exceptions-off guarantee.
add_executable(argmin_no_exceptions_probe no_exceptions_instantiation_probe.cpp)
target_link_libraries(argmin_no_exceptions_probe PRIVATE argmin::argmin)
target_compile_options(argmin_no_exceptions_probe PRIVATE
    ${ARGMIN_TEST_RELEASE_FLAGS}
    -fno-exceptions
    -fno-rtti)
add_test(NAME argmin_no_exceptions_probe COMMAND argmin_no_exceptions_probe)

# Wall-clock-freedom probe: a TU that includes only the step-budget / stepper
# surfaces. Building it confirms those headers are self-contained; the scan test
# then proves no <chrono> reaches their transitive include graph.
add_executable(runner_chrono_freedom_probe runner_chrono_freedom_test.cpp)
target_link_libraries(runner_chrono_freedom_probe PRIVATE argmin::argmin)
target_compile_options(runner_chrono_freedom_probe PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
add_test(NAME runner_chrono_freedom_compiles COMMAND runner_chrono_freedom_probe)

# The <chrono>-freedom invariant is compiler-independent, so proving it on
# GCC/Clang suffices: the dependency-dump scan uses -M-style output that MSVC
# cl.exe does not provide, so MSVC skips the scan (never the compile probe
# above, which runs everywhere). Eigen's include dirs come from a cache variable
# rather than a genex on Eigen3::Eigen, whose find_package-imported target is not
# visible in this sibling scope under system-Eigen (FetchContent aliases it).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_test(NAME runner_chrono_freedom_scan
        COMMAND ${CMAKE_COMMAND}
            -DCXX=${CMAKE_CXX_COMPILER}
            -DSTD=c++20
            -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/runner_chrono_freedom_test.cpp
            "-DINCS=$<TARGET_PROPERTY:argmin::argmin,INTERFACE_INCLUDE_DIRECTORIES>;${ARGMIN_EIGEN_INCLUDE_DIRS}"
            -P ${CMAKE_SOURCE_DIR}/cmake/chrono_freedom_scan.cmake)
endif()
