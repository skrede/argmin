include(Catch)

add_executable(argmin_trajectory_golden_test
    derivative_free_golden.cpp
    gradient_golden.cpp
    least_squares_golden.cpp
    moving_asymptote_golden.cpp
    qp_golden.cpp
    sqp_golden.cpp
    stochastic_golden.cpp
    trust_region_sqp_golden.cpp
    trajectory_golden_controls_test.cpp
    trajectory_golden_perturbation_test.cpp
    trajectory_golden_test.cpp
)

target_include_directories(argmin_trajectory_golden_test PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
target_link_libraries(argmin_trajectory_golden_test
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_definitions(argmin_trajectory_golden_test
    PRIVATE ARGMIN_TRAJECTORY_GOLDEN_DEFAULT="${CMAKE_SOURCE_DIR}/tests/oracles/all_family_trajectory_golden.csv")

# The golden is exact bits of one build identity, so this target takes none of the release test flags:
# no host instruction set (the CI runners differ in vector width), no contraction into fused multiply-add,
# and no link-time optimization.
target_compile_options(argmin_trajectory_golden_test
    PRIVATE
        ${ARGMIN_WARNING_FLAGS}
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror=switch>
)
set_target_properties(argmin_trajectory_golden_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

catch_discover_tests(argmin_trajectory_golden_test
    TEST_SPEC "[oracle-pin]"
    TEST_PREFIX "oracle-pin:"
    PROPERTIES LABELS "oracle-pin")
catch_discover_tests(argmin_trajectory_golden_test
    TEST_SPEC "~[oracle-pin]")
