include(Catch)

add_executable(argmin_oracle_tests
    coverage_keys_test.cpp
    derivative_free_coverage_test.cpp
    gradient_coverage_test.cpp
    least_squares_coverage_test.cpp
    moving_asymptote_coverage_test.cpp
    qp_coverage_test.cpp
    sqp_coverage_test.cpp
    stochastic_coverage_test.cpp
    trust_region_sqp_coverage_test.cpp
    oracle_edge_test.cpp
    oracle_self_test.cpp
    qp_oracle_self_test.cpp
)

target_include_directories(argmin_oracle_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
target_link_libraries(argmin_oracle_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)

# The claim table switches over every solver status with no default, so an unhandled status is a
# compile error rather than a silently unclaimed one.
target_compile_options(argmin_oracle_tests
    PRIVATE
        ${ARGMIN_TEST_RELEASE_FLAGS}
        ${ARGMIN_WARNING_FLAGS}
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror=switch>
)

# A known defect is counted under its own label rather than as an oracle self-test.
catch_discover_tests(argmin_oracle_tests
    TEST_SPEC "~[known-defect]"
    PROPERTIES LABELS "oracle")
catch_discover_tests(argmin_oracle_tests
    TEST_SPEC "[known-defect]"
    PROPERTIES LABELS "known-defect")

function(argmin_solver_coverage_scan_test test_name)
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            -DROOT=${CMAKE_SOURCE_DIR}
            -DINVENTORY=${CMAKE_SOURCE_DIR}/tests/support/cases/solver_inventory.cmake
            -DKEYS=${CMAKE_SOURCE_DIR}/tests/support/cases/case_keys.h
            ${ARGN}
            -P ${CMAKE_SOURCE_DIR}/cmake/solver_coverage_scan.cmake)
    set_tests_properties(${test_name} PROPERTIES LABELS "oracle")
endfunction()

argmin_solver_coverage_scan_test(oracle_coverage_inventory)
argmin_solver_coverage_scan_test(oracle_coverage_inventory_negative_control
    -DEXTRA=lib/argmin/include/argmin/solver/planted_policy.h)
set_tests_properties(oracle_coverage_inventory_negative_control
    PROPERTIES PASS_REGULAR_EXPRESSION "unmapped solver header: .*planted_policy.h")
