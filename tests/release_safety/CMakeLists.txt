include(Catch)

# Every TU here traps eigen_assert as a throw, so an alloc-counter TU (its own eigen_assert) must never join this binary.
add_executable(argmin_release_safety_tests
    compliant_rejection_test.cpp
    qp_empty_problem_test.cpp
    reset_status_test.cpp
    reset_length_surfaces_test.cpp
    reset_length_gradient_test.cpp
    reset_length_sqp_test.cpp
    reset_length_moving_stochastic_test.cpp
    sqp_constraint_shape_test.cpp
    moving_asymptote_shape_test.cpp
    hot_reset_test.cpp
    isres_storage_test.cpp
    reset_rows_test.cpp
    population_range_test.cpp
)
target_include_directories(argmin_release_safety_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/support
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(argmin_release_safety_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_definitions(argmin_release_safety_tests
    PRIVATE _GLIBCXX_ASSERTIONS)
target_compile_options(argmin_release_safety_tests
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})
catch_discover_tests(argmin_release_safety_tests
    PROPERTIES LABELS "release-safety")

# The probes run the compiler's -fsyntax-only, a GCC/Clang flag, so MSVC skips them.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    function(argmin_release_safety_compile_probe test_name source expect_regex)
        add_test(NAME ${test_name}
            COMMAND ${CMAKE_COMMAND}
                -DCXX=${CMAKE_CXX_COMPILER}
                -DSTD=c++20
                -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/probes/${source}
                "-DEXPECT=${expect_regex}"
                "-DINCS=$<TARGET_PROPERTY:argmin::argmin,INTERFACE_INCLUDE_DIRECTORIES>;${ARGMIN_EIGEN_INCLUDE_DIRS}"
                -P ${CMAKE_SOURCE_DIR}/cmake/known_defect_compile_probe.cmake)
        set_tests_properties(${test_name} PROPERTIES LABELS "release-safety")
    endfunction()

    argmin_release_safety_compile_probe(release_safety_start_rows_mismatch_rejected start_rows_mismatch.cpp
        "static assertion failed.*x0 must be a column vector")
    argmin_release_safety_compile_probe(release_safety_start_rows_match_compiles start_rows_match.cpp
        "static assertion failed")
    set_tests_properties(release_safety_start_rows_match_compiles PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")
    argmin_release_safety_compile_probe(release_safety_void_reset_policy_rejected void_reset_policy.cpp
        "static assertion failed.*solver_policy")
    argmin_release_safety_compile_probe(release_safety_status_reset_policy_compiles status_reset_policy.cpp
        "static assertion failed")
    set_tests_properties(release_safety_status_reset_policy_compiles PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")
    argmin_release_safety_compile_probe(release_safety_void_reset_stepper_rejected void_reset_stepper.cpp
        "static assertion failed.*must report a solver status")
    argmin_release_safety_compile_probe(release_safety_status_reset_stepper_compiles status_reset_stepper.cpp
        "static assertion failed")
    set_tests_properties(release_safety_status_reset_stepper_compiles PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")
    argmin_release_safety_compile_probe(release_safety_reset_rows_mismatch_rejected reset_rows_mismatch.cpp
        "static assertion failed.*x0 must be a column vector")
    argmin_release_safety_compile_probe(release_safety_reset_rows_match_compiles reset_rows_match.cpp
        "static assertion failed")
    set_tests_properties(release_safety_reset_rows_match_compiles PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")
endif()

function(argmin_assert_inventory_test test_name root allow)
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            -DROOT=${root}
            -DALLOW=${allow}
            -P ${CMAKE_SOURCE_DIR}/cmake/assert_inventory_scan.cmake)
    set_tests_properties(${test_name} PROPERTIES LABELS "release-safety")
endfunction()

set(assert_fixtures ${CMAKE_CURRENT_SOURCE_DIR}/assert_scan_fixtures)
argmin_assert_inventory_test(release_safety_assert_inventory
    ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/assert_allow_list.txt)
argmin_assert_inventory_test(release_safety_assert_inventory_rejects_unlisted
    ${assert_fixtures}/unlisted ${assert_fixtures}/unlisted/allow.txt)
argmin_assert_inventory_test(release_safety_assert_inventory_rejects_stale
    ${assert_fixtures}/stale ${assert_fixtures}/stale/allow.txt)
set_tests_properties(release_safety_assert_inventory_rejects_unlisted PROPERTIES PASS_REGULAR_EXPRESSION "unlisted assert")
set_tests_properties(release_safety_assert_inventory_rejects_stale PROPERTIES PASS_REGULAR_EXPRESSION "allow-listed assert missing")
