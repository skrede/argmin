include(Catch)

add_executable(argmin_defect_tests
    derivative_free_oracle_test.cpp
    harness_canary_test.cpp
    least_squares_oracle_test.cpp
    moving_asymptote_oracle_test.cpp
    property_findings_test.cpp
    qp_active_set_oracle_test.cpp
    qp_oracle_test.cpp
    qp_test.cpp
    sqp_oracle_test.cpp
    sqp_safety_qp_test.cpp
    sqp_safety_restoration_test.cpp
    sqp_safety_test.cpp
    status_infra_test.cpp
    stochastic_oracle_test.cpp
)

target_include_directories(argmin_defect_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
target_link_libraries(argmin_defect_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_options(argmin_defect_tests
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})

# CI asserts the exact size of the known-defect label, so the canaries register unlabeled.
catch_discover_tests(argmin_defect_tests
    TEST_SPEC "[known-defect]"
    PROPERTIES LABELS "known-defect")
catch_discover_tests(argmin_defect_tests
    TEST_SPEC "~[known-defect]")

function(argmin_defect_scan_test test_name data_file)
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            -DSCAN=${CMAKE_CURRENT_SOURCE_DIR}/scans/${data_file}
            -DROOT=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_SOURCE_DIR}/cmake/known_defect_scan.cmake)
endfunction()

function(argmin_known_defect_scan name data_file)
    argmin_defect_scan_test(known_defect_${name} ${data_file})
    set_tests_properties(known_defect_${name} PROPERTIES LABELS "known-defect")
endfunction()

argmin_defect_scan_test(known_defect_scan_canary_present canary_present.cmake)
argmin_defect_scan_test(known_defect_scan_canary_absent canary_absent.cmake)
set_tests_properties(known_defect_scan_canary_absent PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")

argmin_known_defect_scan(conventions_discarded_value_attribute conventions_discarded_value_attribute.cmake)
argmin_known_defect_scan(qp_active_set_empty_problem qp_active_set_empty_problem.cmake)
argmin_known_defect_scan(qp_admm_empty_problem qp_admm_empty_problem.cmake)
argmin_known_defect_scan(qp_active_set_multiplier_pivoted_qr qp_active_set_multiplier_pivoted_qr.cmake)
argmin_known_defect_scan(qp_lsi_lsei_dense_factorizations qp_lsi_lsei_dense_factorizations.cmake)
argmin_known_defect_scan(qp_admm_rho_update_gram_rebuild qp_admm_rho_update_gram_rebuild.cmake)
argmin_known_defect_scan(sqp_kkt_active_set_capacity sqp_kkt_active_set_capacity.cmake)
argmin_known_defect_scan(constraint_count_cap_unchecked constraint_count_cap_unchecked.cmake)
argmin_known_defect_scan(sqp_underived_tolerance_defaults sqp_underived_tolerance_defaults.cmake)

# The probes run the compiler's -fsyntax-only, a GCC/Clang flag, so MSVC skips them.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    function(argmin_defect_compile_probe_test test_name source expect_regex)
        add_test(NAME ${test_name}
            COMMAND ${CMAKE_COMMAND}
                -DCXX=${CMAKE_CXX_COMPILER}
                -DSTD=c++20
                -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/probes/${source}
                "-DEXPECT=${expect_regex}"
                "-DINCS=$<TARGET_PROPERTY:argmin::argmin,INTERFACE_INCLUDE_DIRECTORIES>;${ARGMIN_EIGEN_INCLUDE_DIRS}"
                -P ${CMAKE_SOURCE_DIR}/cmake/known_defect_compile_probe.cmake)
    endfunction()

    function(argmin_known_defect_compile_probe name source expect_regex)
        argmin_defect_compile_probe_test(known_defect_${name} ${source} "${expect_regex}")
        set_tests_properties(known_defect_${name} PROPERTIES LABELS "known-defect")
    endfunction()

    argmin_defect_compile_probe_test(known_defect_compile_canary_fails compile_canary_fails.cpp
        "static assertion failed.*known-defect compile canary")
    argmin_defect_compile_probe_test(known_defect_compile_canary_compiles compile_canary_compiles.cpp
        "static assertion failed")
    set_tests_properties(known_defect_compile_canary_compiles PROPERTIES PASS_REGULAR_EXPRESSION "unexpected pass")
endif()
