include(Catch)

add_executable(argmin_defect_tests
    bound_ls_al_augmented_lagrangian_test.cpp
    bound_ls_al_detail_test.cpp
    bound_ls_al_least_squares_test.cpp
    bound_ls_al_test.cpp
    derivative_free_model_test.cpp
    derivative_free_oracle_test.cpp
    derivative_free_test.cpp
    harness_canary_test.cpp
    least_squares_oracle_test.cpp
    moving_asymptote_oracle_test.cpp
    moving_asymptote_solve_test.cpp
    moving_asymptote_test.cpp
    property_findings_test.cpp
    qp_active_set_oracle_test.cpp
    qp_oracle_test.cpp
    qp_test.cpp
    sqp_algorithm_step_test.cpp
    sqp_algorithm_test.cpp
    sqp_oracle_test.cpp
    sqp_safety_qp_test.cpp
    sqp_safety_restoration_test.cpp
    sqp_safety_test.cpp
    status_infra_group_test.cpp
    status_infra_sqp_test.cpp
    status_infra_test.cpp
    stochastic_oracle_test.cpp
    stochastic_sampling_test.cpp
    stochastic_test.cpp
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
argmin_known_defect_scan(sqp_soc_full_cascade_resolve sqp_soc_full_cascade_resolve.cmake)
argmin_known_defect_scan(sqp_lin_more_restart_refactorization sqp_lin_more_restart_refactorization.cmake)
argmin_known_defect_scan(sqp_dense_hessian_handoff sqp_dense_hessian_handoff.cmake)
argmin_known_defect_scan(sqp_absolute_zero_guard sqp_absolute_zero_guard.cmake)
argmin_known_defect_scan(sqp_bare_restoration_step_budget sqp_bare_restoration_step_budget.cmake)
argmin_known_defect_scan(moving_asymptote_equality_probe_overflow moving_asymptote_equality_probe_overflow.cmake)
argmin_known_defect_scan(moving_asymptote_bare_dual_budget moving_asymptote_bare_dual_budget.cmake)
argmin_known_defect_scan(moving_asymptote_subproblem_dual_budget moving_asymptote_subproblem_dual_budget.cmake)
argmin_known_defect_scan(cmaes_dense_sampling_and_ranking cmaes_dense_sampling_and_ranking.cmake)
argmin_known_defect_scan(seeded_standard_distributions seeded_standard_distributions.cmake)
argmin_known_defect_scan(bobyqa_hessian_packing_helper bobyqa_hessian_packing_helper.cmake)
argmin_known_defect_scan(bobyqa_safety_cap_converged bobyqa_safety_cap_converged.cmake)
argmin_known_defect_scan(bobyqa_bare_constants bobyqa_bare_constants.cmake)
argmin_known_defect_scan(lbfgs_middle_matrix_refactorization lbfgs_middle_matrix_refactorization.cmake)
argmin_known_defect_scan(augmented_lagrangian_bare_inner_constants augmented_lagrangian_bare_inner_constants.cmake)
argmin_known_defect_scan(least_squares_bare_diagonal_clamp least_squares_bare_diagonal_clamp.cmake)
argmin_known_defect_scan(fast_math_guard_diagnostic fast_math_guard_diagnostic.cmake)
argmin_known_defect_scan(fast_math_documentation fast_math_documentation.cmake)
argmin_known_defect_scan(convergence_bare_feasibility_floor convergence_bare_feasibility_floor.cmake)
argmin_known_defect_scan(convergence_bare_stall_windows convergence_bare_stall_windows.cmake)
argmin_known_defect_scan(convergence_bare_stall_buffer_cap convergence_bare_stall_buffer_cap.cmake)
argmin_known_defect_scan(convergence_vacuous_stationarity_citation convergence_vacuous_stationarity_citation.cmake)
argmin_known_defect_scan(line_search_bare_evaluation_budget line_search_bare_evaluation_budget.cmake)
argmin_known_defect_scan(finite_difference_jacobian_allocation finite_difference_jacobian_allocation.cmake)

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

    argmin_known_defect_compile_probe(isres_fixed_dimension_ten isres_fixed_dimension_ten.cpp
        "static assertion failed.*OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG")
    argmin_known_defect_compile_probe(cmaes_fixed_dimension_thirty_three cmaes_fixed_dimension_thirty_three.cpp
        "static assertion failed.*OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG")
endif()
