include(Catch)

add_executable(argmin_defect_tests
    bound_ls_al_augmented_lagrangian_test.cpp
    bound_ls_al_detail_test.cpp
    bound_ls_al_least_squares_test.cpp
    bound_ls_al_test.cpp
    conventions_test.cpp
    derivative_free_model_test.cpp
    derivative_free_oracle_test.cpp
    derivative_free_test.cpp
    embedded_test.cpp
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
    state_docs_test.cpp
    status_infra_group_test.cpp
    status_infra_sqp_test.cpp
    status_infra_test.cpp
    status_underreporting_oracle_test.cpp
    stochastic_oracle_test.cpp
    stochastic_reset_test.cpp
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

# The allocation sensor redefines eigen_assert for every Eigen include, so its cases cannot share a binary with the
# other defect cases without two definitions of Eigen's inline functions.
add_executable(argmin_defect_alloc_tests
    embedded_alloc_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/alloc_sensor_counters.cpp
)
target_link_libraries(argmin_defect_alloc_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_definitions(argmin_defect_alloc_tests
    PRIVATE ARGMIN_BENCH_TRACE_ALLOC=1 EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(argmin_defect_alloc_tests
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})
catch_discover_tests(argmin_defect_alloc_tests
    TEST_SPEC "[known-defect]"
    PROPERTIES LABELS "known-defect")
catch_discover_tests(argmin_defect_alloc_tests
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
argmin_known_defect_scan(conventions_deduced_return_types conventions_deduced_return_types.cmake)
argmin_known_defect_scan(conventions_in_class_member_initializers conventions_in_class_member_initializers.cmake)
argmin_known_defect_scan(conventions_include_order conventions_include_order.cmake)
argmin_known_defect_scan(conventions_plain_int conventions_plain_int.cmake)
argmin_known_defect_scan(conventions_dual_problem_raw_pointers conventions_dual_problem_raw_pointers.cmake)
argmin_known_defect_scan(conventions_optional_literature_defaults conventions_optional_literature_defaults.cmake)
argmin_known_defect_scan(conventions_planning_language_comments conventions_planning_language_comments.cmake)
argmin_known_defect_scan(conventions_british_spelling conventions_british_spelling.cmake)
argmin_known_defect_scan(conventions_design_history_comments conventions_design_history_comments.cmake)
argmin_known_defect_scan(conventions_compatibility_comments conventions_compatibility_comments.cmake)
argmin_known_defect_scan(conventions_dead_trust_region_helpers conventions_dead_trust_region_helpers.cmake)
argmin_known_defect_scan(conventions_dead_quadratic_model_svd_path conventions_dead_quadratic_model_svd_path.cmake)
argmin_known_defect_scan(conventions_dead_active_set_augmented_storage conventions_dead_active_set_augmented_storage.cmake)
argmin_known_defect_scan(conventions_dead_isres_offspring_buffers conventions_dead_isres_offspring_buffers.cmake)
argmin_known_defect_scan(conventions_dead_sqp_buffers conventions_dead_sqp_buffers.cmake)
argmin_known_defect_scan(conventions_dead_asymptote_options conventions_dead_asymptote_options.cmake)
argmin_known_defect_scan(conventions_dead_ccsa_acceptance_state conventions_dead_ccsa_acceptance_state.cmake)
argmin_known_defect_scan(conventions_shared_ownership conventions_shared_ownership.cmake)
argmin_known_defect_scan(conventions_duplicate_time_budget_driver conventions_duplicate_time_budget_driver.cmake)
argmin_known_defect_scan(conventions_forced_verbose_makefile conventions_forced_verbose_makefile.cmake)
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
argmin_known_defect_scan(embedded_solver_static_storage_guidance embedded_solver_static_storage_guidance.cmake)
argmin_known_defect_scan(embedded_flash_footprint_documentation embedded_flash_footprint_documentation.cmake)
argmin_known_defect_scan(denormal_behavior_documentation denormal_behavior_documentation.cmake)
argmin_known_defect_scan(alloc_counter_atomic_increment alloc_counter_atomic_increment.cmake)
argmin_known_defect_scan(active_set_layout_stack_limit active_set_layout_stack_limit.cmake)
argmin_known_defect_scan(time_budget_unguarded_deadline time_budget_unguarded_deadline.cmake)
argmin_known_defect_scan(documented_compiler_floor_below_ci documented_compiler_floor_below_ci.cmake)
argmin_known_defect_scan(integration_consumer_cxx20 integration_consumer_cxx20.cmake)
argmin_known_defect_scan(windows_cxx20_configure windows_cxx20_configure.cmake)
argmin_known_defect_scan(umbrella_include_order umbrella_include_order.cmake)
argmin_known_defect_scan(benchmark_driver_include_order benchmark_driver_include_order.cmake)
argmin_known_defect_scan(choosing_a_solver_policy_count choosing_a_solver_policy_count.cmake)
argmin_known_defect_scan(rt_matrix_intrinsic_leaf_caps rt_matrix_intrinsic_leaf_caps.cmake)
argmin_known_defect_scan(rt_matrix_bounded_iterations_definition rt_matrix_bounded_iterations_definition.cmake)
argmin_known_defect_scan(readme_bounded_iterations_claim readme_bounded_iterations_claim.cmake)
argmin_known_defect_scan(rt_matrix_restoration_evidence_provenance rt_matrix_restoration_evidence_provenance.cmake)

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
    argmin_known_defect_compile_probe(mma_zero_fixed_constraint_rows mma_zero_fixed_constraint_rows.cpp
        "static assertion failed.*YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX")
    argmin_known_defect_compile_probe(argmin_under_min_max_macros argmin_under_min_max_macros.cpp
        "requires 2 arguments, but only 1 given|too few arguments provided to function-like macro invocation")
endif()
