include(Catch)

add_executable(argmin_unit_tests
    detail_test.cpp
    lm_test.cpp
    sqp_test.cpp
    ccsa_quadratic_test.cpp
    mma_reciprocal_test.cpp
    gcmma_move_limit_shrink_test.cpp
    gcmma_rho_wval_test.cpp
    gcmma_raa_augmented_test.cpp
    isres_test.cpp
    isres_alternatives_test.cpp
    cobyla_test.cpp
    bobyqa_test.cpp
    lbfgsb_test.cpp
    cmaes_test.cpp
    cmaes_sampling_test.cpp
    restarting_policy_test.cpp
    tolerance_test.cpp
    scheduling_test.cpp
    convergence_test.cpp
    line_search_test.cpp
    multistart_test.cpp
    projected_gn_test.cpp
    solver_group_test.cpp
    kraft_slsqp_test.cpp
    kraft_lsq_qp_test.cpp
    kraft_lsq_qp_recovery_test.cpp
    basic_solver_test.cpp
    active_set_qp_test.cpp
    test_functions_test.cpp
    filter_nw_sqp_test.cpp
    hock_schittkowski_test.cpp
    augmented_lagrangian_test.cpp
    convergence_policy_test.cpp
    filter_sqp_test.cpp
    zero_copy_test.cpp
    finite_difference_test.cpp
    kkt_residual_test.cpp
    dense_ldl_bfgs_test.cpp
    merit_function_test.cpp
)

target_link_libraries(argmin_unit_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)

target_compile_options(argmin_unit_tests PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS})
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(argmin_unit_tests PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

catch_discover_tests(argmin_unit_tests)
