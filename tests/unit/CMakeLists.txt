include(Catch)

add_executable(nablapp_unit_tests
    detail_test.cpp
    lm_test.cpp
    sqp_test.cpp
    mma_test.cpp
    mma_reciprocal_test.cpp
    isres_test.cpp
    cobyla_test.cpp
    bobyqa_test.cpp
    lbfgsb_test.cpp
    cmaes_test.cpp
    tolerance_test.cpp
    scheduling_test.cpp
    convergence_test.cpp
    line_search_test.cpp
    multistart_test.cpp
    projected_gn_test.cpp
    solver_group_test.cpp
    kraft_slsqp_test.cpp
    kraft_lsq_qp_test.cpp
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
)

target_link_libraries(nablapp_unit_tests
    PRIVATE
        nablapp::nablapp
        Catch2::Catch2WithMain
)

target_compile_options(nablapp_unit_tests PRIVATE ${NABLAPP_TEST_RELEASE_FLAGS})
if(NABLAPP_LTO_SUPPORTED)
    set_target_properties(nablapp_unit_tests PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

catch_discover_tests(nablapp_unit_tests)
