include(Catch)

add_executable(nablapp_unit_tests
    test_functions_test.cpp
    basic_solver_test.cpp
    detail_test.cpp
    scheduling_test.cpp
    solver_group_test.cpp
    finite_difference_test.cpp
    line_search_test.cpp
    hessian_test.cpp
    halton_test.cpp
    lbfgsb_test.cpp
    active_set_qp_test.cpp
    bobyqa_test.cpp
    sqp_test.cpp
    kraft_slsqp_test.cpp
    augmented_lagrangian_test.cpp
    mma_test.cpp
    hock_schittkowski_test.cpp
)

target_link_libraries(nablapp_unit_tests
    PRIVATE
        nablapp::nablapp
        Catch2::Catch2WithMain
)

catch_discover_tests(nablapp_unit_tests)
