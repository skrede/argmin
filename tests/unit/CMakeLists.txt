include(Catch)

add_executable(nablapp_unit_tests
    test_functions_test.cpp
    basic_solver_test.cpp
    scheduling_test.cpp
    solver_group_test.cpp
    finite_difference_test.cpp
    line_search_test.cpp
    hessian_test.cpp
    halton_test.cpp
)

target_link_libraries(nablapp_unit_tests
    PRIVATE
        nablapp::nablapp
        Catch2::Catch2WithMain
)

catch_discover_tests(nablapp_unit_tests)
