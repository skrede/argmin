include(Catch)

add_executable(nablapp_unit_tests
    test_functions_test.cpp
    basic_solver_test.cpp
)

target_link_libraries(nablapp_unit_tests
    PRIVATE
        nablapp::nablapp
        Catch2::Catch2WithMain
)

catch_discover_tests(nablapp_unit_tests)
