include(Catch)

add_executable(argmin_property_tests
    least_squares_property_test.cpp
    planted_property_test.cpp
    qp_property_test.cpp
    reset_property_test.cpp
    sampling_property_test.cpp
    smooth_property_test.cpp
)

target_include_directories(argmin_property_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
target_link_libraries(argmin_property_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
        rapidcheck
)
target_compile_options(argmin_property_tests
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})

catch_discover_tests(argmin_property_tests
    PROPERTIES LABELS "property")
