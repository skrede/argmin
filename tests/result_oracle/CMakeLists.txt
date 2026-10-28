include(Catch)

add_executable(argmin_oracle_tests
    oracle_edge_test.cpp
    oracle_self_test.cpp
    qp_oracle_self_test.cpp
)

target_include_directories(argmin_oracle_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
target_link_libraries(argmin_oracle_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)

# The claim table switches over every solver status with no default, so an unhandled status is a
# compile error rather than a silently unclaimed one.
target_compile_options(argmin_oracle_tests
    PRIVATE
        ${ARGMIN_TEST_RELEASE_FLAGS}
        ${ARGMIN_WARNING_FLAGS}
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror=switch>
)

# A known defect is counted under its own label rather than as an oracle self-test.
catch_discover_tests(argmin_oracle_tests
    TEST_SPEC "~[known-defect]"
    PROPERTIES LABELS "oracle")
catch_discover_tests(argmin_oracle_tests
    TEST_SPEC "[known-defect]"
    PROPERTIES LABELS "known-defect")
