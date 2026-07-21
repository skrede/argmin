# Standalone regression gate. Consumes a publish-summary CSV and a checked-in
# per-cell baseline; exits non-zero on any pass cell breaching a numeric bound
# or any expected-fail cell unexpectedly converging. C++20 std-only -- no Eigen,
# no argmin, no Catch2, no NLopt; pure CSV bound comparison. The publication run
# invokes it post-bench from the benchmark wrapper script. It lives here rather
# than in the benchmark tree so the argmin-only gate logic and its self-tests
# survive a deletion of that tree; the live publication gate still invokes this
# same target when the benchmarks are built.
#
# Linux-only, via a CMake platform guard rather than a source ifdef: the parser
# relies on std::from_chars for doubles, which libstdc++ implements but Apple
# Clang's libc++ does not, and the publication/regression pipeline runs only on
# the Linux bench host and in Linux CI. Guarding here keeps the macOS and MSVC
# test builds green while leaving the gate fully live where it actually runs.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(regression_check regression_check.cpp)
    target_compile_options(regression_check PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -fno-math-errno
            -fno-trapping-math
        >
    )

    add_test(NAME regression_check_self_test
             COMMAND regression_check --self-test)
    set_tests_properties(regression_check_self_test PROPERTIES
        LABELS "regression;bench")
    add_test(NAME regression_check_prohibited_self_test
             COMMAND regression_check --self-test-prohibited-solvers)
    set_tests_properties(regression_check_prohibited_self_test PROPERTIES
        LABELS "regression;bench")
    add_test(NAME regression_check_instructions_self_test
             COMMAND regression_check --self-test-instructions)
    set_tests_properties(regression_check_instructions_self_test PROPERTIES
        LABELS "regression;bench")
    add_test(NAME regression_check_oracle_witness_self_test
             COMMAND regression_check --self-test-oracle-witness)
    set_tests_properties(regression_check_oracle_witness_self_test PROPERTIES
        LABELS "regression;bench")
    add_test(NAME regression_check_perf_ceiling_relax_self_test
             COMMAND regression_check --self-test-perf-ceiling-relax)
    set_tests_properties(regression_check_perf_ceiling_relax_self_test PROPERTIES
        LABELS "regression;bench")
endif()
