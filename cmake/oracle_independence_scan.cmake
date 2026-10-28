# Include-independence scan for the returned-result oracle.
#
# Runs the C++ compiler's dependency generation (-M) over a probe translation unit and fails if any
# argmin header outside the allow-list reaches the include graph. The oracle judges what a solver
# returns, so it may read the problem and result vocabulary but never the convergence code it judges.
#
# Invoked via add_test with -P. Required -D arguments:
#   CXX  - the C++ compiler command
#   SRC  - the probe source file
#   INCS - ';'-separated include directories
#   STD  - the C++ standard flag value (e.g. c++20)
# Optional:
#   EXPECT_VIOLATION - an argmin-relative header path; the scan then passes only when that header is
#                      the first violation in include order (its own includes follow it), which is how
#                      its negative control proves it can fail.

cmake_minimum_required(VERSION 3.28)

set(inc_flags "")
foreach(dir IN LISTS INCS)
    if(dir)
        list(APPEND inc_flags "-I${dir}")
    endif()
endforeach()

execute_process(
    COMMAND ${CXX} -std=${STD} ${inc_flags} -M ${SRC}
    OUTPUT_VARIABLE deps
    ERROR_VARIABLE scan_err
    RESULT_VARIABLE scan_rc)

if(NOT scan_rc EQUAL 0)
    message(FATAL_ERROR "oracle independence scan: compiler dependency dump failed (rc=${scan_rc}):\n${scan_err}")
endif()

# detail/expected.h and detail/expected_fallback.h are allowed because they define the error-return
# types, not convergence code.
set(allowed_exact "types.h;expected.h;detail/expected.h;detail/expected_fallback.h;qp/qp_types.h")
set(allowed_prefixes "result/;formulation/")

set(violations "")
set(reached_oracle FALSE)
string(REGEX MATCHALL "[^ \t\r\n\\\\]+" tokens "${deps}")
foreach(tok IN LISTS tokens)
    if(tok MATCHES "tests/support/oracle/")
        set(reached_oracle TRUE)
    endif()
    string(FIND "${tok}" "/argmin/include/argmin/" pos REVERSE)
    if(pos EQUAL -1)
        continue()
    endif()
    math(EXPR start "${pos} + 23")
    string(SUBSTRING "${tok}" ${start} -1 rel)
    set(allowed FALSE)
    if(rel IN_LIST allowed_exact)
        set(allowed TRUE)
    endif()
    foreach(prefix IN LISTS allowed_prefixes)
        if(rel MATCHES "^${prefix}")
            set(allowed TRUE)
        endif()
    endforeach()
    if(NOT allowed)
        list(APPEND violations "${rel}")
    endif()
endforeach()
list(REMOVE_DUPLICATES violations)

if(NOT reached_oracle)
    message(FATAL_ERROR "oracle independence scan FAILED: the probe does not include any header under tests/support/oracle/")
endif()

if(DEFINED EXPECT_VIOLATION)
    set(first "")
    if(violations)
        list(GET violations 0 first)
    endif()
    if(NOT first STREQUAL "${EXPECT_VIOLATION}")
        message(FATAL_ERROR "oracle independence negative control FAILED: expected '${EXPECT_VIOLATION}' as the first violation, got '${violations}'")
    endif()
    message(STATUS "oracle independence negative control: OK -- the scan reported ${EXPECT_VIOLATION}")
    return()
endif()

if(violations)
    message(FATAL_ERROR "oracle independence scan FAILED: the oracle include graph reaches argmin headers outside the allow-list: ${violations}")
endif()

message(STATUS "oracle independence scan: OK -- only result, formulation, error-return and QP vocabulary headers reached")
