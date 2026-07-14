# Toolchain-authoritative <chrono>-freedom scan.
#
# Runs the C++ compiler's dependency generation (-M) over a probe translation
# unit and fails if <chrono> appears anywhere in the transitive include graph.
# This inspects the compiler's real include set rather than an implementation-
# defined guard macro, so it holds across libstdc++, libc++, and MSVC STL.
#
# Invoked via add_test with -P. Required -D arguments:
#   CXX  - the C++ compiler command
#   SRC  - the probe source file
#   INCS - ';'-separated include directories (argmin + its usage requirements)
#   STD  - the C++ standard flag value (e.g. c++20)

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
    message(FATAL_ERROR "chrono-freedom scan: compiler dependency dump failed "
                        "(rc=${scan_rc}):\n${scan_err}")
endif()

# Tokenize the -M output (space / backslash-continuation / newline separated)
# and reject any dependency whose file name is exactly the <chrono> header.
string(REGEX MATCHALL "[^ \t\r\n\\\\]+" tokens "${deps}")
foreach(tok IN LISTS tokens)
    get_filename_component(base "${tok}" NAME)
    if(base STREQUAL "chrono")
        message(FATAL_ERROR
            "chrono-freedom scan FAILED: <chrono> reached the step-budget / "
            "stepper include graph via '${tok}'. These real-time surfaces must "
            "not depend on the wall clock; only the time-budget drivers may "
            "include <chrono>.")
    endif()
endforeach()

message(STATUS "chrono-freedom scan: OK -- no <chrono> in the step-budget / stepper include graph")
