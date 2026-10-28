# Inverted compile probe for a known defect, run with -P.
#
#   CXX, STD, INCS - the compiler, the standard and the ';'-separated include directories
#   SRC            - the probe translation unit
#   EXPECT         - a regex one diagnostic line must match while the defect is present
#
# Passes while SRC fails to compile with the expected diagnostic. A probe that compiles is an "unexpected
# pass" the fixing change must retire; a probe that fails differently is a "wrong reason". EXPECT is matched
# line by line, so the source line a compiler echoes under a diagnostic cannot satisfy it on its own.

set(inc_flags "")
foreach(dir IN LISTS INCS)
    if(dir)
        list(APPEND inc_flags "-I${dir}")
    endif()
endforeach()

execute_process(
    COMMAND ${CXX} -std=${STD} -fsyntax-only ${inc_flags} ${SRC}
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE rc)

if(rc EQUAL 0)
    message(FATAL_ERROR "unexpected pass: ${SRC} now compiles; retire this known-defect probe")
endif()

set(rest "${out}${err}")
set(first_error "")
while(NOT rest STREQUAL "")
    string(FIND "${rest}" "\n" eol)
    if(eol EQUAL -1)
        set(line "${rest}")
        set(rest "")
    else()
        string(SUBSTRING "${rest}" 0 ${eol} line)
        math(EXPR eol "${eol} + 1")
        string(SUBSTRING "${rest}" ${eol} -1 rest)
    endif()
    if(line MATCHES "${EXPECT}")
        message(STATUS "known defect still present: ${SRC} fails with '${line}'")
        return()
    endif()
    if(first_error STREQUAL "" AND line MATCHES "error")
        set(first_error "${line}")
    endif()
endwhile()

if(first_error STREQUAL "")
    set(first_error "exit ${rc} with no error line")
endif()
message(FATAL_ERROR "wrong reason: ${first_error}")
