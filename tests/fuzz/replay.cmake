# Every stored input executed once, without mutation. A replay is a regression check and never stands in for a
# budgeted fuzzing run, which is what reaches inputs that need a mutation.

file(GLOB inputs "${CORPUS}/*")
list(LENGTH inputs available)
if(available LESS MINIMUM)
    message(FATAL_ERROR "corpus ${CORPUS} holds ${available} inputs, fewer than the ${MINIMUM} required")
endif()

execute_process(
    COMMAND ${TARGET} -runs=0 -timeout=25 -rss_limit_mb=2048 ${CORPUS}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE report
    ERROR_VARIABLE report
)

if(NOT status EQUAL 0)
    message(FATAL_ERROR "${TARGET} exited ${status}\n${report}")
endif()

# A zero exit status alone does not distinguish a clean replay from a binary that never reached its entry point.
if(NOT report MATCHES "Done ([0-9]+) runs")
    message(FATAL_ERROR "${TARGET} reported no execution count\n${report}")
endif()
if(CMAKE_MATCH_1 LESS available)
    message(FATAL_ERROR "${TARGET} executed ${CMAKE_MATCH_1} inputs against a corpus of ${available}")
endif()
