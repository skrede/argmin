# Inverted witness scan for a known defect, run with -P.
#
#   SCAN - a data file that sets SCAN_NAME (the property the fix restores), SCAN_FILES (globs relative to
#          ROOT, matched recursively) and SCAN_REGEX (the witness). The regex lives in a data file so no
#          shell quotes it differently on Windows.
#   ROOT - the source tree the globs are relative to.
#
# Passes while the witness is present; stops with "unexpected pass" once it is gone, so the fixing change
# has to retire the scan. A glob that matches nothing is an error, not a fixed defect.

include(${SCAN})

foreach(required IN ITEMS SCAN_NAME SCAN_FILES SCAN_REGEX)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "known-defect scan: ${SCAN} does not set ${required}")
    endif()
endforeach()

set(files "")
foreach(pattern IN LISTS SCAN_FILES)
    file(GLOB_RECURSE matched "${ROOT}/${pattern}")
    list(APPEND files ${matched})
endforeach()

if(NOT files)
    message(FATAL_ERROR "known-defect scan: no file under ${ROOT} matches '${SCAN_FILES}', "
                        "so the scan for '${SCAN_NAME}' reads nothing")
endif()

set(count 0)
foreach(path IN LISTS files)
    file(READ "${path}" text)
    string(REGEX MATCHALL "${SCAN_REGEX}" hits "${text}")
    list(LENGTH hits n)
    math(EXPR count "${count} + ${n}")
endforeach()

if(count EQUAL 0)
    message(FATAL_ERROR "unexpected pass: the witness for '${SCAN_NAME}' is gone; "
                        "retire this known-defect scan")
endif()

message(STATUS "known defect still present: ${SCAN_NAME} (${count} matches)")
