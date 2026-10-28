# Inverted witness scan for a known defect, run with -P.
#
#   SCAN - a data file that sets SCAN_NAME (the property the fix restores), SCAN_FILES (paths relative to
#          ROOT; a path with a wildcard is matched recursively, a plain path names only that file) and
#          SCAN_REGEX (the witness). The regex lives in a data file so no shell quotes it differently on Windows.
#   ROOT - the source tree the globs are relative to.
#   SCAN_ABSENT - optional; when TRUE the defect is a missing construct, so the scan passes while the
#          regex matches nothing and stops with "unexpected pass" once it matches.
#
# Passes while the witness is present; stops with "unexpected pass" once it is gone, so the fixing change
# has to retire the scan. A file set that matches nothing is an error, not a fixed defect.

include(${SCAN})

foreach(required IN ITEMS SCAN_NAME SCAN_FILES SCAN_REGEX)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "known-defect scan: ${SCAN} does not set ${required}")
    endif()
endforeach()

set(files "")
foreach(pattern IN LISTS SCAN_FILES)
    if(pattern MATCHES "[*?]")
        file(GLOB_RECURSE matched "${ROOT}/${pattern}")
        list(APPEND files ${matched})
    elseif(EXISTS "${ROOT}/${pattern}")
        list(APPEND files "${ROOT}/${pattern}")
    endif()
endforeach()

if(NOT files)
    message(FATAL_ERROR "known-defect scan: no file under ${ROOT} matches '${SCAN_FILES}', "
                        "so the scan for '${SCAN_NAME}' reads nothing")
endif()

# A match may contain ';', which would split a MATCHALL list, so each match is replaced by one marker byte and the
# markers are counted instead.
string(ASCII 1 marker)
set(count 0)
foreach(path IN LISTS files)
    file(READ "${path}" text)
    string(REGEX REPLACE "${SCAN_REGEX}" "${marker}" marked "${text}")
    string(REGEX REPLACE "[^${marker}]" "" marked "${marked}")
    string(LENGTH "${marked}" n)
    math(EXPR count "${count} + ${n}")
endforeach()

if(SCAN_ABSENT)
    if(count GREATER 0)
        message(FATAL_ERROR "unexpected pass: '${SCAN_NAME}' now holds (${count} matches); "
                            "retire this known-defect scan")
    endif()
    message(STATUS "known defect still present: ${SCAN_NAME} (no match)")
    return()
endif()

if(count EQUAL 0)
    message(FATAL_ERROR "unexpected pass: the witness for '${SCAN_NAME}' is gone; "
                        "retire this known-defect scan")
endif()

message(STATUS "known defect still present: ${SCAN_NAME} (${count} matches)")
