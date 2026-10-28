# Disk-anchored coverage scan of the solver headers, run with -P.
#
#   ROOT      - the source tree the inventory paths are relative to.
#   INVENTORY - a data file that sets SOLVER_INVENTORY, one "<path> key|alias|exempt <args>" entry per header.
#   KEYS      - the header holding the case_keys array.
#   EXTRA     - optional; one more discovered path, which a negative control plants to prove the scan fails.
#
# Fails on a solver header without an entry, an entry naming a missing file, an alias whose target is not
# mapped to a key, a key absent from case_keys, and a case_keys key that no header reaches.

include(${INVENTORY})

file(GLOB discovered RELATIVE "${ROOT}"
    "${ROOT}/lib/argmin/include/argmin/solver/*_policy.h"
    "${ROOT}/lib/argmin/include/argmin/solver/alternative/*/*.h"
    "${ROOT}/lib/argmin/include/argmin/qp/*.h")
if(DEFINED EXTRA)
    list(APPEND discovered "${EXTRA}")
endif()

file(READ "${KEYS}" keys_text)
string(REGEX MATCH "case_keys\\{[^}]*\\}" keys_block "${keys_text}")
string(REGEX MATCHALL "\"[a-z0-9_]+\"" quoted_keys "${keys_block}")
if(NOT quoted_keys)
    message(FATAL_ERROR "no case_keys array in ${KEYS}")
endif()

set(mapped_paths "")
set(key_paths "")
foreach(entry IN LISTS SOLVER_INVENTORY)
    if(NOT entry MATCHES "^([^ ]+) (key|alias|exempt) (.+)$")
        message(FATAL_ERROR "malformed inventory entry: ${entry}")
    endif()
    set(path "${CMAKE_MATCH_1}")
    list(FIND mapped_paths "${path}" seen)
    if(NOT seen EQUAL -1)
        message(FATAL_ERROR "duplicate inventory entry: ${path}")
    endif()
    list(APPEND mapped_paths "${path}")
    set(kind_${path} "${CMAKE_MATCH_2}")
    set(args_${path} "${CMAKE_MATCH_3}")
    if(CMAKE_MATCH_2 STREQUAL "key")
        list(APPEND key_paths "${path}")
    endif()
endforeach()

foreach(path IN LISTS discovered)
    list(FIND mapped_paths "${path}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "unmapped solver header: ${path}")
    endif()
endforeach()

set(reached "")
foreach(path IN LISTS mapped_paths)
    if(NOT EXISTS "${ROOT}/${path}")
        message(FATAL_ERROR "inventory names a missing file: ${path}")
    endif()
    if(kind_${path} STREQUAL "alias")
        list(FIND key_paths "${args_${path}}" target)
        if(target EQUAL -1)
            message(FATAL_ERROR "alias target is not mapped to a key: ${path} -> ${args_${path}}")
        endif()
    elseif(kind_${path} STREQUAL "key")
        string(REPLACE " " ";" keys "${args_${path}}")
        foreach(key IN LISTS keys)
            list(FIND quoted_keys "\"${key}\"" listed)
            if(listed EQUAL -1)
                message(FATAL_ERROR "inventory key not in case_keys: ${key} (${path})")
            endif()
            list(APPEND reached "\"${key}\"")
        endforeach()
    endif()
endforeach()

foreach(key IN LISTS quoted_keys)
    list(FIND reached "${key}" hit)
    if(hit EQUAL -1)
        message(FATAL_ERROR "case_keys key that no solver header reaches: ${key}")
    endif()
endforeach()

list(LENGTH discovered header_count)
message(STATUS "solver coverage scan: ${header_count} headers mapped")
