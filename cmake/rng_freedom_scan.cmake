# Toolchain-authoritative random-number-freedom scan.
#
# Proves that the policies a probe translation unit includes carry no source of
# randomness -- the property behind every "identical inputs drive an identical
# trajectory" claim. A policy with no RNG in it cannot diverge between two runs
# on the same input, because there is nothing in it left to diverge.
#
# The scan has two parts, and it needs both. Neither is sufficient alone:
#
#   1. No <random> anywhere in the compiler's transitive include graph. This is
#      the same -M dependency dump the chrono scan uses, and it holds across
#      libstdc++, libc++ and MSVC STL because it reads the real include set
#      rather than a guard macro. It catches every std facility: the engines,
#      the distributions, random_device, and seed_seq.
#
#   2. A token scan of the argmin-owned headers the graph actually reaches. Part
#      1 cannot stand alone: <cstdlib> is reachable from Eigen in every build, so
#      rand() and srand() are always callable and no include scan will ever say
#      otherwise. Nor can a hand-rolled generator -- a linear congruential
#      sequence is a dozen tokens and needs no header at all. So the argmin files
#      in the dependency graph are read and rejected if they name any RNG
#      facility: a call whose name contains "rand" (which catches rand, srand,
#      drand48, and a hand-rolled lcg_rand alike), a std engine or distribution
#      type, or Eigen's own Random/setRandom.
#
# The file set for part 2 comes from the dependency graph, not from a glob, so it
# is exactly what the probe pulls in -- transitively. A policy that names no RNG
# itself but reaches one through a header is caught.
#
# The scan reads comments as well as code. That is deliberate: the pattern is
# call-shaped and type-shaped, so ordinary prose about randomness does not match
# it, and the residual risk is a false red on a comment quoting an RNG call --
# which is loud, cheap to fix, and fails in the safe direction.
#
# Invoked via add_test with -P. Required -D arguments:
#   CXX   - the C++ compiler command
#   SRC   - the probe source file
#   INCS  - ';'-separated include directories (argmin + its usage requirements)
#   STD   - the C++ standard flag value (e.g. c++20)
#   OWNED - directory prefix identifying argmin's own headers; only files under
#           it are token-scanned, since third-party code is not argmin's claim
#   SCOPE - human-readable name of the surface the probe covers, for messages

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
    message(FATAL_ERROR "rng-freedom scan: compiler dependency dump failed "
                        "(rc=${scan_rc}):\n${scan_err}")
endif()

string(REGEX MATCHALL "[^ \t\r\n\\\\]+" tokens "${deps}")

# ---- Part 1: no <random> in the transitive include graph --------------------
foreach(tok IN LISTS tokens)
    get_filename_component(base "${tok}" NAME)
    if(base STREQUAL "random")
        message(FATAL_ERROR
            "rng-freedom scan FAILED: <random> reached the ${SCOPE} include "
            "graph via '${tok}'. These surfaces claim that identical inputs "
            "drive an identical trajectory, which rests on their carrying no "
            "source of randomness; a policy that needs an RNG belongs with the "
            "stochastic policies, whose determinism is gated by a seeded "
            "reproducibility test instead.")
    endif()
endforeach()

# ---- Part 2: no RNG facility named in the argmin headers the graph reaches ---
#
# A call whose identifier contains "rand" is required to be call-shaped (a '('
# must follow) so that an English word containing the letters does not match.
set(rng_pattern
    "[A-Za-z_][A-Za-z_0-9]*rand[A-Za-z_0-9]*[ \t\r\n]*\\(|[^A-Za-z_0-9]rand[ \t\r\n]*\\(|mt19937|random_device|default_random_engine|minstd_rand|ranlux|knuth_b|drand48|lrand48|arc4random|getrandom|seed_seq|_distribution|setRandom|::Random[ \t\r\n]*\\(")

set(scanned 0)
foreach(tok IN LISTS tokens)
    string(FIND "${tok}" "${OWNED}" owned_at)
    if(NOT owned_at EQUAL 0)
        continue()
    endif()
    if(NOT EXISTS "${tok}")
        continue()
    endif()

    math(EXPR scanned "${scanned} + 1")
    file(READ "${tok}" content)
    string(REGEX MATCHALL "${rng_pattern}" hits "${content}")
    if(hits)
        list(REMOVE_DUPLICATES hits)
        string(REPLACE ";" ", " hit_text "${hits}")
        message(FATAL_ERROR
            "rng-freedom scan FAILED: '${tok}', reached from the ${SCOPE} "
            "include graph, names a random-number facility: ${hit_text}. These "
            "surfaces claim that identical inputs drive an identical "
            "trajectory, which rests on their carrying no source of "
            "randomness. A generator seeded from a constant is still a "
            "generator: it makes the claim rest on the seeding rather than on "
            "the absence of an RNG, so it must be gated by a seeded "
            "reproducibility test rather than by this scan.")
    endif()
endforeach()

# A scan that reached no files would pass silently while proving nothing at all.
if(scanned EQUAL 0)
    message(FATAL_ERROR
        "rng-freedom scan: no argmin-owned header under '${OWNED}' appears in "
        "the ${SCOPE} dependency graph; the token scan examined nothing, so it "
        "cannot report a verdict. Check the OWNED prefix against the include "
        "paths the dependency dump reports.")
endif()

message(STATUS "rng-freedom scan: OK -- no <random> and no RNG facility named "
               "in the ${scanned} argmin headers the ${SCOPE} graph reaches")
