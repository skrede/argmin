# Runs the augmented_lagrangian move-then-step ASan instrument
# (policy_move_asan_test.cpp) and inverts its result: augmented_lagrangian_
# policy's state is move-unsafe today (self-referential subproblem
# pointers copied, not rebased), so the case hard-aborts (SIGABRT / Eigen
# out-of-bounds assertion). A process abort cannot be caught by Catch2's
# [!shouldfail], so this script runs the case out-of-process and treats
# the expected abort as ctest PASS. If augmented_lagrangian_policy's move
# constructor is later fixed to rebase those pointers, the case will exit
# cleanly and this script fails loudly, forcing the [move_asan_xfail] tag
# and this wrapper to be removed so the case rejoins the normal
# [move_asan] suite.
#
# Expected variables (set via -D on the cmake -P invocation):
#   TEST_EXECUTABLE       -- path to argmin_unit_tests
#   TEST_WORKING_DIRECTORY -- ctest working directory for the executable

execute_process(
    COMMAND "${TEST_EXECUTABLE}" "augmented_lagrangian survives move-then-step"
    WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    RESULT_VARIABLE _al_move_asan_result
)

if(_al_move_asan_result STREQUAL "Subprocess aborted")
    message(STATUS
        "augmented_lagrangian survives move-then-step aborted as expected "
        "-- move-safety instrument still firing.")
elseif(_al_move_asan_result EQUAL 0)
    message(FATAL_ERROR
        "augmented_lagrangian survives move-then-step no longer aborts -- "
        "the move-safety bug appears fixed. Remove the [move_asan_xfail] "
        "tag from tests/unit/policy_move_asan_test.cpp and this wrapper "
        "from tests/unit/CMakeLists.txt so the case rejoins the normal "
        "[move_asan] discovery.")
else()
    message(FATAL_ERROR
        "augmented_lagrangian survives move-then-step failed differently "
        "than the expected abort (result: ${_al_move_asan_result}). "
        "Investigate before assuming the move-safety bug is fixed.")
endif()
