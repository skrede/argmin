set(SCAN_NAME "move_limit_shrink and raa_augmented reject equality rows at run time before the constraint probe")
set(SCAN_FILES "lib/argmin/include/argmin/solver/alternative/gcmma/move_limit_shrink_policy.h;lib/argmin/include/argmin/solver/alternative/gcmma/raa_augmented_policy.h")
set(SCAN_REGEX "assert\\(problem\\.num_equality\\(\\) == 0[^;]*;[ \t\r\n]+const int n = problem\\.dimension\\(\\);[ \t\r\n]+const int m = problem\\.num_inequality\\(\\);")
