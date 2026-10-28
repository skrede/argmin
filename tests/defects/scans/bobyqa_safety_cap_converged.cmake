set(SCAN_NAME "the BOBYQA driver safety cap does not end in a converged status")
set(SCAN_FILES "lib/argmin/include/argmin/solver/bobyqa_policy.h")
set(SCAN_REGEX "if\\(\\+\\+safety > safety_cap\\)[ \t\r\n]*{[ \t\r\n]*terminated = true;.*if\\(terminated\\)[ \t\r\n]*r\\.policy_status = solver_status::converged;")
