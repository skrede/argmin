set(SCAN_NAME "options with a literature default hold the value, not an optional resolved to it")
set(SCAN_FILES "lib/argmin/include/argmin/solver/lm_policy.h;lib/argmin/include/argmin/solver/bobyqa_policy.h")
set(SCAN_REGEX "std::optional<Scalar> (tau|nu_initial|damping_factor|diagonal_min_clamp|lambda_min|lambda_max|final_trust_radius)[{][}];")
