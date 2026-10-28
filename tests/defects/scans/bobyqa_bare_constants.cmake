set(SCAN_NAME "the BOBYQA safety cap, scale guard and model thresholds carry a citation or a derivation")
set(SCAN_FILES "lib/argmin/include/argmin/solver/bobyqa_policy.h;lib/argmin/include/argmin/options/derivative_free_defaults.h;lib/argmin/include/argmin/detail/quadratic_model.h")
set(SCAN_REGEX "[;{][ \t]*\r?\n[ \t]*(const int safety_cap = 200000|static constexpr double scale_guard = 1e-15|static constexpr double (svd_threshold|lagrange_floor) = 1e-12);[ \t]*\r?\n")
