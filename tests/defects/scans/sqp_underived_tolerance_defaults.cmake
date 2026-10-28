set(SCAN_NAME "the SQP relative step tolerances and the minimum trust radius carry a derivation or a relative form")
set(SCAN_FILES "lib/argmin/include/argmin/options/line_search_sqp_defaults.h;lib/argmin/include/argmin/options/trust_region_sqp_defaults.h")
set(SCAN_REGEX "[;{][ \t]*\r?\n[ \t]*static constexpr double (step_tolerance_rel|step_tolerance_rel_accurate|min_trust_radius) = 1e-12")
