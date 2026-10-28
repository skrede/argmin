set(SCAN_NAME "every comparison against an SQP or L-BFGS-B zero guard is scaled by the magnitude it guards")
set(SCAN_FILES "lib/argmin/include/argmin/*.h")
set(SCAN_REGEX "(<|>|<=|>=)[ \t]*(line_search_sqp_defaults|trust_region_sqp_defaults|filter_restoration_constants|lbfgsb_direction_constants)<Scalar>::zero_guard[ \t]*(\\)|&&)")
