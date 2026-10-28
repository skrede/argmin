set(SCAN_NAME "the CCSA state holds no acceptance flags that only feed their own update")
set(SCAN_FILES "lib/argmin/include/argmin/solver/ccsa_quadratic_policy.h")
set(SCAN_REGEX "bool feasible[{]false[}];|Scalar infeasibility[{]Scalar\\(0\\.0\\)[}];")
