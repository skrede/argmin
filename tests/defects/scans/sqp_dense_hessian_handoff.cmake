set(SCAN_NAME "the line-search SQP policies hand the QP the factored Hessian instead of a materialized dense one")
set(SCAN_FILES "lib/argmin/include/argmin/solver/nw_sqp_policy.h;lib/argmin/include/argmin/solver/filter_nw_sqp_policy.h")
set(SCAN_REGEX "(solve_into\\(|\n[ \t]*)s\\.hessian\\.hessian\\(\\),")
