set(SCAN_NAME "the SLSQP second-order correction re-solves only the right-hand side of the factored main QP")
set(SCAN_FILES "lib/argmin/include/argmin/solver/kraft_slsqp_policy.h;lib/argmin/include/argmin/solver/filter_slsqp_policy.h")
set(SCAN_REGEX "solve_with_factored_hessian_into(<[^>]*>)?\\([ \t\r\n]*soc_res,")
