set(SCAN_NAME "the moving-asymptote dual problems hold their non-nullable step-state views as references")
set(SCAN_FILES "lib/argmin/include/argmin/detail/ccsa_dual_problem.h;lib/argmin/include/argmin/detail/mma_augmented_dual_problem.h;lib/argmin/include/argmin/detail/mma_raa_augmented_dual_problem.h;lib/argmin/include/argmin/detail/mma_reciprocal_dual_problem.h")
set(SCAN_REGEX "\n[ \t]*const [A-Za-z:<>, ]+[*] ([A-Zabd-z]|c[A-Za-z0-9])[A-Za-z0-9_]*[{][}];")
