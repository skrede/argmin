set(SCAN_NAME "the moving-asymptote policies and dual problems reject alpha above beta before clamping the primal")
set(SCAN_FILES "lib/argmin/include/argmin/solver/mma_policy.h;lib/argmin/include/argmin/solver/alternative/gcmma/*.h;lib/argmin/include/argmin/detail/mma_*dual_problem.h")
set(SCAN_REGEX "alpha(\\[[a-z]+\\])?\\)? *(>|<=) *\\(?beta|beta(\\[[a-z]+\\])?\\)? *(<|>=) *\\(?alpha|\\(alpha(\\.array\\(\\))? *> *beta")
set(SCAN_ABSENT TRUE)
