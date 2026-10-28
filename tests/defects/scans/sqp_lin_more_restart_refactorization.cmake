set(SCAN_NAME "the Lin-More restart updates the augmented factorization and keeps the conjugate-gradient forcing reference across restarts")
set(SCAN_FILES "lib/argmin/include/argmin/detail/byrd_omojokun.h;lib/argmin/include/argmin/detail/steihaug_cg.h")
set(SCAN_REGEX "ws\\.ldlt_aug\\.compute\\(ws\\.AAt_aug\\);|const Scalar rg0 = rg;")
