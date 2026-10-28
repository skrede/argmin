set(SCAN_NAME "an adaptive rho update reuses cached per-class Gram blocks instead of rebuilding A^T diag(rho) A")
set(SCAN_FILES "lib/argmin/include/argmin/qp/dense_admm_qp.h")
set(SCAN_REGEX "rebuild_rho_vec\\(\\);[ \t\r\n]+posed_ = build_and_factor_K\\(\\);")
