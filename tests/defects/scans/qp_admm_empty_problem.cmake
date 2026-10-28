set(SCAN_NAME "dense ADMM rejects n = 0 before it reduces an empty matrix")
set(SCAN_FILES "lib/argmin/include/argmin/qp/dense_admm_qp.h")
set(SCAN_REGEX "dimension_mismatch;[ \t\r\n]+if\\(n != n_cap_ \\|\\| m > m_cap_\\)")
