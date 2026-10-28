set(SCAN_NAME "the dense active-set QP rejects n = 0 before it reduces an empty matrix")
set(SCAN_FILES "lib/argmin/include/argmin/qp/dense_active_set_qp.h")
set(SCAN_REGEX "dimension_mismatch;[ \n]+if\\(n != n_cap_ \\|\\| m > m_cap_\\)")
