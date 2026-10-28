set(SCAN_NAME "the SQP state allocates no buffer its policy never reads, and the dense LDL BFGS holds no unused product")
set(SCAN_FILES "lib/argmin/include/argmin/detail/sqp_common.h;lib/argmin/include/argmin/detail/dense_ldl_bfgs.h")
set(SCAN_REGEX "\n[ \t]+lag_A_buf\\.resize\\(m, n\\);\r?\n\r?\n[ \t]+E_buf\\.resize\\(n, n\\);\r?\n[ \t]+f_buf\\.resize\\(n\\);|Eigen::Vector<Scalar, N> Bs_;")
