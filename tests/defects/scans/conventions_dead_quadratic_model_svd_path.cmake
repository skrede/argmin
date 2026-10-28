set(SCAN_NAME "the BOBYQA quadratic model keeps no test-only SVD interpolation path")
set(SCAN_FILES "lib/argmin/include/argmin/detail/quadratic_model.h")
set(SCAN_REGEX "#include <Eigen/SVD>|quadratic_model<Scalar, N> build_model\\(|Eigen::VectorX<Scalar> compute_lagrange_at_point\\(|Eigen::VectorX<Scalar> update_model_incremental\\(")
