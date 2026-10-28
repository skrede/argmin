set(SCAN_NAME "the finite-difference Jacobian evaluates into caller-owned buffers instead of allocating per call")
set(SCAN_FILES "lib/argmin/include/argmin/derivative/finite_difference.h")
set(SCAN_REGEX "Eigen::VectorX<Scalar> f_(plus|minus)\\(num_outputs\\);")
