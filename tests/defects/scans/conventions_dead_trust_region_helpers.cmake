set(SCAN_NAME "the trust-region helpers keep no function or options struct that no solver calls")
set(SCAN_FILES "lib/argmin/include/argmin/detail/trust_region.h;lib/argmin/include/argmin/options/trust_region_options.h")
set(SCAN_REGEX "Scalar compute_rho\\(|Scalar update_radius\\(|int check_geometry\\(|altmov_(line_search|cauchy_step|geometry_step)\\(\r?\n[ \t]+const Eigen::Matrix<Scalar, N, Eigen::Dynamic>&|struct trust_region_options")
