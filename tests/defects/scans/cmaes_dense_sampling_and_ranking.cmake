set(SCAN_NAME "CMA-ES samples through a factored product, forms rank-mu without dense outer products, and ranks an all-feasible population without bubble sweeps")
set(SCAN_FILES "lib/argmin/include/argmin/detail/alternative/cmaes_sampling_marsaglia.h;lib/argmin/include/argmin/detail/cmaes_covariance.h;lib/argmin/include/argmin/detail/stochastic_ranking.h")
set(SCAN_REGEX "B \\* D\\.asDiagonal\\(\\) \\* z|rank_mu\\.noalias\\(\\) \\+= w_i \\* di \\* di\\.transpose\\(\\);|unit_uniform_distribution<Scalar> uniform;[ \t\r\n]+for\\(std::size_t sweep = 0;")
