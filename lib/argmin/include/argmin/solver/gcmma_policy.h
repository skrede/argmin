#ifndef HPP_GUARD_ARGMIN_SOLVER_GCMMA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_GCMMA_POLICY_H

// GCMMA (Globally Convergent MMA) — production alias.
//
// gcmma_policy aliases argmin::alternative::gcmma::rho_wval_policy,
// chosen as the production GCMMA implementation by the empirical
// comparison reported in benchmarks/micro_mma.cpp:
//
//   - rho_wval wins the GCMMA-variant comparison (wall time and outer
//     iter count) on HS024, HS043, and HS076.
//   - The strict Svanberg 2002 raa-augmented variant converges
//     correctly but is approximately 2x slower than rho_wval; the
//     asymptote-divergent d_j penalty does not pay off in practice on
//     this benchmark set.
//   - The move-limit-shrink variant fails to reach the optimum on
//     HS076 and burns wall time on contracted trust-region dual
//     solves; it is non-viable as a default.
//
// Both losing variants are preserved in solver/alternative/gcmma/ for
// reproducibility of the comparison reported in the paper.
//
// Algorithmically: MMA reciprocal approximation augmented with a
// separable quadratic penalty whose weight (rho) grows on
// non-conservative trials per the NLopt mma.c formula
// (Svanberg 2002 SIAM J. Optim. 12(2), Section 4.2; NLopt
// ccsa_quadratic.c lines 388-391). Per-component primal x_j(y)
// computed via Newton iteration in the augmented dual.

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"

#include "argmin/types.h"

namespace argmin
{

template <int N = dynamic_dimension,
          template<int> typename DualPolicy = lbfgsb_policy>
using gcmma_policy = alternative::gcmma::rho_wval_policy<N, DualPolicy>;

}

#endif
