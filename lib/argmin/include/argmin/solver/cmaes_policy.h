#ifndef HPP_GUARD_ARGMIN_SOLVER_CMAES_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_CMAES_POLICY_H

// CMA-ES production policy.
//
// cmaes_policy is a thin alias to the empirical winner among the
// boundary-handling variants under solver/alternative/cmaes/. The
// variants are A/B-measured on bounded global cells; the alias is
// updated in-place when the empirical-winner status changes. Losing
// variants stay buildable under solver/alternative/cmaes/ for
// re-comparison per solver/alternative/README.md lifecycle.
//
// Empirical winner: pwq_reparameterization_policy (libcmaes
// pwq_bound_strategy.cc:35-125 invertible reparameterization).
//
// Verdict source: 5-seed publish_bench median over the bounded global
// cells (rastrigin_2/10, griewank_2, schwefel_2, ackley_2/10). On the
// binding D-08 cell schwefel_2, pwq is the only variant that ever
// reaches the libcmaes_ipop optimum (2.5e-05) at fixed seed; on
// griewank_2 pwq has the lowest 5-seed median; on the rastrigin /
// ackley cells the variants are within noise. By sum of log10 median
// objective across the 6 cells: pwq -14.80, no_repair -14.58, l2
// -14.34. Both losing variants stay under solver/alternative/cmaes/
// per the README.md lifecycle so the comparison is re-runnable.

#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"

#include "argmin/types.h"

namespace argmin
{

template <int N = dynamic_dimension, int MaxPopulation = dynamic_dimension>
using cmaes_policy = alternative::cmaes::pwq_reparameterization_policy<N, MaxPopulation>;

}

#endif
