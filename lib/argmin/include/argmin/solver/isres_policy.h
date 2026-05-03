#ifndef HPP_GUARD_ARGMIN_SOLVER_ISRES_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ISRES_POLICY_H

// ISRES (Improved Stochastic Ranking Evolution Strategy) production alias.
//
// Aliases the empirical winner among the variants in
// `solver/alternative/isres/`. Initial alias is `nlopt_faithful_policy`
// (least-controversial vs the citation trail). The alias is
// resettable to the empirical winner once `benchmarks/micro_isres`
// results are in.
//
// References:
//   Runarsson, T. P., and Yao, X. (2005), "Search Biases in
//     Constrained Evolutionary Optimization," IEEE Trans. Systems,
//     Man, and Cybernetics, Part C, 35(2):233-243.
//   Kochenderfer, M. J., and Wheeler, T. A., "Algorithms for
//     Optimization", 2e, MIT Press 2019, Section 8.6 (Evolution
//     Strategies).

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"

namespace argmin
{

template <int N = dynamic_dimension>
using isres_policy = alternative::isres::nlopt_faithful_policy<N>;

}

#endif
