#ifndef HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_RANKING_H
#define HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_RANKING_H

// Stochastic ranking for constrained evolutionary optimization.
//
// Bubble-sort ranking with probabilistic comparison that balances
// objective fitness against constraint violation. Feasible individuals
// are compared by objective; infeasible individuals are compared by
// violation with probability (1 - pf).
//
// Reference: Runarsson & Yao (2005), "Search Biases in Constrained
//            Evolutionary Optimization", IEEE Trans. SMC-C, eq. 2.

#include "argmin/detail/scalar_distributions.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <vector>
#include <cstdint>

namespace argmin::detail
{

// Stochastic ranking: sort indices by a probabilistic mix of
// objective value and constraint violation.
//
// indices: permutation to sort (modified in place)
// objectives: f(x_i) values (lower is better)
// violations: total constraint violation (0 = feasible, >0 = infeasible)
// pf: probability of comparing by objective even when infeasible
//     (Runarsson & Yao 2005 recommend pf = 0.45)
// rng: random number generator
//
// Reference: Runarsson & Yao (2005), Algorithm 1, eq. 2.
template <scalar Scalar, int Lambda = argmin::dynamic_dimension, typename Rng>
void stochastic_rank(std::vector<std::uint32_t>& indices,
                     const Eigen::Vector<Scalar, Lambda>& objectives,
                     const Eigen::Vector<Scalar, Lambda>& violations,
                     Scalar pf,
                     Rng& rng)
{
    const auto n = indices.size();
    if(n <= 1)
        return;

    unit_uniform_distribution<Scalar> uniform;

    for(std::size_t sweep = 0; sweep < n; ++sweep)
    {
        bool swapped = false;

        for(std::size_t j = 0; j + 1 < n; ++j)
        {
            auto a = indices[j];
            auto b = indices[j + 1];

            bool compare_by_objective =
                (violations[a] <= Scalar(0) && violations[b] <= Scalar(0))
                || uniform(rng) < pf;

            bool should_swap = compare_by_objective
                ? objectives[a] > objectives[b]
                : violations[a] > violations[b];

            if(should_swap)
            {
                std::swap(indices[j], indices[j + 1]);
                swapped = true;
            }
        }

        if(!swapped)
            break;
    }
}

}

#endif
