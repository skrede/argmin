#ifndef HPP_GUARD_NABLAPP_DETAIL_STOCHASTIC_RANKING_H
#define HPP_GUARD_NABLAPP_DETAIL_STOCHASTIC_RANKING_H

// Stochastic ranking for constrained evolutionary optimization.
//
// Bubble-sort ranking with probabilistic comparison that balances
// objective fitness against constraint violation. Feasible individuals
// are compared by objective; infeasible individuals are compared by
// violation with probability (1 - pf).
//
// Reference: Runarsson & Yao (2005), "Search Biases in Constrained
//            Evolutionary Optimization", IEEE Trans. SMC-C, eq. 2.

#include <Eigen/Core>

#include <cstdint>
#include <random>
#include <vector>

namespace nablapp::detail
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
template <typename Rng>
void stochastic_rank(std::vector<std::uint32_t>& indices,
                     const Eigen::VectorXd& objectives,
                     const Eigen::VectorXd& violations,
                     double pf,
                     Rng& rng)
{
    const auto n = indices.size();
    if(n <= 1)
        return;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for(std::size_t sweep = 0; sweep < n; ++sweep)
    {
        bool swapped = false;

        for(std::size_t j = 0; j + 1 < n; ++j)
        {
            auto a = indices[j];
            auto b = indices[j + 1];

            bool compare_by_objective =
                (violations[a] <= 0.0 && violations[b] <= 0.0)
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
