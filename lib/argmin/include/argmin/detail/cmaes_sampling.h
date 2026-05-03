#ifndef HPP_GUARD_ARGMIN_DETAIL_CMAES_SAMPLING_H
#define HPP_GUARD_ARGMIN_DETAIL_CMAES_SAMPLING_H

// CMA-ES offspring sampling and boundary repair.
//
// Production sample_offspring forwards to the empirical winner under
// detail/alternative/. The current alias target is the Marsaglia
// polar variant (Marsaglia & Bray 1964, SIAM Review 6(3)) per the
// perf-record A/B on micro_cmaes:
//   - production (std::normal_distribution<xoshiro256+>):  10.55%
//     self-time on the Gaussian-transform slice
//   - marsaglia (alternative::marsaglia::marsaglia_normal):
//     8.39% self-time, 5-rep median wall 495 ms
//   - ziggurat  (alternative::ziggurat::ziggurat_normal):
//     7.51% self-time, 5-rep median wall 496 ms
// Marsaglia wins the conjoined "lowest combined sample_offspring
// shell + Gaussian draw self-time" tiebreaker (11.86% vs ziggurat's
// 12.02%) at indistinguishable median wall, with simpler
// implementation (no compile-time table; same scalar arithmetic on
// every draw). Loser stays buildable under
// detail/alternative/cmaes_sampling_ziggurat.h for future
// re-comparison per the README lifecycle.
//
// Generates lambda offspring from the multivariate normal
// distribution N(mean, sigma^2 * C) and provides repair+penalty for
// box constraints.
//
// Reference: K&W Section 8.7, Algorithm 8.10 step 1.
//            Hansen (2023) boundary handling tutorial.
//            Marsaglia & Bray (1964), SIAM Review 6(3), 260-264 --
//              Gaussian transform under the production alias.

#include "argmin/detail/alternative/cmaes_sampling_marsaglia.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <random>

namespace argmin::detail
{

// Sample lambda offspring: x_i = mean + sigma * B * D * z_i,
// z_i ~ N(0, I). Returns n x lambda matrix of offspring.
// MaxLambda bounds the column storage for stack allocation when
// known at compile time. RNG must satisfy UniformRandomBitGenerator
// (e.g. detail::xoshiro256, std::mt19937).
//
// Forwarding wrapper to the empirical-winner Marsaglia polar variant
// under detail::alternative::marsaglia. Function templates can't be
// re-exported via `using`; the wrapper preserves the production
// template signature exactly so all callsites compile unchanged.
//
// Reference: K&W Algorithm 8.10 step 1.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int MaxLambda = Eigen::Dynamic, typename RNG = std::mt19937>
auto sample_offspring(
    const Eigen::Vector<Scalar, N>& mean,
    Scalar sigma,
    const Eigen::Matrix<Scalar, N, N>& B,
    const Eigen::Vector<Scalar, N>& D,
    int lambda,
    RNG& rng)
    -> Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0,
        N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxLambda>
{
    return alternative::marsaglia::sample_offspring<Scalar, N, MaxLambda, RNG>(
        mean, sigma, B, D, lambda, rng);
}

// Repair infeasible point by clipping to bounds and return penalty.
// Penalty = sum of squared repair distances. Orthogonal to the
// Gaussian-transform fork; remains in this header for the
// repair_l2_penalty_policy callsite.
// Reference: Hansen boundary handling tutorial (D-03).
template <typename Scalar = double, int N = argmin::dynamic_dimension>
Scalar repair_and_penalize(Eigen::Vector<Scalar, N>& x,
                           const Eigen::Vector<Scalar, N>& lower,
                           const Eigen::Vector<Scalar, N>& upper)
{
    Scalar penalty = Scalar(0);
    for(int i = 0; i < x.size(); ++i)
    {
        if(x[i] < lower[i])
        {
            Scalar d = lower[i] - x[i];
            penalty += d * d;
            x[i] = lower[i];
        }
        else if(x[i] > upper[i])
        {
            Scalar d = x[i] - upper[i];
            penalty += d * d;
            x[i] = upper[i];
        }
    }
    return penalty;
}

}

#endif
