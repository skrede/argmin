#ifndef HPP_GUARD_NABLAPP_DETAIL_CMAES_SAMPLING_H
#define HPP_GUARD_NABLAPP_DETAIL_CMAES_SAMPLING_H

// CMA-ES offspring sampling and boundary repair.
//
// Generates lambda offspring from the multivariate normal distribution
// N(mean, sigma^2 * C) and provides repair+penalty for box constraints.
//
// Reference: K&W Section 8.7, Algorithm 8.10 step 1.
//            Hansen (2023) boundary handling tutorial.

#include "nablapp/types.h"

#include <Eigen/Core>

#include <random>

namespace nablapp::detail
{

// Sample lambda offspring: x_i = mean + sigma * B * D * z_i, z_i ~ N(0,I).
// Returns n x lambda matrix of offspring.
// MaxLambda bounds the column storage for stack allocation when known at compile time.
// RNG must satisfy UniformRandomBitGenerator (e.g. detail::xoshiro256, std::mt19937).
// Reference: K&W Algorithm 8.10 step 1.
template <typename Scalar = double, int N = nablapp::dynamic_dimension,
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
    const int n = mean.size();
    std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));

    Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0,
        N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxLambda> offspring(n, lambda);
    for(int i = 0; i < lambda; ++i)
    {
        Eigen::Vector<Scalar, N> z(n);
        for(int j = 0; j < n; ++j)
            z[j] = normal(rng);

        offspring.col(i).noalias() = mean + sigma * (B * D.asDiagonal() * z);
    }

    return offspring;
}

// Repair infeasible point by clipping to bounds and return penalty.
// Penalty = sum of squared repair distances.
// Reference: Hansen boundary handling tutorial (D-03).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
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
