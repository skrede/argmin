#ifndef HPP_GUARD_NABLAPP_DETAIL_CMAES_SAMPLING_H
#define HPP_GUARD_NABLAPP_DETAIL_CMAES_SAMPLING_H

// CMA-ES offspring sampling and boundary repair.
//
// Generates lambda offspring from the multivariate normal distribution
// N(mean, sigma^2 * C) and provides repair+penalty for box constraints.
//
// Reference: K&W Section 8.7, Algorithm 8.10 step 1.
//            Hansen (2023) boundary handling tutorial.

#include <Eigen/Core>

#include <random>

namespace nablapp::detail
{

// Sample lambda offspring: x_i = mean + sigma * B * D * z_i, z_i ~ N(0,I).
// Returns n x lambda matrix of offspring.
// Reference: K&W Algorithm 8.10 step 1.
inline Eigen::MatrixXd sample_offspring(const Eigen::VectorXd& mean,
                                        double sigma,
                                        const Eigen::MatrixXd& B,
                                        const Eigen::VectorXd& D,
                                        int lambda,
                                        std::mt19937& rng)
{
    const int n = mean.size();
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd offspring(n, lambda);
    for(int i = 0; i < lambda; ++i)
    {
        Eigen::VectorXd z(n);
        for(int j = 0; j < n; ++j)
            z[j] = normal(rng);

        offspring.col(i) = mean + sigma * (B * D.asDiagonal() * z);
    }

    return offspring;
}

// Repair infeasible point by clipping to bounds and return penalty.
// Penalty = sum of squared repair distances.
// Reference: Hansen boundary handling tutorial (D-03).
inline double repair_and_penalize(Eigen::VectorXd& x,
                                  const Eigen::VectorXd& lower,
                                  const Eigen::VectorXd& upper)
{
    double penalty = 0.0;
    for(int i = 0; i < x.size(); ++i)
    {
        if(x[i] < lower[i])
        {
            double d = lower[i] - x[i];
            penalty += d * d;
            x[i] = lower[i];
        }
        else if(x[i] > upper[i])
        {
            double d = x[i] - upper[i];
            penalty += d * d;
            x[i] = upper[i];
        }
    }
    return penalty;
}

}

#endif
