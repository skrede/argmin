#ifndef HPP_GUARD_NABLAPP_DETAIL_ISRES_OPERATORS_H
#define HPP_GUARD_NABLAPP_DETAIL_ISRES_OPERATORS_H

// ISRES mutation, differential variation, and step-size adaptation operators.
//
// Implements the evolutionary operators for Improved Stochastic Ranking
// Evolution Strategy: uniform population initialization, log-normal
// self-adaptive step sizes, differential variation toward the best
// individual, and constraint violation computation.
//
// Reference: Runarsson & Yao (2005), "Search Biases in Constrained
//            Evolutionary Optimization", IEEE Trans. SMC-C.
//            K&W Section 8.6 (evolution strategies).

#include "nablapp/types.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

namespace nablapp::detail
{

// Initialize population uniformly within bounds.
//
// Returns n x lambda matrix. If bounds are infinite on a dimension,
// uses [-1, 1] as a fallback range.
//
// Reference: K&W Section 8.6 (ES initialization).
template <int N = nablapp::dynamic_dimension, typename Scalar = double, typename Rng>
Eigen::Matrix<Scalar, N, Eigen::Dynamic> initialize_population(
    int n, int lambda,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Rng& rng)
{
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
    std::uniform_real_distribution<Scalar> uniform(Scalar(0), Scalar(1));

    Eigen::Matrix<Scalar, N, Eigen::Dynamic> pop(n, lambda);

    for(int j = 0; j < lambda; ++j)
    {
        for(int i = 0; i < n; ++i)
        {
            Scalar lo = (lower[i] > -inf) ? lower[i] : Scalar(-1);
            Scalar hi = (upper[i] <  inf) ? upper[i] : Scalar(1);
            pop(i, j) = lo + uniform(rng) * (hi - lo);
        }
    }

    return pop;
}

// Initialize step sizes from bound widths.
//
// Returns n x lambda matrix. sigma_i = (upper_i - lower_i) / sqrt(n).
// If infinite range on a dimension, uses 1.0.
//
// Reference: Runarsson & Yao (2005).
template <int N = nablapp::dynamic_dimension, typename Scalar = double>
Eigen::Matrix<Scalar, N, Eigen::Dynamic> initialize_sigmas(
    int n, int lambda,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
    Scalar sqrt_n = std::sqrt(static_cast<Scalar>(n));

    Eigen::Matrix<Scalar, N, Eigen::Dynamic> sigmas(n, lambda);

    for(int i = 0; i < n; ++i)
    {
        Scalar range = (lower[i] > -inf && upper[i] < inf)
            ? (upper[i] - lower[i])
            : sqrt_n;
        Scalar s = range / sqrt_n;
        sigmas.row(i).setConstant(s);
    }

    return sigmas;
}

// Mutate a single individual with log-normal step-size adaptation
// and differential variation.
//
// Returns (offspring position, new sigma).
//
// Steps:
//   sigma_new = sigma * exp(tau * N(0,1) + tau_prime * N_i(0,1))
//   x_new = parent + sigma_new .* N(0,I) + alpha * (x_best - parent)
//   Clamp x_new to [lower, upper]
//
// Reference: Runarsson & Yao (2005), K&W Section 8.6.
template <int N = nablapp::dynamic_dimension, typename Scalar = double, typename Rng>
std::pair<Eigen::Vector<Scalar, N>, Eigen::Vector<Scalar, N>> mutate_individual(
    const Eigen::Vector<Scalar, N>& parent,
    const std::type_identity_t<Eigen::Vector<Scalar, N>>& sigma,
    const Eigen::Vector<Scalar, N>& x_best,
    Scalar alpha,
    Scalar tau,
    Scalar tau_prime,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Rng& rng)
{
    const int n = parent.size();
    std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));

    // Log-normal self-adaptation (element-wise)
    Scalar global_noise = tau * normal(rng);
    Eigen::Vector<Scalar, N> sigma_new(n);
    for(int i = 0; i < n; ++i)
        sigma_new[i] = sigma[i] * std::exp(global_noise + tau_prime * normal(rng));

    // Gaussian perturbation + differential variation
    Eigen::Vector<Scalar, N> x_new(n);
    for(int i = 0; i < n; ++i)
    {
        x_new[i] = parent[i]
                  + sigma_new[i] * normal(rng)
                  + alpha * (x_best[i] - parent[i]);
    }

    // Clamp to bounds
    x_new = x_new.cwiseMax(lower).cwiseMin(upper);

    return {x_new, sigma_new};
}

// Stateful ISRES operator workspace with pre-allocated mutation temporaries.
//
// Pre-allocates the per-individual mutation vectors (sigma_new, x_new) that
// are created on every mutation call to eliminate per-generation allocation.
//
// Reference: Runarsson & Yao (2005), K&W Section 8.6.

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class isres_operator_workspace
{
public:
    explicit isres_operator_workspace(int n)
        : sigma_new_(n)
        , x_new_(n)
    {
    }

    isres_operator_workspace() = default;

    // Mutate a single individual using pre-allocated workspace.
    //
    // Reference: Runarsson & Yao (2005), K&W Section 8.6.
    template <typename Rng>
    std::pair<Eigen::Vector<Scalar, N>, Eigen::Vector<Scalar, N>> mutate(
        const Eigen::Vector<Scalar, N>& parent,
        const std::type_identity_t<Eigen::Vector<Scalar, N>>& sigma,
        const Eigen::Vector<Scalar, N>& x_best,
        Scalar alpha,
        Scalar tau,
        Scalar tau_prime,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        Rng& rng)
    {
        const int n = parent.size();
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));

        Scalar global_noise = tau * normal(rng);
        for(int i = 0; i < n; ++i)
            sigma_new_[i] = sigma[i] * std::exp(global_noise + tau_prime * normal(rng));

        for(int i = 0; i < n; ++i)
        {
            x_new_[i] = parent[i]
                       + sigma_new_[i] * normal(rng)
                       + alpha * (x_best[i] - parent[i]);
        }

        x_new_ = x_new_.cwiseMax(lower).cwiseMin(upper);

        return {x_new_, sigma_new_};
    }

private:
    Eigen::Vector<Scalar, N> sigma_new_;
    Eigen::Vector<Scalar, N> x_new_;
};

// Compute total constraint violation for each individual.
//
// constraints_matrix: (n_eq + n_ineq) x lambda matrix of constraint values.
// First n_eq rows are equality constraints (must be zero).
// Remaining n_ineq rows are inequality constraints (c >= 0 convention).
//
// Returns: VectorXd of violations per individual.
// violation = sum(|c_eq|) + sum(max(0, -c_ineq))
//
// Reference: N&W eq. 15.24 / 17.44.
inline Eigen::VectorXd compute_violations(
    const Eigen::MatrixXd& constraints_matrix,
    int n_eq,
    int n_ineq)
{
    const int lambda = static_cast<int>(constraints_matrix.cols());
    Eigen::VectorXd violations(lambda);

    for(int j = 0; j < lambda; ++j)
    {
        double v = 0.0;

        for(int i = 0; i < n_eq; ++i)
            v += std::abs(constraints_matrix(i, j));

        for(int i = 0; i < n_ineq; ++i)
            v += std::max(0.0, -constraints_matrix(n_eq + i, j));

        violations[j] = v;
    }

    return violations;
}

// Standard ES learning rates for log-normal step-size adaptation.
//
// tau = 1 / sqrt(2*n), tau_prime = 1 / sqrt(2*sqrt(n))
//
// Reference: K&W Section 8.6.
struct es_learning_rates
{
    double tau;
    double tau_prime;
};

inline es_learning_rates compute_es_rates(int n)
{
    double nd = static_cast<double>(n);
    return {
        .tau = 1.0 / std::sqrt(2.0 * nd),
        .tau_prime = 1.0 / std::sqrt(2.0 * std::sqrt(nd)),
    };
}

}

#endif
