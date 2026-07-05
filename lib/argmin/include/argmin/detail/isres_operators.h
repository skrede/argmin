#ifndef HPP_GUARD_ARGMIN_DETAIL_ISRES_OPERATORS_H
#define HPP_GUARD_ARGMIN_DETAIL_ISRES_OPERATORS_H

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

#include "argmin/detail/tuple_contains.h"
#include "argmin/solver/convergence.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <tuple>
#include <limits>
#include <random>
#include <vector>
#include <cstdint>
#include <utility>
#include <type_traits>

namespace argmin::detail
{

// Initialize population uniformly within bounds.
//
// Returns n x lambda matrix. If bounds are infinite on a dimension,
// uses [-1, 1] as a fallback range.
//
// Reference: K&W Section 8.6 (ES initialization).
template <int N = argmin::dynamic_dimension, typename Scalar = double, typename Rng>
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
template <int N = argmin::dynamic_dimension, typename Scalar = double>
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
template <int N = argmin::dynamic_dimension, typename Scalar = double, typename Rng>
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

template <typename Scalar = double, int N = argmin::dynamic_dimension>
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
// Returns: vector of violations per individual.
// violation = sum(|c_eq|) + sum(max(0, -c_ineq))
//
// Reference: N&W eq. 15.24 / 17.44.
template <typename Scalar = double, int M = argmin::dynamic_dimension, int Lambda = argmin::dynamic_dimension>
Eigen::Vector<Scalar, Lambda> compute_violations(
    const Eigen::Matrix<Scalar, M, Lambda>& constraints_matrix,
    int n_eq,
    int n_ineq)
{
    const int lambda = static_cast<int>(constraints_matrix.cols());
    Eigen::Vector<Scalar, Lambda> violations(lambda);

    for(int j = 0; j < lambda; ++j)
    {
        Scalar v = Scalar(0);

        for(int i = 0; i < n_eq; ++i)
            v += std::abs(constraints_matrix(i, j));

        for(int i = 0; i < n_ineq; ++i)
            v += std::max(Scalar(0), -constraints_matrix(n_eq + i, j));

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

// Differential variation operator (Runarsson-Yao 2005, Algorithm 1).
//
// Top-mu DE-style recombination: for k in [0, mu),
//   x[rk] += gamma * (x0_snapshot[anchor_k] - x0_snapshot[anchor_(k+1)])
// componentwise. Caller is responsible for selecting anchor semantics
// (NLopt-faithful: physical-slot indices into the unsorted snapshot;
// runarsson_yao_paper: rank-permuted indices into the snapshot). For
// k+1 == mu the corresponding column is left unmodified; the caller
// owns the standard-mutation fallback for that boundary case.
//
// The lower / upper arguments are accepted for caller-side anchor
// continuity (subsequent variant code may want them for inline bounds
// checks); the operator itself does not clip - bounds enforcement is
// the caller's responsibility (resample-on-bound or clip).
//
// Reference: Runarsson, T. P., and Yao, X. (2005), "Search Biases in
//            Constrained Evolutionary Optimization," IEEE Trans. Systems,
//            Man, and Cybernetics, Part C, 35(2):233-243.
//            K&W 2e Section 8.6.
//            NLopt 2.10.0 isres.c lines 254-260 (operator body).
template <int N = argmin::dynamic_dimension, typename Scalar = double>
inline void differential_variation(
    Eigen::Ref<Eigen::Matrix<Scalar, N, Eigen::Dynamic>> population,
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& x0_snapshot,
    const std::vector<std::uint32_t>& indices,
    int mu,
    Scalar gamma,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    (void)lower;
    (void)upper;
    for(int k = 0; k + 1 < mu; ++k)
    {
        const std::uint32_t rk = indices[static_cast<std::size_t>(k)];
        population.col(rk).array() += gamma
            * (x0_snapshot.col(0).array()
               - x0_snapshot.col(k + 1).array());
    }
}

// Log-normal self-adaptation of sigma with optional upper clamp.
//
// sigma_new = min(sigma * exp(tau_prime_rand + tau * N(0, 1)),
//                 sigma_max).
// Caller pre-samples the per-individual `tau_prime_rand` once per
// individual (constant across components within a single mutation) and
// provides the per-component upper clamp `sigma_max`. To disable the
// clamp pass std::numeric_limits<Scalar>::infinity() for sigma_max.
//
// Reference: Runarsson, T. P., and Yao, X. (2005), IEEE Trans. SMC-C
//            35(2):233-243, Section III; K&W 2e Section 8.6.
//            NLopt 2.10.0 isres.c lines 240-244 (sigma_max =
//            (ub-lb)/sqrt(n) plus per-mutation clamp placement).
template <typename Scalar = double, typename Rng>
inline Scalar log_normal_mutate(
    Scalar sigma,
    Scalar tau,
    Scalar tau_prime_rand,
    Scalar sigma_max,
    Rng& rng)
{
    std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
    const Scalar sigma_new = sigma
        * std::exp(tau_prime_rand + tau * normal(rng));
    return std::min(sigma_new, sigma_max);
}

// Sigma-collapse predicate (bound-relative form).
//
// Returns true when the population's mean per-individual sigma over the
// top-mu survivors falls below `ratio * mean(ub - lb) / sqrt(n)`.
// Default ratio 1e-9 (1000x looser than CMA-ES sigma_collapse_threshold
// because per-individual ISRES sigma is noisier than CMA-ES's
// covariance scaling).
//
// State must expose `s.sigmas` as an N-row matrix and `s.mu` as the
// active survivor count. The function reads only `s.sigmas.leftCols(mu)`.
//
// Reference: Runarsson, T. P., and Yao, X. (2005), IEEE Trans. SMC-C
//            35(2):233-243, Section V (termination); K&W 2e Section 8.6.
//            Hansen, N. (2023), "The CMA Evolution Strategy: A Tutorial,"
//            arXiv:1604.00772, Section B.5 (TolX termination convention,
//            design rationale only).
template <typename State, int N>
inline bool sigma_collapsed_bound_relative(
    const State& s,
    double ratio,
    int n,
    const Eigen::Vector<double, N>& lower,
    const Eigen::Vector<double, N>& upper)
{
    const double mean_sigma = s.sigmas.leftCols(s.mu).mean();
    const double mean_range = (upper - lower).mean();
    return mean_sigma < ratio * mean_range / std::sqrt(static_cast<double>(n));
}

// Sigma-collapse predicate (xtol-coupled form, convergence-policy template).
//
// Reads the step_tolerance_criterion::threshold from the convergence
// policy (a direct-value literature default, never absent -- see
// convergence.h); if the policy lacks step_tolerance_criterion entirely
// (e.g. slsqp_compatible_convergence, which only carries
// step_tolerance_rel_criterion), silently falls back to the
// bound-relative form with the variant's `fallback_ratio`. The
// tuple_contains_v gate keeps the function compilable against any
// convergence type.
//
// Reference: Runarsson, T. P., and Yao, X. (2005), IEEE Trans. SMC-C
//            35(2):233-243, Section V (termination); K&W 2e Section 8.6.
template <typename State, typename Convergence, int N>
inline bool sigma_collapsed_xtol_coupled(
    const State& s,
    const Convergence& convergence,
    double fallback_ratio,
    int n,
    const Eigen::Vector<double, N>& lower,
    const Eigen::Vector<double, N>& upper)
{
    const double mean_sigma = s.sigmas.leftCols(s.mu).mean();
    if constexpr(tuple_contains_v<step_tolerance_criterion,
                                  decltype(convergence.criteria)>)
    {
        const auto& crit = std::get<step_tolerance_criterion>(convergence.criteria);
        return mean_sigma < crit.threshold;
    }
    const double mean_range = (upper - lower).mean();
    return mean_sigma < fallback_ratio * mean_range
                          / std::sqrt(static_cast<double>(n));
}

// Sigma-collapse predicate (xtol-coupled form, captured-threshold overload).
//
// Variant `step()` bodies cannot thread the Convergence type through
// state_type<P>, so init() captures the threshold value at policy
// init-time (`s.convergence_xtol_threshold = std::optional<double>`)
// and step() calls THIS overload with the captured value. If the caller
// passes a finite, positive `threshold_value`, the predicate uses it;
// otherwise it falls back to the bound-relative form with
// `fallback_ratio`.
//
// Reference: Runarsson, T. P., and Yao, X. (2005), IEEE Trans. SMC-C
//            35(2):233-243, Section V (termination); K&W 2e Section 8.6.
template <typename State, int N>
inline bool sigma_collapsed_xtol_coupled(
    const State& s,
    double threshold_value,
    double fallback_ratio,
    int n,
    const Eigen::Vector<double, N>& lower,
    const Eigen::Vector<double, N>& upper)
{
    const double mean_sigma = s.sigmas.leftCols(s.mu).mean();
    if(std::isfinite(threshold_value) && threshold_value > 0.0)
        return mean_sigma < threshold_value;
    const double mean_range = (upper - lower).mean();
    return mean_sigma < fallback_ratio * mean_range
                          / std::sqrt(static_cast<double>(n));
}

}

#endif
