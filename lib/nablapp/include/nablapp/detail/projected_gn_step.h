#ifndef HPP_GUARD_NABLAPP_DETAIL_PROJECTED_GN_STEP_H
#define HPP_GUARD_NABLAPP_DETAIL_PROJECTED_GN_STEP_H

// Shared detail helpers for projected Gauss-Newton with box constraints.
//
// Provides active-set identification, reduced Hessian LDLT solve, dogleg
// trust-region interpolation, trust-region radius update, and predicted
// reduction for both Nielsen/LM and dogleg/TR globalization modes.
//
// Reference: Nocedal & Wright, Numerical Optimization, 2nd ed., Springer 2006.
//            Sections 4.1 (trust region), 10.2-10.3 (GN), 16.6 (active set).
//            Nielsen, H. B. (1999) "Damping Parameter in Marquardt's Method",
//            IMM-REP-1999-05.
//            Kochenderfer & Wheeler, Algorithms for Optimization, 2nd ed.,
//            MIT Press 2025. Section 6.3 (Levenberg-Marquardt).

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace nablapp::detail
{

// Identify free (non-active) variable indices for active-set projection.
//
// A variable i is active if it sits at a bound and the gradient points into
// that bound (descent direction would push further into the constraint).
// In nablapp convention g = J^T r is the gradient of 0.5*||r||^2:
//   - at_lower AND g(i) > 0: gradient points toward lower bound (active)
//   - at_upper AND g(i) < 0: gradient points toward upper bound (active)
//
// Reference: N&W Section 16.6 (active-set identification for bound constraints).

inline std::vector<int> identify_free_set(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper)
{
    const int n = static_cast<int>(x.size());
    std::vector<int> free_indices;
    free_indices.reserve(static_cast<std::size_t>(n));

    for(int i = 0; i < n; ++i)
    {
        double eps = std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(x(i)));
        bool at_lower = x(i) <= lower(i) + eps;
        bool at_upper = x(i) >= upper(i) - eps;
        bool active = (at_lower && gradient(i) > 0.0) ||
                      (at_upper && gradient(i) < 0.0);
        if(!active)
            free_indices.push_back(i);
    }

    return free_indices;
}

// Extract H_free submatrix and g_free subvector from free indices.
//
// Helper for solve_reduced_gn and dogleg_step to avoid code duplication.

inline void extract_reduced_system(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& gradient,
    const std::vector<int>& free_indices,
    Eigen::MatrixXd& H_free,
    Eigen::VectorXd& g_free)
{
    int n_free = static_cast<int>(free_indices.size());
    H_free.resize(n_free, n_free);
    g_free.resize(n_free);

    for(int i = 0; i < n_free; ++i)
    {
        auto fi = static_cast<std::size_t>(i);
        g_free(i) = gradient(free_indices[fi]);
        for(int j = 0; j < n_free; ++j)
        {
            auto fj = static_cast<std::size_t>(j);
            H_free(i, j) = H(free_indices[fi], free_indices[fj]);
        }
    }
}

// Scatter free-variable step back into full n-vector, zeroing active components.

inline Eigen::VectorXd scatter_to_full(
    const Eigen::VectorXd& h_free,
    const std::vector<int>& free_indices,
    int n)
{
    Eigen::VectorXd h = Eigen::VectorXd::Zero(n);
    int n_free = static_cast<int>(free_indices.size());
    for(int i = 0; i < n_free; ++i)
        h(free_indices[static_cast<std::size_t>(i)]) = h_free(i);
    return h;
}

// Solve the reduced Gauss-Newton system on the free subspace.
//
// Extracts H_free and g_free from free_indices, adds diagonal-scaled damping
// H_free(i,i) += lambda * max(H_free(i,i), diagonal_min_clamp), then solves
// H_free * h_free = -g_free via LDLT. Active components are zeroed.
//
// Reference: K&W Eq. 6.11 (Marquardt diagonal scaling).
//            N&W Section 10.2-10.3 (Gauss-Newton normal equations).

inline Eigen::VectorXd solve_reduced_gn(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& gradient,
    const std::vector<int>& free_indices,
    double lambda,
    double diagonal_min_clamp)
{
    int n = static_cast<int>(gradient.size());
    int n_free = static_cast<int>(free_indices.size());

    if(n_free == 0)
        return Eigen::VectorXd::Zero(n);

    Eigen::MatrixXd H_free;
    Eigen::VectorXd g_free;
    extract_reduced_system(H, gradient, free_indices, H_free, g_free);

    // Diagonal-scaled damping (K&W Eq. 6.11, Marquardt 1963)
    for(int i = 0; i < n_free; ++i)
        H_free(i, i) += lambda * std::max(H_free(i, i), diagonal_min_clamp);

    Eigen::VectorXd h_free = H_free.ldlt().solve(-g_free);
    return scatter_to_full(h_free, free_indices, n);
}

// Extract columns at free_indices from the Jacobian.

inline Eigen::MatrixXd extract_jacobian_free(
    const Eigen::MatrixXd& J,
    const std::vector<int>& free_indices)
{
    int m = static_cast<int>(J.rows());
    int n_free = static_cast<int>(free_indices.size());
    Eigen::MatrixXd J_free(m, n_free);
    for(int i = 0; i < n_free; ++i)
        J_free.col(i) = J.col(free_indices[static_cast<std::size_t>(i)]);
    return J_free;
}

// Dogleg trust-region step on the free subspace.
//
// Interpolates between the scaled Cauchy point and the GN step within the
// trust-region radius delta. Returns step in full n-space (active zeroed).
//
// Reference: N&W Section 4.1 (dogleg method), Algorithm 4.1.

inline Eigen::VectorXd dogleg_step(
    const Eigen::MatrixXd& J,
    const Eigen::MatrixXd& H_free,
    const Eigen::VectorXd& g_free,
    const std::vector<int>& free_indices,
    int n,
    double delta)
{
    int n_free = static_cast<int>(free_indices.size());
    if(n_free == 0)
        return Eigen::VectorXd::Zero(n);

    auto J_free = extract_jacobian_free(J, free_indices);

    // Cauchy scaling: t = ||g||^2 / ||J*g||^2 (steepest descent optimal step)
    double g_sq = g_free.squaredNorm();
    Eigen::VectorXd Jg = (J_free * g_free).eval();
    double Jg_sq = Jg.squaredNorm();
    double t = (Jg_sq > std::numeric_limits<double>::epsilon()) ? g_sq / Jg_sq : 1.0;

    // GN step via LDLT with small epsilon regularization
    Eigen::MatrixXd H_reg = H_free;
    for(int i = 0; i < n_free; ++i)
        H_reg(i, i) += 100.0 * std::numeric_limits<double>::epsilon();
    Eigen::VectorXd gn = H_reg.ldlt().solve(-g_free);

    double gn_norm = gn.norm();
    if(gn_norm <= delta)
        return scatter_to_full(gn, free_indices, n);

    double sd_scaled_norm = t * g_free.norm();
    if(sd_scaled_norm >= delta)
    {
        // Scale Cauchy direction to trust-region boundary
        Eigen::VectorXd cauchy = (-delta / g_free.norm()) * g_free;
        return scatter_to_full(cauchy, free_indices, n);
    }

    // Interpolate along Cauchy-to-GN segment: p = a + beta*(b - a), ||p|| = delta
    Eigen::VectorXd a = -t * g_free;
    Eigen::VectorXd d = gn - a;
    double a_sq = a.squaredNorm();
    double d_sq = d.squaredNorm();
    double a_dot_d = a.dot(d);
    double delta_sq = delta * delta;

    double discriminant = a_dot_d * a_dot_d - d_sq * (a_sq - delta_sq);
    double beta = (-a_dot_d + std::sqrt(std::max(0.0, discriminant))) / d_sq;
    beta = std::clamp(beta, 0.0, 1.0);

    Eigen::VectorXd h_free = a + beta * d;
    return scatter_to_full(h_free, free_indices, n);
}

// Update trust-region radius based on gain ratio.
//
// Reference: N&W Algorithm 4.1 (trust-region radius update).

inline void update_trust_region(
    double& delta, double rho, double step_norm,
    double expand_threshold = 0.75,
    double shrink_threshold = 0.25)
{
    if(rho > expand_threshold && step_norm >= 0.9 * delta)
        delta = 2.0 * delta;
    else if(rho < shrink_threshold)
        delta *= 0.25;
}

// Predicted reduction for Nielsen/LM mode.
//
// pred = 0.5 * h^T (lambda * D * h - g) where D = diag(max(H_ii, clamp)).
//
// Reference: Nielsen (1999) Eq. 2.6, N&W Eq. 10.17.

inline double predicted_reduction_lm(
    const Eigen::VectorXd& step,
    const Eigen::VectorXd& gradient,
    double lambda,
    double diagonal_min_clamp,
    const Eigen::VectorXd& H_diag)
{
    int n = static_cast<int>(step.size());
    Eigen::VectorXd D_h(n);
    for(int i = 0; i < n; ++i)
        D_h(i) = std::max(H_diag(i), diagonal_min_clamp) * step(i);
    return 0.5 * step.dot(lambda * D_h - gradient);
}

// Predicted reduction for dogleg/trust-region mode.
//
// pred = -g^T h - 0.5 * h^T H h
//
// Reference: N&W Eq. 4.2 (quadratic model reduction).

inline double predicted_reduction_tr(
    const Eigen::VectorXd& step,
    const Eigen::VectorXd& gradient,
    const Eigen::MatrixXd& H)
{
    return -gradient.dot(step) - 0.5 * step.dot(H * step);
}

}

#endif
