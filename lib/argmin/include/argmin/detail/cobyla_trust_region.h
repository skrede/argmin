#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRUST_REGION_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRUST_REGION_H

// Trust-region subproblem solver for COBYLA.
//
// Solves the linearised trust-region subproblem: minimise a linear
// objective subject to linearised constraint satisfaction, a trust-region
// bound, and box constraints. Also provides the trust-radius update
// schedule from Powell's COBYLA algorithm.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization
//            method that models the objective and constraint functions
//            by linear interpolation."
//            K&W 2e, Section 10.7.

#include "argmin/detail/bound_projection.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

// Clip a displacement vector d to the intersection of the trust ball
// ||d|| <= rho and box bounds lower <= x + d <= upper.
template <typename Scalar = double, int N = argmin::dynamic_dimension>
Eigen::Vector<Scalar, N> clip_to_trust_and_bounds(
    const Eigen::Vector<Scalar, N>& d,
    Scalar rho,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    const int n = static_cast<int>(d.size());
    Eigen::Vector<Scalar, N> result = d;

    for(int i = 0; i < n; ++i)
        result[i] = std::clamp(result[i], lower[i] - x[i], upper[i] - x[i]);

    if(result.norm() > rho)
        result *= rho / result.norm();

    return result;
}

// Evaluate linearised constraint violation for displacement d.
//
// For inequality constraints: violation if g_j^T d + offset_j < 0.
// For equality constraints (indices [0, n_eq)): both directions.
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
Scalar linearised_violation(
    const Eigen::Matrix<Scalar, M, N>& constraint_gradients,
    const Eigen::Vector<Scalar, M>& constraint_offsets,
    int n_eq,
    const Eigen::Vector<Scalar, N>& d)
{
    const int m = static_cast<int>(constraint_offsets.size());
    Scalar viol = Scalar(0);

    for(int j = 0; j < m; ++j)
    {
        Scalar val = constraint_gradients.row(j).dot(d) + constraint_offsets[j];
        if(j < n_eq)
            viol += std::abs(val);
        else
            viol += std::max(Scalar(0), -val);
    }

    return viol;
}

// Stateful trust-region subproblem solver with pre-allocated workspace.
//
// Pre-allocates the expanded constraint matrices, gradient, and displacement
// vectors used by solve_linear_subproblem.
//
// Reference: Powell 1994, Section 3 (trust-region subproblem).

template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
class cobyla_trust_region_solver
{
    // M_expanded: M constraints + up to M equality duplications (worst case all eq).
    // When M is compile-time, use 2*M as max bound; when dynamic, stay dynamic.
    static constexpr int Mexp = (M == argmin::dynamic_dimension)
        ? argmin::dynamic_dimension : 2 * M;

public:
    explicit cobyla_trust_region_solver(int n, int m_total, int n_eq)
        : n_{n}
        , m_expanded_{m_total + n_eq}
        , G_(m_total + n_eq, n)
        , h_(m_total + n_eq)
        , d_(n)
        , grad_(n)
        , d_new_(n)
    {
    }

    cobyla_trust_region_solver() = default;

    // Solve the COBYLA trust-region subproblem using pre-allocated workspace.
    //
    // Minimise  g_obj^T d
    // subject to  g_cj^T d + offset_j >= 0   (inequality, j >= n_eq)
    //             g_cj^T d + offset_j  = 0   (equality, j < n_eq)
    //             ||d|| <= rho
    //             lower <= x + d <= upper
    //
    // Equality constraints are handled by converting h(x) = 0 to
    // h(x) >= 0 AND -h(x) >= 0 (per Powell 1994, same as NLopt).
    //
    // Uses projected gradient descent on the linearised objective with
    // a penalty for linearised constraint violation.
    //
    // Reference: Powell 1994, Section 3 (trust-region subproblem).
    Eigen::Vector<Scalar, N> solve(
        const Eigen::Vector<Scalar, N>& obj_gradient,
        const Eigen::Matrix<Scalar, M, N>& constraint_gradients,
        const Eigen::Vector<Scalar, M>& constraint_offsets,
        int n_eq,
        Scalar rho,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const Eigen::Vector<Scalar, N>& x_current,
        Scalar parmu)
    {
        const int m = static_cast<int>(constraint_offsets.size());
        int m_expanded = m + n_eq;

        // Expand equality constraints into pairs of inequalities
        G_.topRows(m) = constraint_gradients;
        h_.head(m) = constraint_offsets;

        for(int j = 0; j < n_eq; ++j)
        {
            G_.row(m + j) = -constraint_gradients.row(j);
            h_[m + j] = -constraint_offsets[j];
        }

        d_.setZero();
        // The inner projected-gradient loop needs a non-zero penalty to
        // make progress on constraint violation when parmu is small or
        // zero. Powell's TRSTLP solves the LP exactly, no penalty needed,
        // but our projected-gradient surrogate (C2 in the static audit,
        // deferred to v0.3.x rewrite) requires a positive weight. Use
        // max(parmu, 1) so adapted parmu drives the inner loop once it
        // grows, while a sane default keeps iter-0 progress.
        Scalar penalty = std::max(parmu, Scalar(1));
        Scalar step_size = rho * Scalar(0.5);

        for(int iter = 0; iter < 100; ++iter)
        {
            grad_ = obj_gradient;

            for(int j = 0; j < m_expanded; ++j)
            {
                Scalar val = G_.row(j).dot(d_) + h_[j];
                if(val < Scalar(0))
                    grad_ -= penalty * G_.row(j).transpose();
            }

            if(grad_.norm() < Scalar(1e-15))
                break;

            d_new_ = d_ - step_size * grad_ / grad_.norm();
            d_new_ = clip_to_trust_and_bounds<Scalar, N>(d_new_, rho, x_current, lower, upper);

            auto merit = [&](const Eigen::Vector<Scalar, N>& dd) {
                Scalar obj = obj_gradient.dot(dd);
                Scalar viol = Scalar(0);
                for(int j = 0; j < m_expanded; ++j)
                {
                    Scalar val = G_.row(j).dot(dd) + h_[j];
                    viol += std::max(Scalar(0), -val);
                }
                return obj + penalty * viol;
            };

            if(merit(d_new_) < merit(d_))
            {
                d_ = d_new_;
                step_size = std::min(step_size * Scalar(1.1), rho);
            }
            else
            {
                step_size *= Scalar(0.5);
                if(step_size < rho * Scalar(1e-10))
                    break;
            }
        }

        return d_;
    }

private:
    int n_{0};
    int m_expanded_{0};
    Eigen::Matrix<Scalar, Mexp, N> G_;
    Eigen::Vector<Scalar, Mexp> h_;
    Eigen::Vector<Scalar, N> d_;
    Eigen::Vector<Scalar, N> grad_;
    Eigen::Vector<Scalar, N> d_new_;
};

// Backward-compatible free function wrapper.
//
// Reference: Powell 1994, Section 3 (trust-region subproblem).
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
Eigen::Vector<Scalar, N> solve_linear_subproblem(
    const Eigen::Vector<Scalar, N>& obj_gradient,
    const Eigen::Matrix<Scalar, M, N>& constraint_gradients,
    const Eigen::Vector<Scalar, M>& constraint_offsets,
    int n_eq,
    Scalar rho,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const Eigen::Vector<Scalar, N>& x_current,
    Scalar parmu = Scalar(10))
{
    const int n = static_cast<int>(obj_gradient.size());
    const int m = static_cast<int>(constraint_offsets.size());
    cobyla_trust_region_solver<Scalar, N, M> solver(n, m, n_eq);
    return solver.solve(obj_gradient, constraint_gradients, constraint_offsets,
                        n_eq, rho, lower, upper, x_current, parmu);
}

// Update trust radius following Powell's COBYLA schedule.
//
// If actual/predicted > 0.7: keep rho (good model accuracy).
// If actual/predicted < 0.1: halve rho (poor model accuracy).
// If rho reaches rho_end: signal convergence by returning rho_end.
//
// Returns the new trust radius.
//
// Reference: Powell 1994, Section 4 (trust radius management).
template <typename Scalar = double>
Scalar compute_rho_update(
    Scalar rho,
    Scalar rho_end,
    Scalar actual_reduction,
    Scalar predicted_reduction)
{
    if(std::abs(predicted_reduction) < Scalar(1e-30))
        return std::max(rho * Scalar(0.5), rho_end);

    Scalar ratio = actual_reduction / predicted_reduction;

    if(ratio > Scalar(0.7))
        return rho;

    if(ratio < Scalar(0.1))
        return std::max(rho * Scalar(0.5), rho_end);

    return rho;
}

}

#endif
