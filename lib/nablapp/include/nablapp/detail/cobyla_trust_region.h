#ifndef HPP_GUARD_NABLAPP_DETAIL_COBYLA_TRUST_REGION_H
#define HPP_GUARD_NABLAPP_DETAIL_COBYLA_TRUST_REGION_H

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

#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

// Clip a displacement vector d to the intersection of the trust ball
// ||d|| <= rho and box bounds lower <= x + d <= upper.
inline Eigen::VectorXd clip_to_trust_and_bounds(
    const Eigen::VectorXd& d,
    double rho,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper)
{
    const int n = static_cast<int>(d.size());
    Eigen::VectorXd result = d;

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
inline double linearised_violation(
    const Eigen::MatrixXd& constraint_gradients,
    const Eigen::VectorXd& constraint_offsets,
    int n_eq,
    const Eigen::VectorXd& d)
{
    const int m = static_cast<int>(constraint_offsets.size());
    double viol = 0.0;

    for(int j = 0; j < m; ++j)
    {
        double val = constraint_gradients.row(j).dot(d) + constraint_offsets[j];
        if(j < n_eq)
            viol += std::abs(val);
        else
            viol += std::max(0.0, -val);
    }

    return viol;
}

// Solve the COBYLA trust-region subproblem.
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
// a penalty for linearised constraint violation. The problem is small
// (n dimensions), so 100 iterations with adaptive step size suffices.
//
// Reference: Powell 1994, Section 3 (trust-region subproblem).
inline Eigen::VectorXd solve_linear_subproblem(
    const Eigen::VectorXd& obj_gradient,
    const Eigen::MatrixXd& constraint_gradients,
    const Eigen::VectorXd& constraint_offsets,
    int n_eq,
    double rho,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    const Eigen::VectorXd& x_current)
{
    const int n = static_cast<int>(obj_gradient.size());
    const int m = static_cast<int>(constraint_offsets.size());

    // Expand equality constraints into pairs of inequalities
    int m_expanded = m + n_eq;
    Eigen::MatrixXd G(m_expanded, n);
    Eigen::VectorXd h(m_expanded);

    // Original constraints
    G.topRows(m) = constraint_gradients;
    h.head(m) = constraint_offsets;

    // Negated equality constraints
    for(int j = 0; j < n_eq; ++j)
    {
        G.row(m + j) = -constraint_gradients.row(j);
        h[m + j] = -constraint_offsets[j];
    }

    Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
    double penalty = 10.0;
    double step_size = rho * 0.5;

    for(int iter = 0; iter < 100; ++iter)
    {
        // Gradient of penalised objective: g_obj + penalty * sum(violation gradients)
        Eigen::VectorXd grad = obj_gradient;

        for(int j = 0; j < m_expanded; ++j)
        {
            double val = G.row(j).dot(d) + h[j];
            if(val < 0.0)
                grad -= penalty * G.row(j).transpose();
        }

        // Steepest descent step
        if(grad.norm() < 1e-15)
            break;

        Eigen::VectorXd d_new = d - step_size * grad / grad.norm();
        d_new = clip_to_trust_and_bounds(d_new, rho, x_current, lower, upper);

        // Evaluate merit: objective + penalty * violation
        auto merit = [&](const Eigen::VectorXd& dd) {
            double obj = obj_gradient.dot(dd);
            double viol = 0.0;
            for(int j = 0; j < m_expanded; ++j)
            {
                double val = G.row(j).dot(dd) + h[j];
                viol += std::max(0.0, -val);
            }
            return obj + penalty * viol;
        };

        if(merit(d_new) < merit(d))
        {
            d = d_new;
            step_size = std::min(step_size * 1.1, rho);
        }
        else
        {
            step_size *= 0.5;
            if(step_size < rho * 1e-10)
                break;
        }
    }

    return d;
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
inline double compute_rho_update(
    double rho,
    double rho_end,
    double actual_reduction,
    double predicted_reduction)
{
    if(std::abs(predicted_reduction) < 1e-30)
        return std::max(rho * 0.5, rho_end);

    double ratio = actual_reduction / predicted_reduction;

    if(ratio > 0.7)
        return rho;

    if(ratio < 0.1)
        return std::max(rho * 0.5, rho_end);

    return rho;
}

}

#endif
