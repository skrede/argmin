#ifndef HPP_GUARD_ARGMIN_DETAIL_FILTER_RESTORATION_H
#define HPP_GUARD_ARGMIN_DETAIL_FILTER_RESTORATION_H

// Feasibility restoration strategies for filter-based SQP.
//
// When the filter rejects all trial step sizes, the iterate is stuck:
// no alpha produces an acceptable (f, h) pair. Restoration aims to
// reduce constraint violation h(x) to re-enter the filter-acceptable
// region.
//
// Three strategies:
//   l1_penalty    — steepest descent on sigma * h(x), cheap and often
//                   sufficient for mild infeasibility.
//   feasibility_qp — solve min 0.5 ||p||^2 s.t. linearized feasibility
//                    via the policy's own QP solver. More expensive but
//                    handles severe infeasibility.
//   hybrid        — try l1_penalty first, fall back to feasibility_qp
//                   on stall. Default. Matches the Knitro hybrid pattern.
//
// Reference: Wachter & Biegler 2006, Section 3 (feasibility restoration);
//            Fletcher, Leyffer & Toint 2002 (filter-SQP convergence);
//            N&W Section 15.5 (filter methods overview).

#include "argmin/detail/lagrangian.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace argmin::detail
{

enum class restoration_strategy : std::uint8_t
{
    l1_penalty,
    feasibility_qp,
    hybrid
};

template <typename Scalar, int N>
struct restoration_result
{
    Eigen::Vector<Scalar, N> x;
    Scalar constraint_violation{};
    bool success{false};
    std::uint16_t iterations_used{0};
};

// L1 penalty restoration: steepest descent on h(x) with backtracking.
//
// Takes steepest-descent steps on the L1 constraint violation
// h(x) = ||c_eq||_1 + ||max(0, -c_ineq)||_1 using the Jacobian
// transpose to form the descent direction. Includes stall detection
// to allow fallback to feasibility QP in hybrid mode.
//
// Reference: Wachter & Biegler 2006, Section 3.
template <typename Scalar, int N, typename Problem>
restoration_result<Scalar, N> restore_l1(
    const Problem& problem,
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Vector<Scalar, N>& /*gradient*/,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Scalar /*sigma_restore*/ = Scalar(1e4),
    std::uint16_t max_steps = 10,
    Scalar stall_ratio = Scalar(0.99),
    std::uint16_t stall_window = 5)
{
    const auto n = x0.size();
    Eigen::Vector<Scalar, N> x = x0;

    // Ring buffer for stall detection
    std::vector<Scalar> h_history;
    h_history.reserve(max_steps);

    for(std::uint16_t step = 0; step < max_steps; ++step)
    {
        // Evaluate constraints and Jacobian at current x
        Eigen::VectorX<Scalar> c_eq, c_ineq;
        problem.eval_constraints(x, c_eq, c_ineq);

        Scalar h_k = constraint_violation(c_eq, c_ineq);

        if(h_k < Scalar(1e-12))
        {
            return {x, h_k, true, step};
        }

        h_history.push_back(h_k);

        // Stall detection: check if recent progress is too slow
        if(h_history.size() >= stall_window)
        {
            auto oldest_idx = h_history.size() - stall_window;
            if(h_history.back() > stall_ratio * h_history[oldest_idx])
            {
                return {x, h_k, false, step};
            }
        }

        // Compute descent direction: p = -J^T * sign_vector
        // where sign_vector encodes the L1 subgradient
        Eigen::MatrixX<Scalar> J_eq, J_ineq;
        problem.eval_constraint_jacobians(x, J_eq, J_ineq);

        Eigen::Vector<Scalar, N> p = Eigen::Vector<Scalar, N>::Zero(n);

        // Equality constraint contribution: sign(c_eq_i)
        if(c_eq.size() > 0)
        {
            Eigen::VectorX<Scalar> sign_eq(c_eq.size());
            for(Eigen::Index i = 0; i < c_eq.size(); ++i)
                sign_eq[i] = (c_eq[i] > Scalar(0)) ? Scalar(1) : Scalar(-1);
            p -= J_eq.transpose() * sign_eq;
        }

        // Inequality constraint contribution: only violated (c_ineq < 0)
        if(c_ineq.size() > 0)
        {
            Eigen::VectorX<Scalar> sign_ineq = Eigen::VectorX<Scalar>::Zero(c_ineq.size());
            for(Eigen::Index i = 0; i < c_ineq.size(); ++i)
            {
                if(c_ineq[i] < Scalar(0))
                    sign_ineq[i] = Scalar(-1);
            }
            p -= J_ineq.transpose() * sign_ineq;
        }

        Scalar p_norm = p.norm();
        if(p_norm < Scalar(1e-15))
        {
            return {x, h_k, false, step};
        }

        // Backtracking line search on constraint violation
        Scalar alpha = Scalar(1);
        constexpr Scalar alpha_min = Scalar(1e-10);

        while(alpha > alpha_min)
        {
            Eigen::Vector<Scalar, N> x_trial = (x + alpha * p).cwiseMax(lower).cwiseMin(upper);

            Eigen::VectorX<Scalar> c_eq_trial, c_ineq_trial;
            problem.eval_constraints(x_trial, c_eq_trial, c_ineq_trial);
            Scalar h_trial = constraint_violation(c_eq_trial, c_ineq_trial);

            if(h_trial < h_k)
            {
                x = x_trial;
                break;
            }
            alpha *= Scalar(0.5);
        }

        if(alpha <= alpha_min)
        {
            return {x, h_k, false, step};
        }
    }

    // Final violation evaluation
    Eigen::VectorX<Scalar> c_eq_final, c_ineq_final;
    problem.eval_constraints(x, c_eq_final, c_ineq_final);
    Scalar h_final = constraint_violation(c_eq_final, c_ineq_final);

    return {x, h_final, h_final < Scalar(1e-12), max_steps};
}

// Feasibility QP restoration: minimize constraint violation via the
// policy's own QP solver.
//
// Each step solves min 0.5 ||p||^2 s.t. J_eq*p + c_eq = 0,
// J_ineq*p + c_ineq >= 0 using an identity Hessian and zero gradient.
// This finds the minimum-norm step that linearizes away the constraint
// violation.
//
// Reference: Wachter & Biegler 2006, Section 3;
//            N&W Section 15.5 (restoration phase).
template <typename Scalar, int N, typename Problem, typename QpSolver>
restoration_result<Scalar, N> restore_feasibility_qp(
    const Problem& problem,
    QpSolver& qp_solver,
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    std::uint16_t max_steps = 10)
{
    const auto n = x0.size();
    Eigen::Vector<Scalar, N> x = x0;

    for(std::uint16_t step = 0; step < max_steps; ++step)
    {
        Eigen::VectorX<Scalar> c_eq, c_ineq;
        problem.eval_constraints(x, c_eq, c_ineq);

        Scalar h_k = constraint_violation(c_eq, c_ineq);

        if(h_k < Scalar(1e-12))
        {
            return {x, h_k, true, step};
        }

        // Compute constraint Jacobians
        Eigen::MatrixX<Scalar> J_eq, J_ineq;
        problem.eval_constraint_jacobians(x, J_eq, J_ineq);

        // Solve feasibility QP: min 0.5 p^T I p s.t. linearized constraints
        Eigen::MatrixX<Scalar> H = Eigen::MatrixX<Scalar>::Identity(n, n);
        Eigen::VectorX<Scalar> g = Eigen::VectorX<Scalar>::Zero(n);

        // Bounds on the step: lower - x <= p <= upper - x
        Eigen::VectorX<Scalar> p_lower = (lower - x).eval();
        Eigen::VectorX<Scalar> p_upper = (upper - x).eval();

        auto qp_result = qp_solver.solve(H, g, J_eq, -c_eq, J_ineq, -c_ineq,
                                         p_lower, p_upper);

        Eigen::Vector<Scalar, N> p = qp_result.x;
        Scalar p_norm = p.norm();

        if(p_norm < Scalar(1e-15))
        {
            return {x, h_k, false, step};
        }

        // Backtracking on constraint violation
        Scalar alpha = Scalar(1);
        constexpr Scalar alpha_min = Scalar(1e-10);

        while(alpha > alpha_min)
        {
            Eigen::Vector<Scalar, N> x_trial = (x + alpha * p).cwiseMax(lower).cwiseMin(upper);

            Eigen::VectorX<Scalar> c_eq_trial, c_ineq_trial;
            problem.eval_constraints(x_trial, c_eq_trial, c_ineq_trial);
            Scalar h_trial = constraint_violation(c_eq_trial, c_ineq_trial);

            if(h_trial < h_k)
            {
                x = x_trial;
                break;
            }
            alpha *= Scalar(0.5);
        }

        if(alpha <= alpha_min)
        {
            return {x, h_k, false, step};
        }
    }

    Eigen::VectorX<Scalar> c_eq_final, c_ineq_final;
    problem.eval_constraints(x, c_eq_final, c_ineq_final);
    Scalar h_final = constraint_violation(c_eq_final, c_ineq_final);

    return {x, h_final, h_final < Scalar(1e-12), max_steps};
}

}

#endif
