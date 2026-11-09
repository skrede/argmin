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
#include "argmin/detail/filter_restoration_workspace.h"

#include "argmin/types.h"

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

template <scalar S>
struct filter_restoration_constants;

template <>
struct filter_restoration_constants<double>
{
    static constexpr double infeasibility_floor = 1e-12;
    static constexpr double zero_guard = 1e-15;
    static constexpr double alpha_min = 1e-10;
};

template <>
struct filter_restoration_constants<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float infeasibility_floor = 5e-6f;  // rule
    static constexpr float zero_guard = 2e-7f;           // rule
    static constexpr float alpha_min = 4e-5f;            // rule
};

template <typename Scalar>
bool l1_stalled(std::vector<Scalar>& h_history, Scalar h_k, Scalar stall_ratio, std::uint16_t stall_window)
{
    h_history.push_back(h_k);
    if(h_history.size() < stall_window)
        return false;
    return h_history.back() > stall_ratio * h_history[h_history.size() - stall_window];
}

// The L1 subgradient direction p = -J_eq^T sign(c_eq) - J_ineq^T s, where s_i = -1 on a violated inequality row.
template <typename Scalar, int N>
void l1_descent(filter_restoration_workspace<Scalar, N>& ws, Eigen::Vector<Scalar, N>& p)
{
    p.setZero();
    if(ws.c_eq.size() > 0)
    {
        ws.sign_eq.resize(ws.c_eq.size());
        for(Eigen::Index i = 0; i < ws.c_eq.size(); ++i)
            ws.sign_eq[i] = (ws.c_eq[i] > Scalar(0)) ? Scalar(1) : Scalar(-1);
        ws.descent.noalias() = ws.J_eq.transpose() * ws.sign_eq;
        p -= ws.descent;
    }
    if(ws.c_ineq.size() > 0)
    {
        ws.sign_ineq.resize(ws.c_ineq.size());
        for(Eigen::Index i = 0; i < ws.c_ineq.size(); ++i)
            ws.sign_ineq[i] = (ws.c_ineq[i] < Scalar(0)) ? Scalar(-1) : Scalar(0);
        ws.descent.noalias() = ws.J_ineq.transpose() * ws.sign_ineq;
        p -= ws.descent;
    }
}

template <typename Scalar, int N, typename Problem>
bool l1_backtrack(const Problem& problem, filter_restoration_workspace<Scalar, N>& ws, Eigen::Vector<Scalar, N>& x,
                  const Eigen::Vector<Scalar, N>& p, Scalar h_k, const Eigen::Vector<Scalar, N>& lower, const Eigen::Vector<Scalar, N>& upper)
{
    for(Scalar alpha = Scalar(1); alpha > filter_restoration_constants<Scalar>::alpha_min; alpha *= Scalar(0.5))
    {
        Eigen::Vector<Scalar, N> x_trial = (x + alpha * p).cwiseMax(lower).cwiseMin(upper);
        problem.eval_constraints(x_trial, ws.c_eq, ws.c_ineq);
        if(constraint_violation(ws.c_eq, ws.c_ineq) < h_k)
        {
            x = x_trial;
            return true;
        }
    }
    return false;
}

// Steepest descent with backtracking on h(x) = ||c_eq||_1 + ||max(0, -c_ineq)||_1. Stalled progress returns
// unsuccessfully so the hybrid strategy can fall back to the feasibility QP.
//
// Reference: Wachter & Biegler 2006, Section 3.
template <typename Scalar, int N, typename Problem>
restoration_result<Scalar, N> restore_l1(
    const Problem& problem,
    filter_restoration_workspace<Scalar, N>& ws,
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Vector<Scalar, N>& /*gradient*/,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Scalar /*sigma_restore*/ = Scalar(1e4),
    std::uint16_t max_steps = 10,
    Scalar stall_ratio = Scalar(0.99),
    std::uint16_t stall_window = 5)
{
    constexpr Scalar h_floor = filter_restoration_constants<Scalar>::infeasibility_floor;
    Eigen::Vector<Scalar, N> x = x0;
    Eigen::Vector<Scalar, N> p = Eigen::Vector<Scalar, N>::Zero(x0.size());
    ws.h_history.clear();
    ws.h_history.reserve(max_steps);
    for(std::uint16_t step = 0; step < max_steps; ++step)
    {
        problem.eval_constraints(x, ws.c_eq, ws.c_ineq);
        const Scalar h_k = constraint_violation(ws.c_eq, ws.c_ineq);
        if(h_k < h_floor)
            return {x, h_k, true, step};
        if(l1_stalled(ws.h_history, h_k, stall_ratio, stall_window))
            return {x, h_k, false, step};
        problem.eval_constraint_jacobians(x, ws.J_eq, ws.J_ineq);
        l1_descent(ws, p);
        if(p.norm() < filter_restoration_constants<Scalar>::zero_guard || !l1_backtrack(problem, ws, x, p, h_k, lower, upper))
            return {x, h_k, false, step};
    }
    problem.eval_constraints(x, ws.c_eq, ws.c_ineq);
    const Scalar h_final = constraint_violation(ws.c_eq, ws.c_ineq);
    return {x, h_final, h_final < h_floor, max_steps};
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

        if(h_k < filter_restoration_constants<Scalar>::infeasibility_floor)
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

        if(p_norm < filter_restoration_constants<Scalar>::zero_guard)
        {
            return {x, h_k, false, step};
        }

        // Backtracking on constraint violation
        Scalar alpha = Scalar(1);
        constexpr Scalar alpha_min = filter_restoration_constants<Scalar>::alpha_min;

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

    return {x, h_final, h_final < filter_restoration_constants<Scalar>::infeasibility_floor, max_steps};
}

}

#endif
