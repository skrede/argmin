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
//   l1_penalty    — steepest descent on h(x), cheap and often
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
#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"
#include "argmin/detail/filter_restoration_workspace.h"

#include "argmin/types.h"

#include "argmin/options/qp_options.h"

#include <Eigen/Core>

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <type_traits>

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
bool backtrack_violation(const Problem& problem, filter_restoration_workspace<Scalar, N>& ws, Eigen::Vector<Scalar, N>& x,
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
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
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
        if(p.norm() < filter_restoration_constants<Scalar>::zero_guard || !backtrack_violation(problem, ws, x, p, h_k, lower, upper))
            return {x, h_k, false, step};
    }
    problem.eval_constraints(x, ws.c_eq, ws.c_ineq);
    const Scalar h_final = constraint_violation(ws.c_eq, ws.c_ineq);
    return {x, h_final, h_final < h_floor, max_steps};
}

// The QP min 0.5 ||p||^2 s.t. J_eq p = -c_eq, J_ineq p >= -c_ineq at the current linearization.
template <typename Scalar, int N>
void load_feasibility_qp(filter_restoration_workspace<Scalar, N>& ws)
{
    ws.A_eq = ws.J_eq;
    ws.A_ineq = ws.J_ineq;
    ws.b_eq = -ws.c_eq;
    ws.b_ineq = -ws.c_ineq;
}

template <typename Scalar, int N>
const Eigen::Vector<Scalar, N>& feasibility_step(kraft_lsq_qp_recovery_solver<Scalar, N>& qp_solver, qp_result<Scalar, N>& qp,
                                                 filter_restoration_workspace<Scalar, N>& ws, const Eigen::Vector<Scalar, N>& x,
                                                 const Eigen::Vector<Scalar, N>& lower, const Eigen::Vector<Scalar, N>& upper)
{
    ws.step_lower = lower - x;
    ws.step_upper = upper - x;
    qp_solver.solve_into(qp, ws.identity, ws.zero, ws.A_eq, ws.b_eq, ws.A_ineq, ws.b_ineq, ws.step_lower, ws.step_upper);
    return qp.x;
}

// The active-set QP takes no box here and starts from p = 0; the backtracking projects each trial point instead.
template <typename Scalar, int N, int M>
const Eigen::Vector<Scalar, N>& feasibility_step(active_set_qp_solver<Scalar, N, M>& qp_solver, typename active_set_qp_solver<Scalar, N, M>::result_type& qp,
                                                 filter_restoration_workspace<Scalar, N>& ws, const Eigen::Vector<Scalar, N>& /*x*/,
                                                 const Eigen::Vector<Scalar, N>& /*lower*/, const Eigen::Vector<Scalar, N>& /*upper*/)
{
    qp_solver.solve_into(ws.identity, ws.zero, ws.A_eq, ws.b_eq, ws.A_ineq, ws.b_ineq, ws.zero, qp_options<Scalar>{}, qp);
    return qp.x;
}

// Each step takes the minimum-norm linearized feasibility step from the policy's own QP solver (feasibility_step,
// found by argument-dependent lookup) and backtracks on the constraint violation.
//
// Reference: Wachter & Biegler 2006, Section 3;
//            N&W Section 15.5 (restoration phase).
template <typename Scalar, int N, typename Problem, typename QpSolver, typename QpResult>
restoration_result<Scalar, N> restore_feasibility_qp(
    const Problem& problem,
    filter_restoration_workspace<Scalar, N>& ws,
    QpSolver& qp_solver,
    QpResult& qp,
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    std::uint16_t max_steps = 10)
{
    constexpr Scalar h_floor = filter_restoration_constants<Scalar>::infeasibility_floor;
    Eigen::Vector<Scalar, N> x = x0;
    for(std::uint16_t step = 0; step < max_steps; ++step)
    {
        problem.eval_constraints(x, ws.c_eq, ws.c_ineq);
        const Scalar h_k = constraint_violation(ws.c_eq, ws.c_ineq);
        if(h_k < h_floor)
            return {x, h_k, true, step};
        problem.eval_constraint_jacobians(x, ws.J_eq, ws.J_ineq);
        load_feasibility_qp(ws);
        const Eigen::Vector<Scalar, N>& p = feasibility_step(qp_solver, qp, ws, x, lower, upper);
        if(p.norm() < filter_restoration_constants<Scalar>::zero_guard || !backtrack_violation(problem, ws, x, p, h_k, lower, upper))
            return {x, h_k, false, step};
    }
    problem.eval_constraints(x, ws.c_eq, ws.c_ineq);
    const Scalar h_final = constraint_violation(ws.c_eq, ws.c_ineq);
    return {x, h_final, h_final < h_floor, max_steps};
}

// The configured strategy on a filter policy's state at its current iterate. The hybrid strategy falls back to the
// feasibility QP when the L1 descent fails; an unsuccessful result carries no usable iterate.
template <typename Scalar, int N, typename State>
restoration_result<Scalar, N> restore_feasibility(restoration_strategy strategy, State& s, std::uint16_t max_steps)
{
    using problem_type = std::remove_cvref_t<decltype(*s.problem)>;
    const restoration_adapter<Scalar, N, problem_type> adapter{s.n_eq, s.n_ineq, *s.problem, s.restoration_ws.c_all, s.restoration_ws.J_all};
    if(strategy != restoration_strategy::feasibility_qp)
    {
        restoration_result<Scalar, N> result = restore_l1<Scalar, N>(adapter, s.restoration_ws, s.x, s.lower, s.upper, max_steps);
        if(result.success || strategy == restoration_strategy::l1_penalty)
            return result;
    }
    return restore_feasibility_qp<Scalar, N>(adapter, s.restoration_ws, s.qp_solver, s.restoration_qp, s.x, s.lower, s.upper, max_steps);
}

}

#endif
