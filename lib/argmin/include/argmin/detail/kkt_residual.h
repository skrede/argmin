#ifndef HPP_GUARD_ARGMIN_DETAIL_KKT_RESIDUAL_H
#define HPP_GUARD_ARGMIN_DETAIL_KKT_RESIDUAL_H

// KKT residual helpers.
//
// Computes the L-infinity first-order optimality error for gradient-aware
// optimization policies so convergence criteria have a single, consistent
// stationarity quantity to gate on. The value is zero iff all first-order
// KKT conditions hold.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Definition 12.1 (KKT conditions: stationarity + primal
//            feasibility + dual feasibility + complementarity);
//            eq. 12.34 (Lagrangian stationarity expression);
//            Section 16.7 (projected gradient optimality for
//            bound-constrained problems).

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace argmin::detail
{

// Legs 2-5 of the composite first-order optimality error (primal
// equality feasibility, primal inequality feasibility, dual
// feasibility, complementarity). Shared by the raw-stationarity and
// bound-aware composite overloads of kkt_residual below so the leg
// definitions are single-sourced.
template <typename Scalar, int Meq, int Mineq>
Scalar kkt_feasibility_legs(
    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>& mu_ineq,
    const Eigen::Ref<const Eigen::Vector<Scalar, Meq>>&   c_eq,
    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>& c_ineq)
{
    // Leg 2 - Primal equality feasibility: ||c_eq||_inf
    const Scalar primal_eq = c_eq.size() > 0
        ? c_eq.template lpNorm<Eigen::Infinity>()
        : Scalar(0);

    // Leg 3 - Primal inequality feasibility: ||max(-c_ineq, 0)||_inf
    Scalar primal_ineq = Scalar(0);
    for(Eigen::Index i = 0; i < c_ineq.size(); ++i)
        primal_ineq = std::max(primal_ineq, std::max(-c_ineq[i], Scalar(0)));

    // Leg 4 - Dual feasibility: ||max(-mu_ineq, 0)||_inf
    Scalar dual_feas = Scalar(0);
    for(Eigen::Index i = 0; i < mu_ineq.size(); ++i)
        dual_feas = std::max(dual_feas, std::max(-mu_ineq[i], Scalar(0)));

    // Leg 5 - Complementarity: max_i min(|mu_i|, |c_i|)
    // Bound the loop by the smaller of the two sizes so a caller passing
    // mismatched mu_ineq / c_ineq lengths (allowed by the Ref-based
    // signature) does not read past either vector. All in-tree call sites
    // currently pass matched n_ineq-sized vectors; the defense mirrors the
    // diagnostic harness in benchmarks/diagnostic_kkt_breakdown.cpp.
    Scalar complementarity = Scalar(0);
    const Eigen::Index m_comp = std::min(mu_ineq.size(), c_ineq.size());
    for(Eigen::Index i = 0; i < m_comp; ++i)
        complementarity = std::max(complementarity,
            std::min(std::abs(mu_ineq[i]), std::abs(c_ineq[i])));

    return std::max({primal_eq, primal_ineq, dual_feas, complementarity});
}

// Composite first-order optimality error E(x, lambda, mu) as the maximum
// over five KKT legs for the general constrained NLP
//
//     min f(x)  s.t.  c_eq(x) = 0,  c_ineq(x) >= 0
//
// with multipliers lambda_eq (free sign) and mu_ineq (>= 0 by the KKT
// dual-feasibility condition). Returns a single Scalar; zero iff KKT
// conditions hold at (x, lambda_eq, mu_ineq).
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e, Definition
//            12.1 (KKT conditions: stationarity + primal feasibility +
//            dual feasibility + complementarity); eq. 12.34 (Lagrangian
//            stationarity expression). L-infinity norm throughout
//            (dimension-independent; matches IPOPT / KNITRO convention).
//            Unscaled composition; IPOPT-style s_d / s_c scaling is a
//            refactor-milestone follow-up.
//
// Sign convention: argmin uses c_ineq(x) >= 0 feasible, matching
//                  detail/lagrangian.h:42-60 and formulation/concepts.h.
//                  Primal-inequality violation is c_ineq[i] < 0, captured
//                  as max(-c_ineq[i], 0). Dual feasibility violates when
//                  mu_ineq[i] < 0, captured as max(-mu_ineq[i], 0).
template <typename Scalar, int N, int Meq, int Mineq>
Scalar kkt_residual(const Eigen::Ref<const Eigen::Vector<Scalar, N>>&       grad_f,
                    const Eigen::Ref<const Eigen::Matrix<Scalar, Meq, N>>&  J_eq,
                    const Eigen::Ref<const Eigen::Matrix<Scalar, Mineq, N>>& J_ineq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Meq>>&     lambda_eq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>&   mu_ineq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Meq>>&     c_eq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>&   c_ineq)
{
    // Leg 1 - Stationarity: ||grad_f - J_eq^T lambda_eq - J_ineq^T mu_ineq||_inf
    Eigen::Vector<Scalar, N> grad_L = grad_f;
    if(J_eq.rows()   > 0) grad_L.noalias() -= J_eq.transpose()   * lambda_eq;
    if(J_ineq.rows() > 0) grad_L.noalias() -= J_ineq.transpose() * mu_ineq;
    const Scalar stationarity = grad_L.size() > 0
        ? grad_L.template lpNorm<Eigen::Infinity>()
        : Scalar(0);

    return std::max(stationarity,
                    kkt_feasibility_legs<Scalar, Meq, Mineq>(mu_ineq, c_eq, c_ineq));
}

// Bound-aware composite first-order optimality error.
//
// Identical to the raw-stationarity composite above except Leg 1 is
// the box-projected stationarity measure
//
//     ||P(x - grad_L, lower, upper) - x||_inf
//
// (the kkt_residual_bound expression applied to the Lagrangian
// gradient). At a solution with an active box bound the raw
// stationarity leg equals the unmodeled bound multiplier and is
// bounded away from zero, structurally preventing kkt-gated
// convergence; the projected leg is zero there. At interior points
// the projection is inactive and the projected leg collapses to the
// raw ||grad_L||_inf, so unbounded problems are numerically unchanged
// under this overload.
//
// Reference: Nocedal and Wright 2e Definition 12.1 (KKT conditions);
//            Section 16.7 (projected gradient optimality for
//            bound-constrained problems).
template <typename Scalar, int N, int Meq, int Mineq>
Scalar kkt_residual(const Eigen::Ref<const Eigen::Vector<Scalar, N>>&       grad_f,
                    const Eigen::Ref<const Eigen::Matrix<Scalar, Meq, N>>&  J_eq,
                    const Eigen::Ref<const Eigen::Matrix<Scalar, Mineq, N>>& J_ineq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Meq>>&     lambda_eq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>&   mu_ineq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Meq>>&     c_eq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, Mineq>>&   c_ineq,
                    const Eigen::Ref<const Eigen::Vector<Scalar, N>>&       x,
                    const Eigen::Ref<const Eigen::Vector<Scalar, N>>&       lower,
                    const Eigen::Ref<const Eigen::Vector<Scalar, N>>&       upper)
{
    // Leg 1 - Box-projected stationarity: ||P(x - grad_L, l, u) - x||_inf.
    Eigen::Vector<Scalar, N> grad_L = grad_f;
    if(J_eq.rows()   > 0) grad_L.noalias() -= J_eq.transpose()   * lambda_eq;
    if(J_ineq.rows() > 0) grad_L.noalias() -= J_ineq.transpose() * mu_ineq;
    const Scalar stationarity = grad_L.size() > 0
        ? ((x - grad_L).cwiseMax(lower).cwiseMin(upper) - x)
              .template lpNorm<Eigen::Infinity>()
        : Scalar(0);

    return std::max(stationarity,
                    kkt_feasibility_legs<Scalar, Meq, Mineq>(mu_ineq, c_eq, c_ineq));
}

// KKT residual for bound-constrained problems.
//
// Reduces to ||P(x - grad_f, lower, upper) - x||_inf where P is
// component-wise box projection. This is zero iff x satisfies the
// bound-constrained first-order conditions (at an interior point it
// collapses to ||grad_f||_inf; at an active bound it captures the
// projected gradient).
//
// Reference: N&W 2e Section 16.7 (projected gradient optimality).
template <typename Scalar, int N>
Scalar kkt_residual_bound(const Eigen::Vector<Scalar, N>& x,
                          const Eigen::Vector<Scalar, N>& grad_f,
                          const Eigen::Vector<Scalar, N>& lower,
                          const Eigen::Vector<Scalar, N>& upper)
{
    Eigen::Vector<Scalar, N> projected_step =
        (x - grad_f).cwiseMax(lower).cwiseMin(upper) - x;
    return projected_step.size() > 0
        ? projected_step.template lpNorm<Eigen::Infinity>()
        : Scalar(0);
}

// Box-projected stationarity measure in displaced-bound form.
//
// Equals kkt_residual_bound(z, grad_L, lower, upper) with the caller
// supplying the DISPLACED box (lower - z, upper - z): the projected
// step is P(z - grad_L, lower, upper) - z = clamp(-grad_L,
// lower - z, upper - z). The TR-SQP policies assemble the displaced
// joint box for Steihaug-CG anyway, so this form avoids materializing
// the absolute joint bounds a second time. At interior points the
// clamp is inactive and the measure collapses to ||grad_L||_inf.
//
// Reference: N&W 2e Section 16.7 (projected gradient optimality).
template <typename Scalar>
Scalar projected_stationarity_displaced(
    const Eigen::Ref<const Eigen::Vector<Scalar, Eigen::Dynamic>>& grad_L,
    const Eigen::Ref<const Eigen::Vector<Scalar, Eigen::Dynamic>>& lower_disp,
    const Eigen::Ref<const Eigen::Vector<Scalar, Eigen::Dynamic>>& upper_disp)
{
    if(grad_L.size() == 0)
        return Scalar(0);
    return (-grad_L).cwiseMax(lower_disp).cwiseMin(upper_disp)
        .template lpNorm<Eigen::Infinity>();
}

}

#endif
