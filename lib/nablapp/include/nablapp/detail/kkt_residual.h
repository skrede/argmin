#ifndef HPP_GUARD_NABLAPP_DETAIL_KKT_RESIDUAL_H
#define HPP_GUARD_NABLAPP_DETAIL_KKT_RESIDUAL_H

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

#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

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
// Sign convention: nablapp uses c_ineq(x) >= 0 feasible, matching
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
    Scalar complementarity = Scalar(0);
    for(Eigen::Index i = 0; i < c_ineq.size(); ++i)
        complementarity = std::max(complementarity,
            std::min(std::abs(mu_ineq[i]), std::abs(c_ineq[i])));

    return std::max({stationarity, primal_eq, primal_ineq, dual_feas, complementarity});
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

}

#endif
