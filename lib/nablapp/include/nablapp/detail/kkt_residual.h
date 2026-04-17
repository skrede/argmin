#ifndef HPP_GUARD_NABLAPP_DETAIL_KKT_RESIDUAL_H
#define HPP_GUARD_NABLAPP_DETAIL_KKT_RESIDUAL_H

// KKT residual helpers.
//
// Computes the L-infinity KKT error for gradient-aware optimization
// policies so convergence criteria have a single, consistent stationarity
// quantity to gate on. The value is zero iff all first-order KKT
// conditions hold.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34 (Lagrangian stationarity);
//            N&W 2e Section 12.1 (KKT conditions: stationarity +
//            feasibility + complementarity);
//            N&W 2e Section 16.7 (projected gradient optimality for
//            bound-constrained problems).

#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// KKT residual for general constrained optimization.
//
// Computes the L-infinity KKT error as the maximum of:
//   1. Stationarity:    ||grad_f - J_eq^T lambda_eq - J_ineq^T mu_ineq||_inf
//   2. Complementarity: max_i min(|mu_ineq_i|, |c_ineq_i|)
//
// Primal feasibility (||c_eq||_inf, ||max(0, -c_ineq)||_inf) is intentionally
// not folded in here because the convergence criteria already gate on
// primal feasibility through feasibility_gate on the objective tolerance
// criteria; keeping the two legs separate preserves the "gate" / "optimum"
// split that basic_solver relies on.
//
// L-infinity is chosen over L1 for dimension-independence and consistency
// with the convergence analysis used throughout N&W 2e Chapters 12 and 18.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34; N&W 2e Section 12.1.
template <typename Scalar, int N, int Meq, int Mineq>
Scalar kkt_residual(const Eigen::Vector<Scalar, N>& grad_f,
                    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
                    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq,
                    const Eigen::Vector<Scalar, Meq>& lambda_eq,
                    const Eigen::Vector<Scalar, Mineq>& mu_ineq,
                    const Eigen::Vector<Scalar, Mineq>& c_ineq)
{
    Eigen::Vector<Scalar, N> grad_L = grad_f;
    if(J_eq.rows() > 0)
        grad_L -= J_eq.transpose() * lambda_eq;
    if(J_ineq.rows() > 0)
        grad_L -= J_ineq.transpose() * mu_ineq;

    Scalar stationarity = grad_L.size() > 0
        ? grad_L.template lpNorm<Eigen::Infinity>()
        : Scalar(0);

    Scalar complementarity = Scalar(0);
    for(Eigen::Index i = 0; i < c_ineq.size(); ++i)
    {
        using std::abs;
        using std::min;
        Scalar local = min(abs(mu_ineq[i]), abs(c_ineq[i]));
        complementarity = std::max(complementarity, local);
    }

    return std::max(stationarity, complementarity);
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
