#ifndef HPP_GUARD_ARGMIN_DETAIL_MERIT_FUNCTION_H
#define HPP_GUARD_ARGMIN_DETAIL_MERIT_FUNCTION_H

// L1 exact penalty (merit) function for SQP globalization.
//
// The l1 merit function is:
//   phi_1(x; sigma) = f(x) + sigma * (||c_eq||_1 + ||[c_ineq]^-||_1)
//
// where sigma >= 0 is the penalty weight. Larger sigma penalizes
// constraint violations more heavily.
//
// Reference: N&W Section 18.5, eq. 15.24 (l1 penalty definition);
//            N&W Section 18.6, eq. 18.36 (directional derivative);
//            N&W eq. 18.36 (penalty update rule).

#include "argmin/detail/lagrangian.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace argmin::detail
{

// L1 merit function value.
//
// phi_1(x; sigma) = f + sigma * violation(c_eq, c_ineq).
//
// Reference: N&W eq. 15.24, Section 18.5.
template <typename Scalar, int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
Scalar l1_merit(Scalar f,
                const Eigen::Vector<Scalar, Meq>& c_eq,
                const Eigen::Vector<Scalar, Mineq>& c_ineq,
                Scalar sigma)
{
    return f + sigma * constraint_violation(c_eq, c_ineq);
}

// Directional derivative of L1 merit along SQP direction p.
//
// When c(x) != 0 (i.e. not at a feasible point):
//   D phi_1(x; sigma)[p] = grad_f^T p - sigma * ||c(x)||_1
//
// This is a first-order approximation valid for the non-smooth l1 merit.
// The term -sigma * ||c||_1 arises because the QP subproblem linearizes
// constraints, driving them toward feasibility.
//
// Reference: N&W eq. 18.36 (directional derivative condition).
template <typename Scalar, int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
Scalar l1_merit_directional_derivative(
    Scalar grad_f_dot_p,
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    Scalar sigma)
{
    return grad_f_dot_p - sigma * constraint_violation(c_eq, c_ineq);
}

// Update penalty parameter sigma.
//
// Ensure sigma >= max_i |lambda_i| + delta so that the SQP direction
// is a descent direction for the l1 merit function. Never decrease
// sigma (monotone update).
//
// Reference: N&W eq. 18.36 (sufficient penalty for descent).
template <typename Scalar, int M = argmin::dynamic_dimension>
Scalar update_penalty(Scalar sigma,
                      const Eigen::Vector<Scalar, M>& lambda,
                      Scalar delta = Scalar(1e-4))
{
    if(lambda.size() == 0)
        return sigma;
    Scalar required = lambda.cwiseAbs().maxCoeff() + delta;
    return std::max(sigma, required);
}

// Cold-start initial penalty calibration.
//
// After the first QP solve at iter 0, sigma must be at least as large
// as the multiplier scale to ensure the SQP direction is a descent
// direction for the L1 merit (N&W eq. 18.36). The default cold-start
// value (1.0) underestimates the required penalty on problems with
// large multipliers (e.g. HS071), causing the line search to reject
// the iter-0 step and the policy to park infeasible.
//
// calibrate_initial_penalty(sigma_in, lambda_qp, safety) returns
//   max(sigma_in, ||lambda_qp||_inf + safety),
// mirroring update_penalty's monotone-up form but as a one-shot
// initialisation rather than a per-step update.
//
// Adopted from: NLopt slsqp.c implicit cold-start from QP lambda
//               (slsqp.c on the iter-0 path uses lambda from the
//                first QP solve to set the penalty scale before
//                any merit evaluation).
// Reference: N&W 2e eq. 18.36 (sigma sufficient for L1-merit descent);
//            Powell 1978 ("A fast algorithm for nonlinearly
//            constrained optimization calculations") Section 6
//            (cold-start safety_factor = 1 recommendation);
//            Kraft 1988 DFVLR-FB 88-28 §2.2.6 (sigma update rule);
//            PITFALLS §B remedy 1 (single-line scalar cold-start).
//
// argmin variant: scalar sigma calibration without per-constraint
//                 mu_j smoothing; NLopt slsqp.c:1988-1994 maintains
//                 a per-constraint mu_j vector; rationale: scalar
//                 fix delivers HS071 closure without the regression
//                 risk of per-constraint smoothing on already-stable
//                 hs026 / hs039 trajectories. Per-constraint mu_j
//                 is an empirically-gated future extension.
template <typename Scalar, int M = argmin::dynamic_dimension, int MaxM = M>
Scalar calibrate_initial_penalty(Scalar sigma_in,
                                 const Eigen::Matrix<Scalar, M, 1, 0, MaxM, 1>& lambda_qp,
                                 Scalar safety = Scalar(1.0))
{
    if(lambda_qp.size() == 0)
        return sigma_in;
    Scalar required = lambda_qp.cwiseAbs().maxCoeff() + safety;
    return std::max(sigma_in, required);
}

// h4-weighted directional derivative of the L1 merit.
//
// Adopted from: argmin/solver/kraft_slsqp_policy.h:541-542 (in-tree
//               precedent — h4-weighted directional derivative).
// Reference: Kraft 1988 DFVLR-FB 88-28 §3.4 (h4 = 1 - relaxation_factor);
//            N&W 2e Section 18.3 (L1 merit directional derivative).
//
// argmin variant: extends l1_merit_directional_derivative with a
//                 multiplicative h4 factor on the violation term.
//                 nw_sqp / filter_slsqp / filter_nw_sqp lineages call
//                 with h4 = 1 (bit-identical to the existing helper).
template <typename Scalar,
          int Meq = argmin::dynamic_dimension,
          int Mineq = argmin::dynamic_dimension>
ARGMIN_FORCE_INLINE Scalar l1_merit_dphi_h4(
    Scalar grad_f_dot_p,
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    Scalar sigma,
    Scalar h4 = Scalar(1))
{
    const Scalar viol = constraint_violation(c_eq, c_ineq);
    return grad_f_dot_p - sigma * h4 * viol;
}

// Cold-bump sigma so that the L1-merit slope becomes strictly negative.
//
// Adopted from: argmin/solver/kraft_slsqp_policy.h:544-555 (in-tree
//               precedent — sigma cold-bump for descent guarantee);
//               argmin/solver/nw_sqp_policy.h:447-463 (parallel pattern).
// Reference: Kraft 1988 DFVLR-FB 88-28 §2.2.6;
//            N&W 2e Section 18.3 (penalty parameter cold-bump).
//
// argmin variant: scalar form with sigma_max cap. Idempotent;
//                 monotone-up: sigma never decreases.
//
// Result of a sigma cold-bump. sigma is the (possibly capped) penalty to
// use; saturated is true iff the descent-restoring bump exceeded sigma_max
// and was clamped, meaning the returned sigma is provably NOT large enough
// to make p a descent direction. Callers act on saturated by short-
// circuiting the doomed line search into their recovery ladder rather than
// backtracking against an unsatisfiable Armijo test. A plain aggregate:
// allocation-free, register-returnable, exception/RTTI-free.
template <typename Scalar>
struct sigma_bump_result
{
    Scalar sigma;
    bool saturated;
};

template <typename Scalar>
ARGMIN_FORCE_INLINE sigma_bump_result<Scalar> bump_sigma_for_descent(
    Scalar sigma_in,
    Scalar grad_f_dot_p,
    Scalar violation,
    Scalar h4 = Scalar(1),
    Scalar sigma_max = Scalar(1e10))
{
    const Scalar dphi = grad_f_dot_p - sigma_in * h4 * violation;
    if(dphi >= Scalar(0) && violation > Scalar(0) && h4 > Scalar(0))
    {
        using std::abs;
        const Scalar bumped = std::max(
            sigma_in,
            abs(grad_f_dot_p) / (violation * h4) + Scalar(1));
        // Saturation is decided on the uncapped bump, BEFORE the clamp:
        // once bumped exceeds sigma_max the clamped sigma leaves the merit
        // slope non-negative, so the caller must recover rather than search.
        const bool saturated = bumped > sigma_max;
        return {std::min(bumped, sigma_max), saturated};
    }
    return {sigma_in, false};
}

}

#endif
