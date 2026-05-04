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

// Cold-start calibration of the L1 merit penalty parameter sigma.
//
// Combines two iter-0 floors so that the first SQP step lands on a
// merit function whose violation term genuinely dominates the
// objective term:
//
//   1. lambda-floor: sigma >= max_i |lambda_i| + delta. Same shape as
//      update_penalty's monotone rule (N&W eq. 18.36) but applied
//      unconditionally at iter-0 with delta = 1.0 instead of the
//      monotone-update delta = 1e-4.
//   2. K-factor magnitude floor:
//      sigma >= K * max(1, |f_0| / (||c_0||_1 + eps)).
//      A problem-scale floor that guards against pathologically
//      small lambda at iter-0 on objective-dominated initial points
//      (e.g. HS071 from x_0 = (1, 5, 5, 1) where ||lambda||_inf is
//      O(1) but |f_0| / ||c_0||_1 is O(5)).
//
// Returns sigma_calibrated >= sigma_in. Caller is responsible for
// gating the call to iter-0 only; this function does not check
// state.iteration.
//
// argmin variant: combines the N&W lambda-floor with a magnitude
// floor; rationale: HS071 cold-start with x_0 = (1, 5, 5, 1) admits
// a near-feasible iter-0 step under raw N&W eq. 18.36 (lambda is
// small) and parks the iterate strongly infeasible.
//
// Reference: N&W 2e Section 18.3 / eq. 18.36 (lambda-floor for
//            descent on the L1 merit);
//            Kraft 1988 DFVLR-FB 88-28 Section 2.2.6 (sigma update
//            companion); the K-factor problem-scale floor is the
//            magnitude-aware companion that closes objective-
//            dominated cold-starts where ||lambda||_inf is small
//            but |f_0| / ||c_0||_1 is large.
template <typename Scalar, int M = argmin::dynamic_dimension>
Scalar calibrate_initial_penalty(Scalar sigma_in,
                                 const Eigen::Vector<Scalar, M>& lambda_qp,
                                 Scalar f_0,
                                 Scalar c_0_l1,
                                 Scalar K_factor = Scalar(10),
                                 Scalar delta = Scalar(1))
{
    Scalar sigma = sigma_in;
    if(lambda_qp.size() > 0)
    {
        const Scalar lambda_floor = lambda_qp.cwiseAbs().maxCoeff() + delta;
        sigma = std::max(sigma, lambda_floor);
    }
    const Scalar denom = c_0_l1 + Scalar(1e-12);
    const Scalar magnitude_floor =
        K_factor * std::max(Scalar(1), std::abs(f_0) / denom);
    return std::max(sigma, magnitude_floor);
}

}

#endif
