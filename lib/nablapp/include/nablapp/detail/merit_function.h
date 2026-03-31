#ifndef HPP_GUARD_NABLAPP_DETAIL_MERIT_FUNCTION_H
#define HPP_GUARD_NABLAPP_DETAIL_MERIT_FUNCTION_H

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

#include "nablapp/detail/lagrangian.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// L1 merit function value.
//
// phi_1(x; sigma) = f + sigma * violation(c_eq, c_ineq).
//
// Reference: N&W eq. 15.24, Section 18.5.
template <typename Scalar>
Scalar l1_merit(Scalar f,
                const Eigen::VectorX<Scalar>& c_eq,
                const Eigen::VectorX<Scalar>& c_ineq,
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
template <typename Scalar>
Scalar l1_merit_directional_derivative(
    Scalar grad_f_dot_p,
    const Eigen::VectorX<Scalar>& c_eq,
    const Eigen::VectorX<Scalar>& c_ineq,
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
template <typename Scalar>
Scalar update_penalty(Scalar sigma,
                      const Eigen::VectorX<Scalar>& lambda,
                      Scalar delta = Scalar(1e-4))
{
    if(lambda.size() == 0)
        return sigma;
    Scalar required = lambda.cwiseAbs().maxCoeff() + delta;
    return std::max(sigma, required);
}

}

#endif
