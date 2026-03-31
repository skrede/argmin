#ifndef HPP_GUARD_NABLAPP_DETAIL_LAGRANGIAN_H
#define HPP_GUARD_NABLAPP_DETAIL_LAGRANGIAN_H

// Lagrangian utilities for constrained optimization.
//
// Provides the gradient of the Lagrangian, constraint violation measure,
// and least-squares multiplier estimation.
//
// Reference: N&W Section 18.1, eq. 18.2-18.3 (Lagrangian gradient);
//            N&W eq. 15.24 / 17.44 (constraint violation);
//            N&W eq. 18.15 (multiplier estimation).

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// Lagrangian gradient: grad_L(x, lambda) = grad_f - A^T * lambda.
//
// A is the constraint Jacobian (m x n), lambda is the multiplier
// vector (m x 1). The Lagrangian is L = f - lambda^T c, so
// grad_x L = grad_f - A^T lambda.
//
// Reference: N&W eq. 18.2-18.3.
template <typename Scalar>
Eigen::VectorX<Scalar> lagrangian_gradient(
    const Eigen::VectorX<Scalar>& grad_f,
    const Eigen::MatrixX<Scalar>& A,
    const Eigen::VectorX<Scalar>& lambda)
{
    if(A.rows() == 0)
        return grad_f;
    return (grad_f - A.transpose() * lambda).eval();
}

// Constraint violation: ||c_eq||_1 + ||max(0, -c_ineq)||_1.
//
// Equality constraints must be zero; inequality constraints must be
// non-negative (c_ineq >= 0 convention). Violated inequalities
// contribute their negative part.
//
// Reference: N&W eq. 15.24 / 17.44 (violation part).
template <typename Scalar>
Scalar constraint_violation(
    const Eigen::VectorX<Scalar>& c_eq,
    const Eigen::VectorX<Scalar>& c_ineq)
{
    Scalar v = Scalar(0);
    if(c_eq.size() > 0)
        v += c_eq.cwiseAbs().sum();
    if(c_ineq.size() > 0)
        v += (-c_ineq).cwiseMax(Scalar(0)).sum();
    return v;
}

// Estimate Lagrange multipliers via least-squares.
//
// Solves min_lambda ||grad_f - A^T lambda||_2 which gives
// lambda = (A A^T)^{-1} A grad_f when A has full row rank.
// Uses ColPivHouseholderQR for numerical robustness.
//
// Reference: N&W eq. 18.15.
template <typename Scalar>
Eigen::VectorX<Scalar> estimate_multipliers(
    const Eigen::VectorX<Scalar>& grad_f,
    const Eigen::MatrixX<Scalar>& A)
{
    if(A.rows() == 0)
        return Eigen::VectorX<Scalar>{};

    // Solve A^T lambda = grad_f in the least-squares sense.
    auto qr = A.transpose().colPivHouseholderQr();
    return qr.solve(grad_f);
}

}

#endif
