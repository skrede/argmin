#ifndef HPP_GUARD_NABLAPP_DETAIL_AUGMENTED_LAGRANGIAN_H
#define HPP_GUARD_NABLAPP_DETAIL_AUGMENTED_LAGRANGIAN_H

// Augmented Lagrangian function value and gradient.
//
// Provides the augmented Lagrangian (method of multipliers) subproblem
// evaluation and multiplier update for equality and inequality
// constraints with penalty parameter mu.
//
// Reference: N&W Section 17.4, eq. 17.46-17.58;
//            K&W Section 10.9, Algorithm 10.2.

#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// Augmented Lagrangian function value.
//
// Equality contribution (N&W eq. 17.46):
//   sum_i [ -lambda_eq[i] * c_eq[i] + c_eq[i]^2 / (2*mu) ]
//
// Inequality contribution (N&W eq. 17.55-17.56, psi function):
//   if c_ineq[i] - mu*lambda_ineq[i] <= 0:
//     -lambda_ineq[i]*c_ineq[i] + c_ineq[i]^2 / (2*mu)
//   else:
//     -(mu/2)*lambda_ineq[i]^2
//
// Note: inequality convention is c_ineq >= 0.
template <typename Scalar>
Scalar augmented_lagrangian_value(
    Scalar f,
    const Eigen::VectorX<Scalar>& c_eq,
    const Eigen::VectorX<Scalar>& c_ineq,
    const Eigen::VectorX<Scalar>& lambda_eq,
    const Eigen::VectorX<Scalar>& lambda_ineq,
    Scalar mu)
{
    Scalar val = f;

    // Equality terms (N&W eq. 17.46)
    for(int i = 0; i < c_eq.size(); ++i)
        val += -lambda_eq[i] * c_eq[i] + c_eq[i] * c_eq[i] / (Scalar(2) * mu);

    // Inequality terms (N&W eq. 17.55-17.56)
    for(int i = 0; i < c_ineq.size(); ++i)
    {
        if(c_ineq[i] - mu * lambda_ineq[i] <= Scalar(0))
            val += -lambda_ineq[i] * c_ineq[i]
                   + c_ineq[i] * c_ineq[i] / (Scalar(2) * mu);
        else
            val += -(mu / Scalar(2)) * lambda_ineq[i] * lambda_ineq[i];
    }

    return val;
}

// Augmented Lagrangian gradient.
//
// Equality contribution (N&W eq. 17.47 derivative):
//   g -= (lambda_eq[i] - c_eq[i]/mu) * J_eq.row(i)^T
//
// Inequality contribution:
//   effective = lambda_ineq[i] - c_ineq[i]/mu
//   if effective > 0: g -= effective * J_ineq.row(i)^T
//
// J_eq and J_ineq are the constraint Jacobians (m_eq x n) and (m_ineq x n).
template <typename Scalar, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> augmented_lagrangian_gradient(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& J_eq,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& J_ineq,
    const Eigen::VectorX<Scalar>& c_eq,
    const Eigen::VectorX<Scalar>& c_ineq,
    const Eigen::VectorX<Scalar>& lambda_eq,
    const Eigen::VectorX<Scalar>& lambda_ineq,
    Scalar mu)
{
    Eigen::Vector<Scalar, N> g = grad_f;

    // Equality contribution
    for(int i = 0; i < c_eq.size(); ++i)
    {
        Scalar coeff = lambda_eq[i] - c_eq[i] / mu;
        g -= coeff * J_eq.row(i).transpose();
    }

    // Inequality contribution
    for(int i = 0; i < c_ineq.size(); ++i)
    {
        Scalar effective = lambda_ineq[i] - c_ineq[i] / mu;
        if(effective > Scalar(0))
            g -= effective * J_ineq.row(i).transpose();
    }

    return g;
}

// Update Lagrange multipliers after an outer iteration.
//
// Equality: lambda -= c/mu  (N&W eq. 17.49)
// Inequality: lambda = max(lambda - c/mu, 0)  (N&W eq. 17.58)
template <typename Scalar>
void update_multipliers(
    Eigen::VectorX<Scalar>& lambda_eq,
    Eigen::VectorX<Scalar>& lambda_ineq,
    const Eigen::VectorX<Scalar>& c_eq,
    const Eigen::VectorX<Scalar>& c_ineq,
    Scalar mu)
{
    for(int i = 0; i < lambda_eq.size(); ++i)
        lambda_eq[i] -= c_eq[i] / mu;

    for(int i = 0; i < lambda_ineq.size(); ++i)
        lambda_ineq[i] = std::max(lambda_ineq[i] - c_ineq[i] / mu, Scalar(0));
}

}

#endif
