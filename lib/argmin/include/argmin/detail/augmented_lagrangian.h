#ifndef HPP_GUARD_ARGMIN_DETAIL_AUGMENTED_LAGRANGIAN_H
#define HPP_GUARD_ARGMIN_DETAIL_AUGMENTED_LAGRANGIAN_H

// Augmented Lagrangian function value and gradient.
//
// Provides the augmented Lagrangian (method of multipliers) subproblem
// evaluation and multiplier update for equality and inequality
// constraints with penalty parameter mu.
//
// Reference: N&W Section 17.4, eq. 17.46-17.58;
//            K&W Section 10.9, Algorithm 10.2.

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace argmin::detail
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
//
// All Eigen-vector parameters are template-deduced expression types so
// the policy can pass head() / tail() segments of a maintained buffer
// without materializing temporaries (static-audit AL4/AL5).
template <typename Scalar,
          typename CeqExpr, typename CineqExpr,
          typename LeqExpr, typename LineqExpr>
Scalar augmented_lagrangian_value(
    Scalar f,
    const Eigen::DenseBase<CeqExpr>& c_eq,
    const Eigen::DenseBase<CineqExpr>& c_ineq,
    const Eigen::DenseBase<LeqExpr>& lambda_eq,
    const Eigen::DenseBase<LineqExpr>& lambda_ineq,
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

// Augmented Lagrangian gradient (in-place mutation of g).
//
// On entry g must contain grad_f at the current x. On exit g contains
// grad_f minus the Lagrangian / penalty contributions.
//
// Equality contribution (N&W eq. 17.47 derivative):
//   g -= J_eq^T * (lambda_eq - c_eq / mu)
//
// Inequality contribution (with the Rockafellar / N&W eq. 17.58 mask):
//   effective = (lambda_ineq - c_ineq / mu).cwiseMax(0)
//   g -= J_ineq^T * effective
//
// Both contributions are single mat-vec products instead of the prior
// row-by-row accumulation (static-audit AL9). All Eigen parameters are
// template-deduced expression types so the policy can pass topRows /
// bottomRows / head / tail segments of maintained buffers without
// materializing temporaries (static-audit AL4 / AL5).
//
// J_eq is (m_eq x n); J_ineq is (m_ineq x n).
template <typename Scalar, int N,
          typename JeqExpr, typename JineqExpr,
          typename CeqExpr, typename CineqExpr,
          typename LeqExpr, typename LineqExpr>
void augmented_lagrangian_gradient_inplace(
    Eigen::Vector<Scalar, N>& g,
    const Eigen::DenseBase<JeqExpr>& J_eq,
    const Eigen::DenseBase<JineqExpr>& J_ineq,
    const Eigen::DenseBase<CeqExpr>& c_eq,
    const Eigen::DenseBase<CineqExpr>& c_ineq,
    const Eigen::DenseBase<LeqExpr>& lambda_eq,
    const Eigen::DenseBase<LineqExpr>& lambda_ineq,
    Scalar mu)
{
    if(c_eq.size() > 0)
        g.noalias() -= J_eq.derived().transpose()
                       * (lambda_eq.derived() - c_eq.derived() / mu);
    if(c_ineq.size() > 0)
        g.noalias() -= J_ineq.derived().transpose()
                       * (lambda_ineq.derived() - c_ineq.derived() / mu).cwiseMax(Scalar(0));
}

// Backward-compatible return-by-value wrapper. Allocates a copy of
// grad_f internally; the in-place form above should be preferred on the
// hot path. Kept for the AL convergence-diagnostic call site that
// computes a one-shot Lagrangian gradient norm.
template <typename Scalar, int N,
          typename JeqExpr, typename JineqExpr,
          typename CeqExpr, typename CineqExpr,
          typename LeqExpr, typename LineqExpr>
Eigen::Vector<Scalar, N> augmented_lagrangian_gradient(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::DenseBase<JeqExpr>& J_eq,
    const Eigen::DenseBase<JineqExpr>& J_ineq,
    const Eigen::DenseBase<CeqExpr>& c_eq,
    const Eigen::DenseBase<CineqExpr>& c_ineq,
    const Eigen::DenseBase<LeqExpr>& lambda_eq,
    const Eigen::DenseBase<LineqExpr>& lambda_ineq,
    Scalar mu)
{
    Eigen::Vector<Scalar, N> g = grad_f;
    augmented_lagrangian_gradient_inplace(
        g, J_eq, J_ineq, c_eq, c_ineq, lambda_eq, lambda_ineq, mu);
    return g;
}

// Update Lagrange multipliers after an outer iteration.
//
// Equality: lambda -= c/mu  (N&W eq. 17.49)
// Inequality: lambda = max(lambda - c/mu, 0)  (N&W eq. 17.58)
//
// Multiplier magnitudes are clamped to lambda_max to prevent divergence
// when mu is small and constraints are poorly satisfied. This safeguard
// is standard practice in augmented Lagrangian implementations; see
// Conn, Gould & Toint, "A globally convergent augmented Lagrangian
// algorithm for optimization with general constraints and simple bounds",
// SIAM J. Numer. Anal. 28(2), 1991, Section 3.
template <typename Scalar, int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
void update_multipliers(
    Eigen::Vector<Scalar, Meq>& lambda_eq,
    Eigen::Vector<Scalar, Mineq>& lambda_ineq,
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    Scalar mu,
    Scalar lambda_max = Scalar(1e8))
{
    for(int i = 0; i < lambda_eq.size(); ++i)
    {
        lambda_eq[i] -= c_eq[i] / mu;
        lambda_eq[i] = std::clamp(lambda_eq[i], -lambda_max, lambda_max);
    }

    for(int i = 0; i < lambda_ineq.size(); ++i)
    {
        lambda_ineq[i] = std::max(lambda_ineq[i] - c_ineq[i] / mu, Scalar(0));
        lambda_ineq[i] = std::min(lambda_ineq[i], lambda_max);
    }
}

}

#endif
