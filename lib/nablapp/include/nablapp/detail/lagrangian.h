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

#include "nablapp/types.h"

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
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> lagrangian_gradient(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, M, N>& A,
    const Eigen::Vector<Scalar, M>& lambda)
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
template <typename Scalar, int Meq = nablapp::dynamic_dimension, int Mineq = nablapp::dynamic_dimension>
Scalar constraint_violation(
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq)
{
    Scalar v = Scalar(0);
    if(c_eq.size() > 0)
        v += c_eq.cwiseAbs().sum();
    if(c_ineq.size() > 0)
        v += (-c_ineq).cwiseMax(Scalar(0)).sum();
    return v;
}

// Primal feasibility (L-infinity): max(||c_eq||_inf, ||max(-c_ineq, 0)||_inf).
// L-infinity composition matches kkt_residual legs 2 and 3 so
// step_result.constraint_violation is dimensionally consistent with
// step_result.kkt_residual (both L-infinity). This is the reporting
// convention used by IPOPT (primal_inf) and matches the KKT primal-
// feasibility measure.
//
// Note: the sibling detail::constraint_violation uses the L1 convention
// (sum |c_eq[i]| + sum max(-c_ineq[j], 0)) required by filter theory
// (Fletcher-Leyffer 2002 Section 2 filter dominance ordering) and L1
// merit functions (N&W eq. 15.24 / 17.44).
//
// Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
template <typename Scalar,
          int Meq = nablapp::dynamic_dimension,
          int Mineq = nablapp::dynamic_dimension>
Scalar primal_feasibility_inf(
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq)
{
    const Scalar eq_leg = c_eq.size() > 0
        ? c_eq.template lpNorm<Eigen::Infinity>()
        : Scalar{0};
    const Scalar ineq_leg = c_ineq.size() > 0
        ? (-c_ineq).cwiseMax(Scalar{0}).template lpNorm<Eigen::Infinity>()
        : Scalar{0};
    return std::max(eq_leg, ineq_leg);
}

// Estimate Lagrange multipliers via least-squares.
//
// Solves min_lambda ||grad_f - A^T lambda||_2 which gives
// lambda = (A A^T)^{-1} A grad_f when A has full row rank.
// Uses ColPivHouseholderQR for numerical robustness.
//
// Reference: N&W eq. 18.15.
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, M> estimate_multipliers(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, M, N>& A)
{
    if(A.rows() == 0)
        return Eigen::Vector<Scalar, M>{};

    // Solve A^T lambda = grad_f in the least-squares sense.
    auto qr = A.transpose().colPivHouseholderQr();
    return qr.solve(grad_f);
}

// Active-set least-squares multiplier estimate.
//
// Detects active inequalities (|c_ineq[i]| < active_tol), solves the
// min-residual LS problem on the union of equality and active inequality
// rows only, and returns the full (n_eq + n_ineq) multiplier vector in
// the canonical [lambda_eq; mu_ineq] layout. Inactive mu_ineq entries
// are zero; active mu_ineq entries are projected to >= 0 (KKT dual
// feasibility).
//
// This is strictly more correct than estimate_multipliers + cwiseMax on
// problems where two inequalities have parallel constraint gradients at
// the optimum and only one is binding: standard LS picks an arbitrary
// split between the parallel rows that breaks dual feasibility after
// sign projection, leaving the stationarity leg positive at the KKT
// point. Restricting the LS to the binding rows only recovers the
// textbook active-set multiplier.
//
// Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set / active-
//            set identification);
//            eq. 18.15 (least-squares lambda);
//            Definition 12.1 (KKT conditions require mu_ineq >= 0 and
//            complementarity).
template <typename Scalar,
          int N = nablapp::dynamic_dimension,
          int Meq = nablapp::dynamic_dimension,
          int Mineq = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, Eigen::Dynamic> estimate_multipliers_active_set(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    Scalar active_tol = Scalar(1e-8))
{
    const Eigen::Index n = grad_f.size();
    const Eigen::Index n_eq = J_eq.rows();
    const Eigen::Index n_ineq = J_ineq.rows();

    Eigen::Vector<Scalar, Eigen::Dynamic> lambda_full =
        Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(n_eq + n_ineq);

    // Count active inequalities (|c_ineq[i]| < active_tol).
    Eigen::Index n_active = 0;
    for(Eigen::Index i = 0; i < n_ineq; ++i)
        if(std::abs(c_ineq[i]) < active_tol)
            ++n_active;

    const Eigen::Index m_active = n_eq + n_active;
    if(m_active == 0)
        return lambda_full;

    // Build reduced constraint Jacobian: [J_eq; J_ineq_active].
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A_active(m_active, n);
    if(n_eq > 0)
        A_active.topRows(n_eq) = J_eq;
    Eigen::Index row = n_eq;
    for(Eigen::Index i = 0; i < n_ineq; ++i)
        if(std::abs(c_ineq[i]) < active_tol)
        {
            A_active.row(row) = J_ineq.row(i);
            ++row;
        }

    // LS on the reduced system (N&W eq. 18.15).
    auto qr = A_active.transpose().colPivHouseholderQr();
    Eigen::Vector<Scalar, Eigen::Dynamic> grad_f_dyn(n);
    grad_f_dyn = grad_f;
    Eigen::Vector<Scalar, Eigen::Dynamic> lambda_active = qr.solve(grad_f_dyn);

    // Equality multipliers: sign is unrestricted.
    if(n_eq > 0)
        lambda_full.head(n_eq) = lambda_active.head(n_eq);

    // Active inequality multipliers: clipped to >= 0 for dual feasibility.
    Eigen::Index ak = 0;
    for(Eigen::Index i = 0; i < n_ineq; ++i)
        if(std::abs(c_ineq[i]) < active_tol)
        {
            lambda_full[n_eq + i] = std::max(Scalar(0), lambda_active[n_eq + ak]);
            ++ak;
        }

    return lambda_full;
}

}

#endif
