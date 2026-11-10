#ifndef HPP_GUARD_ARGMIN_DETAIL_LAGRANGIAN_H
#define HPP_GUARD_ARGMIN_DETAIL_LAGRANGIAN_H

// Lagrangian utilities for constrained optimization.
//
// Provides the gradient of the Lagrangian, constraint violation measure,
// and least-squares multiplier estimation.
//
// Reference: N&W Section 18.1, eq. 18.2-18.3 (Lagrangian gradient);
//            N&W eq. 15.24 / 17.44 (constraint violation);
//            N&W eq. 18.15 (multiplier estimation).

#include "argmin/detail/kkt_working_set.h"
#include "argmin/detail/householder_apply.h"

#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <cmath>
#include <limits>
#include <algorithm>

namespace argmin::detail
{

// Lagrangian gradient: grad_L(x, lambda) = grad_f - A^T * lambda.
//
// A is the constraint Jacobian (m x n), lambda is the multiplier
// vector (m x 1). The Lagrangian is L = f - lambda^T c, so
// grad_x L = grad_f - A^T lambda.
//
// Reference: N&W eq. 18.2-18.3.
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
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
template <typename Scalar, int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
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
          int Meq = argmin::dynamic_dimension,
          int Mineq = argmin::dynamic_dimension>
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
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
Eigen::Vector<Scalar, M> estimate_multipliers(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, M, N>& A)
{
    if(A.rows() == 0)
        return Eigen::Vector<Scalar, M>{};

    auto qr = A.transpose().colPivHouseholderQr();
    return qr.solve(grad_f);
}

template <typename Rows, typename Scalar, int MaxN>
void scatter_multipliers(const Rows& rows, Eigen::Index count, const kkt_multiplier_workspace<Scalar, MaxN>& ws,
                         Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> lam_eq_out,
                         Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> mu_ineq_out)
{
    for(Eigen::Index k = 0; k < count; ++k)
    {
        const Eigen::Index id = ws.row_ids[k];
        if(id < rows.J_eq.rows())
            lam_eq_out[id] = ws.lambda_active[k];
        else
            mu_ineq_out[id - rows.J_eq.rows()] = std::max(Scalar(0), ws.lambda_active[k]);
    }
}

// Active-set least-squares multiplier estimate into caller-owned outputs: the least-squares problem is restricted to
// the equalities and the active inequalities, active mu_ineq entries are projected to >= 0 (KKT dual feasibility) and
// every other entry is zero. Restricting to the binding rows recovers the textbook active-set multiplier where two
// parallel inequalities share a gradient and only one binds; a least-squares split between them breaks dual
// feasibility after the sign projection. A box bound in the band joins the working set as the row e_j, whose free
// multiplier eliminates from eq. 18.15 exactly: the fit runs over the other coordinates and (g - J^T lambda)_j is left
// as the bound multiplier, whose sign the projected-gradient stationarity leg checks.
//
// Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set identification);
//            eq. 18.15 (least-squares lambda); Section 16.7 (projected gradient);
//            Definition 12.1 (KKT conditions require mu_ineq >= 0 and complementarity);
//            Kraft 1988 DFVLR-FB 88-28 Section 2.2.4.
template <typename Scalar,
          int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension,
          int Mineq = argmin::dynamic_dimension,
          int MaxN = Eigen::Dynamic>
ARGMIN_FORCE_INLINE void compute_kkt_multipliers_active_set(
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    kkt_multiplier_workspace<Scalar, MaxN>& ws,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> lam_eq_out,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> mu_ineq_out,
    Scalar active_kappa = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
{
    const Eigen::Index n = g.size();
    lam_eq_out.setZero();
    mu_ineq_out.setZero();
    const active_rows<Scalar, N, Meq, Mineq> rows{active_kappa, x, lower, upper, J_eq, c_ineq, J_ineq};
    const Eigen::Index count = select_rows(rows, n, ws);
    if(count == 0)
        return;

    load_rows(rows, count, n, ws);
    ws.qr.compute(ws.A_active.transpose());
    if(ws.scaled.size() != n) ws.scaled.resize(n);
    if(ws.lambda_active.size() < count) ws.lambda_active.resize(count);
    load_gradient(rows, g, ws);
    col_piv_qr_solve_in_place(ws.qr, ws.grad_f_dyn, ws.lambda_active.head(count), ws.scaled);
    scatter_multipliers(rows, count, ws, lam_eq_out, mu_ineq_out);
}

// The allocating form of compute_kkt_multipliers_active_set, returning the canonical [lambda_eq; mu_ineq] layout.
template <typename Scalar,
          int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension,
          int Mineq = argmin::dynamic_dimension>
Eigen::Vector<Scalar, Eigen::Dynamic> estimate_multipliers_active_set(
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Scalar active_kappa = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
{
    Eigen::Vector<Scalar, Eigen::Dynamic> lambda(J_eq.rows() + J_ineq.rows());
    kkt_multiplier_workspace<Scalar> ws;
    compute_kkt_multipliers_active_set<Scalar, N, Meq, Mineq, Eigen::Dynamic>(
        grad_f, J_eq, J_ineq, c_ineq, x, lower, upper, ws, lambda.head(J_eq.rows()), lambda.tail(J_ineq.rows()), active_kappa);
    return lambda;
}

}

#endif
