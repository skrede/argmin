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

#include "argmin/detail/householder_apply.h"

#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>

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

    // Solve A^T lambda = grad_f in the least-squares sense.
    auto qr = A.transpose().colPivHouseholderQr();
    return qr.solve(grad_f);
}

// The rows a multiplier estimate may use: every equality and each inequality inside its active band. A row id
// indexes the stacked [J_eq; J_ineq].
template <typename Scalar, int N, int Meq, int Mineq>
struct active_rows
{
    Scalar kappa;
    const Eigen::Matrix<Scalar, Meq, N>& J_eq;
    const Eigen::Vector<Scalar, Mineq>& c_ineq;
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq;

    // Active in the row's own units, |c_i| < kappa * max(1, ||J_i||), so rescaling a constraint does not move it in or
    // out of the band; kappa defaults to sqrt(eps), the classical working-set identification band.
    bool active(Eigen::Index i) const { return std::abs(c_ineq[i]) < kappa * std::max(Scalar(1), J_ineq.row(i).norm()); }

    Eigen::Index count() const { return J_eq.rows() + J_ineq.rows(); }

    Eigen::Index next(Eigen::Index id) const
    {
        while(id < count() && id >= J_eq.rows() && !active(id - J_eq.rows()))
            ++id;
        return id;
    }

    template <typename Row>
    void copy(Eigen::Index id, Row&& row) const
    {
        if(id < J_eq.rows())
            copy_row(J_eq, id, row);
        else
            copy_row(J_ineq, id - J_eq.rows(), row);
    }

    // Eigen rejects a row block of a compile-time zero-row matrix, so that copy is never instantiated.
    template <typename Jacobian, typename Row>
    static void copy_row(const Jacobian& J, Eigen::Index i, Row& row)
    {
        if constexpr(Jacobian::RowsAtCompileTime != 0)
            row = J.row(i);
    }
};

// Caller-owned workspace for compute_kkt_multipliers_active_set, held by the SQP policy state and grow-only, so
// re-estimation at a stable active-set shape performs no allocation. The estimate never factorizes more rows than
// variables, so MaxN bounds both QR axes and every buffer, and fixed N stays in inline storage.
template <typename Scalar, int MaxN = Eigen::Dynamic>
struct kkt_multiplier_workspace
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::ColMajor, MaxN, MaxN> A_active;
    Eigen::ColPivHouseholderQR<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::ColMajor, MaxN, MaxN>> qr;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> grad_f_dyn;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> lambda_active;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> scaled;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> row_ids;
};

template <typename Rows, typename Scalar, int MaxN>
void load_rows(const Rows& rows, Eigen::Index count, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    if(ws.A_active.rows() != count || ws.A_active.cols() != n)
        ws.A_active.resize(count, n);
    for(Eigen::Index k = 0; k < count; ++k)
        rows.copy(ws.row_ids[k], ws.A_active.row(k));
}

template <typename Rows, typename Scalar, int MaxN>
Eigen::Index fill_rows(const Rows& rows, Eigen::Index& id, Eigen::Index count, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    for(; count < ws.row_ids.size() && id < rows.count(); id = rows.next(id + 1))
        ws.row_ids[count++] = id;
    return count;
}

// Keeps, in order, the rows that are pivot columns within the rank of A_active^T.
template <typename Rows, typename Scalar, int MaxN>
Eigen::Index keep_pivot_rows(const Rows& rows, Eigen::Index count, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    load_rows(rows, count, n, ws);
    ws.qr.compute(ws.A_active.transpose());
    const auto pivots = ws.qr.colsPermutation().indices().head(ws.qr.rank());
    using pivot_index = typename std::decay_t<decltype(pivots)>::Scalar;
    Eigen::Index kept = 0;
    for(Eigen::Index k = 0; k < count; ++k)
        if((pivots.array() == static_cast<pivot_index>(k)).any())
            ws.row_ids[kept++] = ws.row_ids[k];
    return kept;
}

// At a degenerate vertex the active rows outnumber the variables; column-pivoted QR reduces them to a maximal linearly
// independent subset, refilling dropped slots in row order. Every such subset spans the same row space, so the
// least-squares stationarity residual is unchanged, and the multipliers are not unique there in any case.
template <typename Rows, typename Scalar, int MaxN>
Eigen::Index select_rows(const Rows& rows, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    if(ws.row_ids.size() != n)
        ws.row_ids.resize(n);
    Eigen::Index id = rows.next(0);
    Eigen::Index count = fill_rows(rows, id, 0, ws);
    if(id == rows.count())
        return count;
    while(true)
    {
        count = keep_pivot_rows(rows, count, n, ws);
        if(count == n || id == rows.count())
            return count;
        count = fill_rows(rows, id, count, ws);
    }
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
// feasibility after the sign projection.
//
// Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set identification);
//            eq. 18.15 (least-squares lambda);
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
    kkt_multiplier_workspace<Scalar, MaxN>& ws,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> lam_eq_out,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> mu_ineq_out,
    Scalar active_kappa = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
{
    const Eigen::Index n = g.size();
    lam_eq_out.setZero();
    mu_ineq_out.setZero();
    const active_rows<Scalar, N, Meq, Mineq> rows{active_kappa, J_eq, c_ineq, J_ineq};
    const Eigen::Index count = select_rows(rows, n, ws);
    if(count == 0)
        return;

    load_rows(rows, count, n, ws);
    ws.qr.compute(ws.A_active.transpose());
    if(ws.grad_f_dyn.size() != n) ws.grad_f_dyn.resize(n);
    if(ws.scaled.size() != n) ws.scaled.resize(n);
    if(ws.lambda_active.size() < count) ws.lambda_active.resize(count);
    ws.grad_f_dyn = g;
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
    Scalar active_kappa = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
{
    Eigen::Vector<Scalar, Eigen::Dynamic> lambda(J_eq.rows() + J_ineq.rows());
    kkt_multiplier_workspace<Scalar> ws;
    compute_kkt_multipliers_active_set<Scalar, N, Meq, Mineq, Eigen::Dynamic>(
        grad_f, J_eq, J_ineq, c_ineq, ws, lambda.head(J_eq.rows()), lambda.tail(J_ineq.rows()), active_kappa);
    return lambda;
}

}

#endif
