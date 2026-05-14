#ifndef HPP_GUARD_ARGMIN_DETAIL_LSEI_H
#define HPP_GUARD_ARGMIN_DETAIL_LSEI_H

// Equality + inequality constrained least squares (LSEI).
//
// Solves: min ||E*x - f||^2  s.t.  C*x = d, G*x >= h
//
// Eliminates the equality constraints via a Householder QR of C^T,
// compresses the reduced LS via a second QR of E, then hands off the
// inequality-only problem to detail::lsi.
//
// Reference: Kraft, D. (1988). A Software Package for Sequential
//            Quadratic Programming. DFVLR-FB 88-28, Section 3.2.
//            Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Prentice-Hall. Chapter 23, Section 23.6 (LSEI).

#include "argmin/detail/lsi.h"
#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <limits>
#include <cmath>

namespace argmin::detail
{

// Persistent workspace for lsei() so the routine does not allocate per
// call. Sized once via resize() at the maximum problem shape; subsequent
// calls at smaller shapes reuse the existing storage.
template <typename Scalar>
struct lsei_workspace
{
    using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Vector<Scalar, Eigen::Dynamic>;

    matrix_t R_c_upper;
    vector_t y1;
    matrix_t E_tilde;
    vector_t f_red;
    matrix_t G_tilde;
    vector_t h_red;
    matrix_t R_e;
    vector_t Qt_f;
    matrix_t E_small;
    vector_t f_small_vec;
    matrix_t G_small;
    vector_t y2_dyn;
    vector_t y;

    Eigen::HouseholderQR<matrix_t> qr_c;
    Eigen::HouseholderQR<matrix_t> qr_e;

    // Buffers used to recover lambda_eq from the gradient projection on
    // the y_1 subspace at the LSEI optimum. Sized to n.
    vector_t lambda_eq_resid;   // E*x - f, then Q_c^T * grad
    vector_t lambda_eq_qgrad;   // E^T * resid - G^T * lambda_ineq

    // Nested workspace for the inner lsi() call. Sized for the reduced
    // problem (n2, m_ineq) and reused across solves.
    lsi_workspace<Scalar> lsi_ws;

    void resize(int n, int m_eq, int m_ineq)
    {
        const int n2 = n - m_eq;
        R_c_upper.resize(m_eq, m_eq);
        y1.resize(m_eq);
        E_tilde.resize(n, n);
        f_red.resize(n);
        G_tilde.resize(m_ineq, n);
        h_red.resize(m_ineq);
        if(n2 > 0)
        {
            R_e.resize(n2, n2);
            Qt_f.resize(n);
            E_small.resize(n2, n2);
            f_small_vec.resize(n2);
            G_small.resize(m_ineq, n2);
            y2_dyn.resize(n2);
        }
        y.resize(n);
        lambda_eq_resid.resize(n);
        lambda_eq_qgrad.resize(n);

        // Worst-case shape for inner lsi: when m_eq == 0 (pure LSI
        // fallback), lsi sees the original n; otherwise it sees the
        // reduced n2. Size for the original n and m_ineq so both
        // entry paths are covered.
        lsi_ws.resize(n, m_ineq);
    }
};

// Solve min ||E*x - f||^2 s.t. C*x = d, G*x >= h.
//
// C is (m_eq x n), d is (m_eq), E is (n x n), f is (n),
// G is (m_ineq x n), h is (m_ineq), x is (n) output.
//
// The lsei_workspace ws holds matrix / vector / QR factorization state
// used by lsei itself; the caller owns it and is responsible for sizing
// it via ws.resize() at problem configuration time. The nnls_* scratch
// buffers are passed through to the underlying LDP solver; the caller
// must also ensure they can hold the dual NNLS problem (see ldp()
// documentation).
//
// lambda_eq is (m_eq) output: equality multipliers, sign-free.
// lambda_ineq is (m_ineq) output: inequality multipliers, lambda_ineq
// >= 0, complementary with the active set of G x >= h. The inequality
// multipliers come from the inner LSI/LDP cascade; the equality
// multipliers are recovered from the gradient projection
//   R_c * lambda_eq = (Q_c^T (E^T (E x - f) - G^T lambda_ineq))[0..m_eq]
// where Q_c R_c = QR(C^T). This avoids the post-hoc least-squares fit
// previously done in the QP caller.
//
// Return code:
//   1 on success,
//   4 if the inequality system is incompatible,
//   6 if the equality matrix C is rank deficient.
//
// Reference: Kraft 1988 DFVLR-FB 88-28, Section 3.2, eq. 3.19-3.21;
//            Lawson & Hanson 1974, Ch. 23.6;
//            N&W 2e Section 16.5 (KKT multiplier recovery via QR(C^T)).
template <typename Scalar, int N>
int lsei(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& C,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& d,
    Eigen::Matrix<Scalar, N, N>& E,
    Eigen::Vector<Scalar, N>& f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& G,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& h,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& lambda_eq,
    Eigen::Vector<Scalar, Eigen::Dynamic>& lambda_ineq,
    lsei_workspace<Scalar>& ws,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& nnls_A,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_b,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_x_vec,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_w,
    int n, int m_eq, int m_ineq)
{
    // ------------------------------------------------------------------
    // Pure LSI fallback: no equality constraints.
    // ------------------------------------------------------------------
    if(m_eq <= 0)
    {
        if(lambda_eq.size() != 0) lambda_eq.setZero();
        return lsi<Scalar, N>(
            E, f, G, h, x, lambda_ineq, ws.lsi_ws,
            nnls_A, nnls_b, nnls_x_vec, nnls_w, n, m_ineq);
    }

    if(lambda_eq.size() != m_eq) lambda_eq.resize(m_eq);
    lambda_eq.setZero();
    if(lambda_ineq.size() != m_ineq) lambda_ineq.resize(m_ineq);
    lambda_ineq.setZero();

    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    // ------------------------------------------------------------------
    // QR factorize C^T = Q_C * R_C.
    //
    // Then with x = Q_C * y the equality C*x = d becomes
    //   C * Q_C * y = R_C^T * y = d  (acting on the leading m_eq entries)
    // which is a lower-triangular system for y1 = y(0..m_eq-1).
    // ------------------------------------------------------------------
    ws.qr_c.compute(C.transpose());

    // Q_c is left implicit; subsequent operations apply the Householder
    // sequence directly via Eigen::HouseholderSequence overloads (matrix
    // applyOnTheRight, vector multiplication, transpose-vector
    // multiplication). The explicit Q_c = householderQ() * Identity(n, n)
    // materialization is avoided.
    // Reference: Eigen::HouseholderSequence (Eigen/QR).

    // R_c is m_eq x m_eq upper triangular, extracted from qr_c.matrixQR().
    if(ws.R_c_upper.rows() != m_eq || ws.R_c_upper.cols() != m_eq)
        ws.R_c_upper.resize(m_eq, m_eq);
    ws.R_c_upper = ws.qr_c.matrixQR().topLeftCorner(m_eq, m_eq)
                          .template triangularView<Eigen::Upper>();

    // Check rank via diagonal of R_c.
    for(int i = 0; i < m_eq; ++i)
    {
        if(std::abs(ws.R_c_upper(i, i)) <= eps * Scalar(n) * Scalar(10))
            return 6;
    }

    // Solve R_c^T * y1 = d for y1 (lower triangular solve).
    if(ws.y1.size() != m_eq) ws.y1.resize(m_eq);
    ws.y1 = ws.R_c_upper.transpose()
                        .template triangularView<Eigen::Lower>()
                        .solve(d);

    // ------------------------------------------------------------------
    // Change variables: E_tilde = E * Q_c, split into [E1 (n x m_eq),
    // E2 (n x (n - m_eq))]. Similarly for G. The split is referenced
    // below via leftCols / rightCols on E_tilde directly to avoid a
    // separate E1 / E2 allocation.
    // ------------------------------------------------------------------
    if(ws.E_tilde.rows() != n || ws.E_tilde.cols() != n) ws.E_tilde.resize(n, n);
    // Implicit Householder sequence application; Eigen handles the
    // accumulation without materializing an n x n Q matrix.
    // Reference: Eigen::HouseholderSequence::applyOnTheRight.
    ws.E_tilde = E;
    ws.E_tilde.applyOnTheRight(ws.qr_c.householderQ());

    if(ws.f_red.size() != n) ws.f_red.resize(n);
    ws.f_red.noalias() = f - ws.E_tilde.leftCols(m_eq) * ws.y1;

    // ------------------------------------------------------------------
    // Reduced LS problem in the free variables y2 (length n - m_eq):
    //   min ||E2 * y2 - f_red||^2  s.t.  (G * Qc)_{:,m_eq..} y2
    //                                    >= h - (G * Qc)_{:,0..m_eq-1} y1
    //
    // E2 has more rows than columns; compress via a QR factorization so
    // the inner LSI call receives a square (n-m_eq) x (n-m_eq) system.
    // ------------------------------------------------------------------
    const int n2 = n - m_eq;

    if(m_ineq > 0)
    {
        if(ws.G_tilde.rows() != m_ineq || ws.G_tilde.cols() != n)
            ws.G_tilde.resize(m_ineq, n);
        // Implicit Householder sequence application (see E_tilde site above).
        ws.G_tilde = G;
        ws.G_tilde.applyOnTheRight(ws.qr_c.householderQ());
        if(ws.h_red.size() != m_ineq) ws.h_red.resize(m_ineq);
        ws.h_red.noalias() = h - ws.G_tilde.leftCols(m_eq) * ws.y1;
    }

    if(n2 > 0)
    {
        // QR of E2 (n x n2). E2 = Q_e * [R_e; 0] with R_e of size n2 x n2.
        ws.qr_e.compute(ws.E_tilde.rightCols(n2));
        if(ws.R_e.rows() != n2 || ws.R_e.cols() != n2) ws.R_e.resize(n2, n2);
        ws.R_e = ws.qr_e.matrixQR().topLeftCorner(n2, n2)
                        .template triangularView<Eigen::Upper>();

        if(ws.Qt_f.size() != n) ws.Qt_f.resize(n);
        ws.Qt_f.noalias() = ws.qr_e.householderQ().transpose() * ws.f_red;

        if(ws.y2_dyn.size() != n2) ws.y2_dyn.resize(n2);

        if(m_ineq <= 0)
        {
            // Pure equality-constrained LS: solve R_e y2 = Qt_f.head(n2).
            // Check R_e rank.
            for(int i = 0; i < n2; ++i)
            {
                if(std::abs(ws.R_e(i, i)) <= eps * Scalar(n) * Scalar(10))
                    return 6;
            }
            ws.y2_dyn = ws.R_e.template triangularView<Eigen::Upper>()
                              .solve(ws.Qt_f.head(n2).eval());
        }
        else
        {
            // Build a square n2 x n2 E for the inner LSI call.
            if(ws.E_small.rows() != n2 || ws.E_small.cols() != n2)
                ws.E_small.resize(n2, n2);
            ws.E_small = ws.R_e;
            if(ws.f_small_vec.size() != n2) ws.f_small_vec.resize(n2);
            ws.f_small_vec = ws.Qt_f.head(n2);
            if(ws.G_small.rows() != m_ineq || ws.G_small.cols() != n2)
                ws.G_small.resize(m_ineq, n2);
            ws.G_small = ws.G_tilde.rightCols(n2);

            ws.y2_dyn.setZero();

            if(lambda_ineq.size() != m_ineq) lambda_ineq.resize(m_ineq);
            int lsi_mode = lsi<Scalar, Eigen::Dynamic>(
                ws.E_small, ws.f_small_vec, ws.G_small, ws.h_red, ws.y2_dyn,
                lambda_ineq,
                ws.lsi_ws,
                nnls_A, nnls_b, nnls_x_vec, nnls_w, n2, m_ineq);

            if(lsi_mode != 1)
                return lsi_mode;
        }
    }

    // ------------------------------------------------------------------
    // Assemble y = [y1; y2] and back-transform x = Q_c * y.
    // ------------------------------------------------------------------
    if(ws.y.size() != n) ws.y.resize(n);
    ws.y.head(m_eq) = ws.y1;
    if(n2 > 0)
        ws.y.tail(n2) = ws.y2_dyn;

    // Implicit Householder sequence applied to the assembled y vector;
    // Eigen evaluates the product without materializing Q_c.
    x = ws.qr_c.householderQ() * ws.y;

    // ------------------------------------------------------------------
    // Recover equality multipliers from the gradient projection on y_1.
    //
    // KKT for LSEI:
    //   E^T (E x - f) = C^T lambda_eq + G^T lambda_ineq
    //
    // In y coordinates (x = Q_c y) the equality block becomes:
    //   (Q_c^T (E^T (E x - f) - G^T lambda_ineq))[0..m_eq] = R_c lambda_eq
    //
    // because (C Q_c)^T = [R_c; 0]. R_c is the upper triangular block
    // already extracted as ws.R_c_upper. lambda_ineq is filled by the
    // inner LSI call (or zero on the m_ineq <= 0 reduced-LS branch).
    //
    // Reference: N&W 2e Section 16.5, eq. 16.50 (KKT multiplier recovery
    //            via QR(C^T)); Kraft 1988 DFVLR-FB 88-28 Section 3.2.
    // ------------------------------------------------------------------
    if(ws.lambda_eq_resid.size() != n) ws.lambda_eq_resid.resize(n);
    if(ws.lambda_eq_qgrad.size() != n) ws.lambda_eq_qgrad.resize(n);

    ws.lambda_eq_resid.noalias() = E * x;
    ws.lambda_eq_resid -= f;

    ws.lambda_eq_qgrad.noalias() = E.transpose() * ws.lambda_eq_resid;
    if(m_ineq > 0)
        ws.lambda_eq_qgrad.noalias() -= G.transpose() * lambda_ineq;

    // Re-use lambda_eq_resid as Q_c^T * qgrad scratch.
    // Implicit Householder-sequence transpose application; Q_c is never
    // materialized.
    ws.lambda_eq_resid = ws.qr_c.householderQ().transpose() * ws.lambda_eq_qgrad;

    lambda_eq = ws.R_c_upper.template triangularView<Eigen::Upper>()
                            .solve(ws.lambda_eq_resid.head(m_eq).eval());

    return 1;
}

}

#endif
