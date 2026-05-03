#ifndef HPP_GUARD_ARGMIN_DETAIL_LSI_H
#define HPP_GUARD_ARGMIN_DETAIL_LSI_H

// Inequality-constrained least squares (LSI).
//
// Solves: min ||E*x - f||^2  s.t.  G*x >= h
//
// Reduces the inequality-constrained LS problem to an LDP (least
// distance) problem via a QR factorization of E. The LDP is then solved
// by the dual NNLS cascade in detail::ldp.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Prentice-Hall. Chapter 23, Section 23.5 (LSI).

#include "argmin/detail/ldp.h"
#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <limits>
#include <cmath>

namespace argmin::detail
{

// Persistent workspace for lsi() so the routine does not allocate per
// call. Sized once via resize() at the maximum problem shape; subsequent
// calls at smaller shapes reuse the existing storage.
template <typename Scalar>
struct lsi_workspace
{
    using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Vector<Scalar, Eigen::Dynamic>;

    matrix_t E_dyn;
    Eigen::ColPivHouseholderQR<matrix_t> qr;
    matrix_t R;
    vector_t y1;
    matrix_t G_perm;
    matrix_t G_tT;
    matrix_t G_t;
    vector_t h_t;
    vector_t x_prime;
    vector_t rhs;
    vector_t y;
    vector_t ldp_lambda;  // Inequality multipliers from inner LDP solve.

    void resize(int n, int m_ineq)
    {
        E_dyn.resize(n, n);
        R.resize(n, n);
        y1.resize(n);
        G_perm.resize(m_ineq, n);
        G_tT.resize(n, m_ineq);
        G_t.resize(m_ineq, n);
        h_t.resize(m_ineq);
        x_prime.resize(n);
        rhs.resize(n);
        y.resize(n);
        ldp_lambda.resize(m_ineq);
    }
};

// Solve min ||E*x - f||^2 s.t. G*x >= h.
//
// E is (n x n), f is (n), G is (m_ineq x n), h is (m_ineq), x is (n).
// E and f are read and may be modified by the routine.
//
// The lsi_workspace ws holds matrix / vector / QR factorization state
// used by lsi itself; the caller owns it and is responsible for sizing
// it via ws.resize() at problem configuration time. The nnls_* scratch
// buffers are passed through to the underlying LDP solver; the caller
// must also ensure they can hold the dual NNLS problem (see ldp()
// documentation).
//
// lambda_ineq is (m_ineq) output: Lagrange multipliers for the inequality
// constraints at the LSI optimum, lambda_ineq >= 0, complementary with
// the active set of G x >= h. The LSI inequality multipliers equal the
// inner LDP multipliers because the QR(E) transformation does not
// reshuffle constraint rows -- the change of variables x -> x' is
// applied to the LS objective only.
//
// Return code: 1 on success, 4 if the inequality system is incompatible,
// 6 if E is rank deficient.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Ch. 23.5, Algorithm LSI; eq. 23.18 for KKT lambda.
template <typename Scalar, int N>
int lsi(
    Eigen::Matrix<Scalar, N, N>& E,
    Eigen::Vector<Scalar, N>& f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& G,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& h,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& lambda_ineq,
    lsi_workspace<Scalar>& ws,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& nnls_A,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_b,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_x_vec,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_w,
    int n, int m_ineq)
{
    // QR call site dispatch note.
    //
    // The caller ABI keeps E as Matrix<Scalar, N, N> so fixed-N callers
    // benefit from compile-time storage on the input. For the column-
    // pivoting Householder QR itself, however, we copy E into a dynamic
    // buffer (ws.E_dyn) and factor the copy. At runtime N=6 this routes
    // through Eigen's dynamic-size ColPivHouseholderQR kernel, which
    // profiles measurably faster than the fixed-size 6x6 specialization
    // on the downstream workloads this routine is called from.
    //
    // Downstream quantities (R, y1, x_prime, rhs, y) live in the
    // workspace as dynamic vectors; the back-transform writes the final
    // result into the fixed-size x output.
    //
    // Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least
    //            Squares Problems. Ch. 23.5, Algorithm LSI.

    // Copy E into the dynamic-dimension workspace; QR factorizes in place.
    if(ws.E_dyn.rows() != n || ws.E_dyn.cols() != n) ws.E_dyn.resize(n, n);
    ws.E_dyn = E;
    ws.qr.compute(ws.E_dyn);

    const Scalar thresh = std::numeric_limits<Scalar>::epsilon()
                          * Scalar(n)
                          * ws.qr.maxPivot();
    ws.qr.setThreshold(thresh);
    const int rank = static_cast<int>(ws.qr.rank());
    if(rank < n)
    {
        // E must be full rank for LSI to reduce to an LDP cleanly. A
        // rank-deficient E in this context indicates a degenerate
        // Hessian factorization.
        return 6;
    }

    // R is the upper triangular part of qr.matrixR(). E is square
    // (n = N at the template level), so matrixR() is already n x n
    // and we don't need a topLeftCorner crop.
    if(ws.R.rows() != n || ws.R.cols() != n) ws.R.resize(n, n);
    ws.R = ws.qr.matrixR().template triangularView<Eigen::Upper>();

    // y1 = Q^T f.
    if(ws.y1.size() != n) ws.y1.resize(n);
    ws.y1.noalias() = ws.qr.matrixQ().transpose() * f;

    // Unconstrained case: solution in transformed coordinates is
    // x' = 0, so x = P * R^{-1} * y1.
    if(m_ineq <= 0)
    {
        if(ws.y.size() != n) ws.y.resize(n);
        ws.y = ws.R.template triangularView<Eigen::Upper>().solve(ws.y1);
        x.noalias() = ws.qr.colsPermutation() * ws.y;
        if(lambda_ineq.size() != 0) lambda_ineq.setZero();
        return 1;
    }

    if(lambda_ineq.size() != m_ineq) lambda_ineq.resize(m_ineq);
    lambda_ineq.setZero();

    // Transform inequality constraints into the x' coordinate system.
    //
    //   x' = R * (P^T x) - y1
    //   G x >= h  <=>  (G P R^{-1}) x' >= h - (G P R^{-1}) y1
    //
    // Compute G_t = G * P * R^{-1} by solving R^T * G_t^T = (G P)^T.
    if(ws.G_perm.rows() != m_ineq || ws.G_perm.cols() != n)
        ws.G_perm.resize(m_ineq, n);
    ws.G_perm.noalias() = G * ws.qr.colsPermutation();

    if(ws.G_tT.rows() != n || ws.G_tT.cols() != m_ineq)
        ws.G_tT.resize(n, m_ineq);
    ws.G_tT = ws.R.transpose()
                  .template triangularView<Eigen::Lower>()
                  .solve(ws.G_perm.transpose());

    if(ws.G_t.rows() != m_ineq || ws.G_t.cols() != n)
        ws.G_t.resize(m_ineq, n);
    ws.G_t = ws.G_tT.transpose();

    if(ws.h_t.size() != m_ineq) ws.h_t.resize(m_ineq);
    ws.h_t.noalias() = h - ws.G_t * ws.y1;

    // Solve LDP: min ||x'||^2 s.t. G_t x' >= h_t.
    if(ws.x_prime.size() != n) ws.x_prime.resize(n);
    ws.x_prime.setZero();

    if(ws.ldp_lambda.size() != m_ineq) ws.ldp_lambda.resize(m_ineq);
    int ldp_mode = ldp<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
        ws.G_t, ws.h_t, ws.x_prime, ws.ldp_lambda,
        nnls_A, nnls_b, nnls_x_vec, nnls_w,
        m_ineq, n);

    // LSI inequality multipliers = LDP multipliers (constraint rows
    // unaffected by the QR(E) variable change). Forward immediately so
    // the caller has them even if the LDP fallback below recovers x.
    lambda_ineq = ws.ldp_lambda;

    if(ldp_mode != 1)
    {
        // Check whether x_prime = 0 is already feasible for the
        // transformed constraints (this is how the "origin is feasible"
        // degenerate LDP case is resolved by the LSI caller).
        bool feasible = true;
        const Scalar feas_eps = std::numeric_limits<Scalar>::epsilon()
                                * Scalar(n) * Scalar(10);
        for(int i = 0; i < m_ineq; ++i)
        {
            if(ws.h_t[i] > feas_eps)
            {
                feasible = false;
                break;
            }
        }
        if(!feasible)
            return 4;
        ws.x_prime.setZero();
    }

    // Back-transform: x = P * R^{-1} * (x' + y1).
    if(ws.rhs.size() != n) ws.rhs.resize(n);
    ws.rhs.noalias() = ws.x_prime + ws.y1;

    if(ws.y.size() != n) ws.y.resize(n);
    ws.y = ws.R.template triangularView<Eigen::Upper>().solve(ws.rhs);
    x.noalias() = ws.qr.colsPermutation() * ws.y;

    return 1;
}

}

#endif
