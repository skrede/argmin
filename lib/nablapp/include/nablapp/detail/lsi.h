#ifndef HPP_GUARD_NABLAPP_DETAIL_LSI_H
#define HPP_GUARD_NABLAPP_DETAIL_LSI_H

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

#include "nablapp/detail/ldp.h"
#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <limits>
#include <cmath>

namespace nablapp::detail
{

// Solve min ||E*x - f||^2 s.t. G*x >= h.
//
// E is (n x n), f is (n), G is (m_ineq x n), h is (m_ineq), x is (n).
// E and f are read and may be modified by the routine.
//
// The nnls_* scratch buffers are passed through to the underlying LDP
// solver; the caller must ensure they can hold the dual NNLS problem
// (see ldp() documentation).
//
// Return code: 1 on success, 4 if the inequality system is incompatible,
// 6 if E is rank deficient.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Ch. 23.5, Algorithm LSI.
template <typename Scalar, int N>
int lsi(
    Eigen::Matrix<Scalar, N, N>& E,
    Eigen::Vector<Scalar, N>& f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& G,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& h,
    Eigen::Vector<Scalar, N>& x,
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
    // pivoting Householder QR itself, however, we copy E into a local
    // Matrix<Scalar, Dynamic, Dynamic> and factor the copy. At runtime
    // N=6 this routes through Eigen's dynamic-size ColPivHouseholderQR
    // kernel, which profiles measurably faster than the fixed-size 6x6
    // specialization on the downstream workloads this routine is called
    // from. The copy is one cache line (36 doubles = 288 B at N=6) and
    // does not show up as a measurable cost in any micro-benchmark.
    //
    // Downstream quantities (R, y1, x_prime, rhs, y) stay on the
    // fixed-size aliases so the solve, back-transform, and constraint
    // projection continue to use the compile-time kernels.
    //
    // Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least
    //            Squares Problems. Ch. 23.5, Algorithm LSI.
    using E_matrix_t = Eigen::Matrix<Scalar, N, N>;
    using QR_matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using N_vector_t = Eigen::Vector<Scalar, N>;
    using ineq_matrix_N_cols_t = Eigen::Matrix<Scalar, Eigen::Dynamic, N>;
    using ineq_matrix_N_rows_t = Eigen::Matrix<Scalar, N, Eigen::Dynamic>;
    using dyn_vector_t = Eigen::Vector<Scalar, Eigen::Dynamic>;

    // Copy E into a dynamic-dimension local; QR factorizes the copy.
    QR_matrix_t E_dyn = E;
    Eigen::ColPivHouseholderQR<QR_matrix_t> qr(E_dyn);

    const Scalar thresh = std::numeric_limits<Scalar>::epsilon()
                          * Scalar(n)
                          * qr.maxPivot();
    qr.setThreshold(thresh);
    const int rank = static_cast<int>(qr.rank());
    if(rank < n)
    {
        // E must be full rank for LSI to reduce to an LDP cleanly. A
        // rank-deficient E in this context indicates a degenerate
        // Hessian factorization.
        return 6;
    }

    // R is the upper triangular part of qr.matrixR(). E is square
    // (n = N at the template level), so matrixR() is already N x N
    // and we don't need a topLeftCorner crop.
    E_matrix_t R = qr.matrixR().template triangularView<Eigen::Upper>();

    // y1 = Q^T f.
    N_vector_t y1 = qr.matrixQ().transpose() * f;

    // Unconstrained case: solution in transformed coordinates is
    // x' = 0, so x = P * R^{-1} * y1.
    if(m_ineq <= 0)
    {
        N_vector_t y = R.template triangularView<Eigen::Upper>().solve(y1);
        x.noalias() = qr.colsPermutation() * y;
        return 1;
    }

    // Transform inequality constraints into the x' coordinate system.
    //
    //   x' = R * (P^T x) - y1
    //   G x >= h  <=>  (G P R^{-1}) x' >= h - (G P R^{-1}) y1
    //
    // Compute G_t = G * P * R^{-1} by solving R^T * G_t^T = (G P)^T.
    // G is (m_ineq x N), so G_perm, G_tT, G_t all have compile-time
    // N in one dimension and dynamic m_ineq in the other.
    ineq_matrix_N_cols_t G_perm = G * qr.colsPermutation();
    ineq_matrix_N_rows_t G_tT = R.transpose()
                                  .template triangularView<Eigen::Lower>()
                                  .solve(G_perm.transpose());
    ineq_matrix_N_cols_t G_t = G_tT.transpose();

    dyn_vector_t h_t = h - G_t * y1;

    // Solve LDP: min ||x'||^2 s.t. G_t x' >= h_t.
    N_vector_t x_prime = N_vector_t::Zero(n);

    int ldp_mode = ldp<Scalar, Eigen::Dynamic, N>(
        G_t, h_t, x_prime, nnls_A, nnls_b, nnls_x_vec, nnls_w, m_ineq, n);

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
            if(h_t[i] > feas_eps)
            {
                feasible = false;
                break;
            }
        }
        if(!feasible)
            return 4;
        x_prime.setZero();
    }

    // Back-transform: x = P * R^{-1} * (x' + y1).
    N_vector_t rhs = x_prime + y1;
    N_vector_t y = R.template triangularView<Eigen::Upper>().solve(rhs);
    x.noalias() = qr.colsPermutation() * y;

    return 1;
}

}

#endif
