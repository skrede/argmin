#ifndef HPP_GUARD_NABLAPP_DETAIL_LSEI_H
#define HPP_GUARD_NABLAPP_DETAIL_LSEI_H

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

#include "nablapp/detail/lsi.h"
#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <limits>
#include <cmath>

namespace nablapp::detail
{

// Solve min ||E*x - f||^2 s.t. C*x = d, G*x >= h.
//
// C is (m_eq x n), d is (m_eq), E is (n x n), f is (n),
// G is (m_ineq x n), h is (m_ineq), x is (n) output.
//
// Return code:
//   1 on success,
//   4 if the inequality system is incompatible,
//   6 if the equality matrix C is rank deficient.
//
// Reference: Kraft 1988 DFVLR-FB 88-28, Section 3.2, eq. 3.19-3.21.
//            Lawson & Hanson 1974, Ch. 23.6.
template <typename Scalar, int N>
int lsei(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& C,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& d,
    Eigen::Matrix<Scalar, N, N>& E,
    Eigen::Vector<Scalar, N>& f,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& G,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& h,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& nnls_A,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_b,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_x_vec,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_w,
    int n, int m_eq, int m_ineq)
{
    using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Vector<Scalar, Eigen::Dynamic>;

    // ------------------------------------------------------------------
    // Pure LSI fallback: no equality constraints.
    // ------------------------------------------------------------------
    if(m_eq <= 0)
    {
        return lsi<Scalar, N>(
            E, f, G, h, x, nnls_A, nnls_b, nnls_x_vec, nnls_w, n, m_ineq);
    }

    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    // ------------------------------------------------------------------
    // QR factorize C^T = Q_C * R_C.
    //
    // Then with x = Q_C * y the equality C*x = d becomes
    //   C * Q_C * y = R_C^T * y = d  (acting on the leading m_eq entries)
    // which is a lower-triangular system for y1 = y(0..m_eq-1).
    // ------------------------------------------------------------------
    Eigen::HouseholderQR<matrix_t> qr_c(C.transpose());
    matrix_t Qc = qr_c.householderQ() * matrix_t::Identity(n, n);

    // R_c is m_eq x m_eq upper triangular, extracted from qr_c.matrixQR().
    matrix_t R_c_upper = qr_c.matrixQR().topLeftCorner(m_eq, m_eq)
                              .template triangularView<Eigen::Upper>();

    // Check rank via diagonal of R_c.
    for(int i = 0; i < m_eq; ++i)
    {
        if(std::abs(R_c_upper(i, i)) <= eps * Scalar(n) * Scalar(10))
            return 6;
    }

    // Solve R_c^T * y1 = d for y1 (lower triangular solve).
    vector_t y1 = R_c_upper.transpose()
                           .template triangularView<Eigen::Lower>()
                           .solve(d);

    // ------------------------------------------------------------------
    // Change variables: E_tilde = E * Q_c, split into [E1 (n x m_eq),
    // E2 (n x (n - m_eq))]. Similarly for G.
    // ------------------------------------------------------------------
    matrix_t E_tilde = E * Qc;
    matrix_t E1 = E_tilde.leftCols(m_eq);
    matrix_t E2 = E_tilde.rightCols(n - m_eq);

    vector_t f_red = f - E1 * y1;

    // ------------------------------------------------------------------
    // Reduced LS problem in the free variables y2 (length n - m_eq):
    //   min ||E2 * y2 - f_red||^2  s.t.  (G * Qc)_{:,m_eq..} y2
    //                                    >= h - (G * Qc)_{:,0..m_eq-1} y1
    //
    // E2 has more rows than columns; compress via a QR factorization so
    // the inner LSI call receives a square (n-m_eq) x (n-m_eq) system.
    // ------------------------------------------------------------------
    const int n2 = n - m_eq;

    matrix_t G_tilde;
    vector_t h_red;
    if(m_ineq > 0)
    {
        G_tilde = G * Qc;
        h_red = h - G_tilde.leftCols(m_eq) * y1;
    }

    // y2 solution vector in the reduced problem.
    vector_t y2 = vector_t::Zero(n2);

    if(n2 > 0)
    {
        // QR of E2 (n x n2). E2 = Q_e * [R_e; 0] with R_e of size n2 x n2.
        Eigen::HouseholderQR<matrix_t> qr_e(E2);
        matrix_t R_e = qr_e.matrixQR().topLeftCorner(n2, n2)
                                      .template triangularView<Eigen::Upper>();

        vector_t Qt_f = qr_e.householderQ().transpose() * f_red;
        vector_t f_small = Qt_f.head(n2);

        if(m_ineq <= 0)
        {
            // Pure equality-constrained LS: solve R_e y2 = f_small.
            // Check R_e rank.
            for(int i = 0; i < n2; ++i)
            {
                if(std::abs(R_e(i, i)) <= eps * Scalar(n) * Scalar(10))
                    return 6;
            }
            y2 = R_e.template triangularView<Eigen::Upper>().solve(f_small);
        }
        else
        {
            // Build a square n2 x n2 E for the inner LSI call.
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> E_small = R_e;
            Eigen::Vector<Scalar, Eigen::Dynamic> f_small_vec = f_small;
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> G_small =
                G_tilde.rightCols(n2);

            Eigen::Vector<Scalar, Eigen::Dynamic> y2_dyn =
                Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(n2);

            int lsi_mode = lsi<Scalar, Eigen::Dynamic>(
                E_small, f_small_vec, G_small, h_red, y2_dyn,
                nnls_A, nnls_b, nnls_x_vec, nnls_w, n2, m_ineq);

            if(lsi_mode != 1)
                return lsi_mode;

            y2 = y2_dyn;
        }
    }

    // ------------------------------------------------------------------
    // Assemble y = [y1; y2] and back-transform x = Q_c * y.
    // ------------------------------------------------------------------
    vector_t y(n);
    y.head(m_eq) = y1;
    if(n2 > 0)
        y.tail(n2) = y2;

    x.noalias() = Qc * y;

    return 1;
}

}

#endif
