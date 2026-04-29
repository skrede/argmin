#ifndef HPP_GUARD_NABLAPP_DETAIL_NNLS_H
#define HPP_GUARD_NABLAPP_DETAIL_NNLS_H

// Lawson-Hanson non-negative least squares.
//
// Solves: min ||A*x - b||^2  s.t.  x >= 0
//
// Active-set iteration over the partition of the column index set into a
// "positive" set P (strictly positive components) and a "zero" set Z
// (bound-active at zero). On each outer step the largest positive dual
// variable in Z is swapped into P and the unconstrained LS subproblem
// restricted to the columns in P is solved; inner ratio tests move newly
// negative components back to Z.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Prentice-Hall. Chapter 23, Section 23.3 (NNLS).

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <limits>
#include <cmath>

namespace nablapp::detail
{

template <typename Scalar>
struct nnls_result
{
    int mode{1};             // 1 = converged, 2 = bad dimensions, 3 = max iterations
    Scalar residual_norm{};
    int iterations{};
};

// Solve min ||A*x - b||^2 s.t. x >= 0.
//
// A is (m x n), b is (m), x is (n), w is (n) dual vector.
// A and b may be modified by the routine; the caller should not rely on
// their post-call contents. x is set to the solution, w holds the final
// dual vector (w_j <= 0 for all j in the zero set at convergence).
//
// The iteration limit is fixed at 3*n, matching Lawson & Hanson and the
// NLopt / SciPy convention; on exhaustion the routine returns mode=3.
//
// Per-call workspace: r, A_p, z_p, p_list, in_P live in thread_local
// storage to eliminate the per-inner-iter heap allocation that the
// prior `Eigen::Matrix Ap(m, nsetp)` + `Ap.householderQr().solve()`
// path triggered. NNLS is invoked from inside LDP -> LSI / LSEI on
// every QP solve in kraft_slsqp; the allocator churn was a measurable
// fraction of per-step wall time on small constrained HS problems.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Ch. 23.3, Algorithm NNLS.
template <typename Scalar, int M, int N>
nnls_result<Scalar> nnls(
    Eigen::Matrix<Scalar, M, N>& A,
    Eigen::Vector<Scalar, M>& b,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Vector<Scalar, N>& w,
    int m, int n)
{
    using std::abs;
    using std::sqrt;

    nnls_result<Scalar> result;
    if(m <= 0 || n <= 0)
    {
        result.mode = 2;
        return result;
    }

    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const int max_iter = 3 * n;

    // Thread-local workspaces. Sized lazily on first use; subsequent
    // calls at smaller shapes reuse the existing storage. The QR
    // factorization object is also hoisted so its internal Householder
    // tau / hCoeffs storage is reused.
    thread_local Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A_buf;
    thread_local Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Ap_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> b_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> r_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> z_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> zp_buf;
    thread_local Eigen::Vector<int, Eigen::Dynamic> in_P_buf;
    thread_local Eigen::Vector<int, Eigen::Dynamic> p_list_buf;
    thread_local Eigen::HouseholderQR<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> qr;

    if(A_buf.rows() < m || A_buf.cols() < n) A_buf.resize(m, n);
    if(Ap_buf.rows() < m || Ap_buf.cols() < n) Ap_buf.resize(m, n);
    if(b_buf.size() < m) b_buf.resize(m);
    if(r_buf.size() < m) r_buf.resize(m);
    if(z_buf.size() < n) z_buf.resize(n);
    if(zp_buf.size() < n) zp_buf.resize(n);
    if(in_P_buf.size() < n) in_P_buf.resize(n);
    if(p_list_buf.size() < n) p_list_buf.resize(n);

    auto A_ref = A_buf.topLeftCorner(m, n);
    auto b_ref = b_buf.head(m);
    A_ref = A;
    b_ref = b;

    auto z = z_buf.head(n);
    auto in_P = in_P_buf.head(n);
    auto p_list = p_list_buf.head(n);

    x.setZero();
    w.setZero();
    in_P.setZero();
    int nsetp = 0;

    int iter = 0;
    for(; iter < max_iter; ++iter)
    {
        // Compute residual r = b - A x and dual w = A^T r.
        auto r = r_buf.head(m);
        r.noalias() = b_ref - A_ref * x;
        w.noalias() = A_ref.transpose() * r;

        // Find argmax of w over Z.
        int t = -1;
        Scalar wmax = Scalar(0);
        for(int j = 0; j < n; ++j)
        {
            if(in_P[j]) continue;
            if(w[j] > wmax)
            {
                wmax = w[j];
                t = j;
            }
        }

        // KKT satisfied: all duals on Z are non-positive.
        if(t < 0 || wmax <= eps * Scalar(n))
        {
            result.mode = 1;
            break;
        }

        // Move index t from Z to P.
        in_P[t] = 1;
        p_list[nsetp++] = t;

        // Inner ratio-test loop: enforce non-negativity on P.
        int inner_iter = 0;
        const int inner_max = 3 * n;
        while(inner_iter++ < inner_max)
        {
            // Solve LS for the current P: min || A_P z_P - b ||.
            // Build Ap into the hoisted workspace via column-gather.
            for(int k = 0; k < nsetp; ++k)
                Ap_buf.col(k).head(m) = A_ref.col(p_list[k]);
            qr.compute(Ap_buf.topLeftCorner(m, nsetp));
            zp_buf.head(nsetp) = qr.solve(b_ref);

            // Scatter zp into z using the full n layout.
            z.setZero();
            for(int k = 0; k < nsetp; ++k)
                z[p_list[k]] = zp_buf[k];

            // Check if all P components of z are strictly positive.
            Scalar alpha = std::numeric_limits<Scalar>::max();
            int q = -1;
            for(int k = 0; k < nsetp; ++k)
            {
                const int j = p_list[k];
                if(z[j] <= Scalar(0))
                {
                    const Scalar denom = x[j] - z[j];
                    if(denom > eps)
                    {
                        const Scalar ratio = x[j] / denom;
                        if(ratio < alpha)
                        {
                            alpha = ratio;
                            q = k;
                        }
                    }
                }
            }

            if(q < 0)
            {
                // All positive: accept zP, clear Z components of x.
                for(int j = 0; j < n; ++j)
                    x[j] = in_P[j] ? z[j] : Scalar(0);
                break;
            }

            // Interpolate: x <- x + alpha*(z - x).
            for(int j = 0; j < n; ++j)
                x[j] += alpha * (z[j] - x[j]);

            // Drop every P column whose x is now <= eps.
            int write = 0;
            for(int k = 0; k < nsetp; ++k)
            {
                const int j = p_list[k];
                if(x[j] <= eps * Scalar(10))
                {
                    in_P[j] = 0;
                    x[j] = Scalar(0);
                }
                else
                {
                    p_list[write++] = j;
                }
            }
            nsetp = write;

            if(nsetp == 0)
                break;
        }
    }

    result.iterations = iter;
    if(iter >= max_iter)
        result.mode = 3;

    // Final residual norm on the ORIGINAL system.
    auto r_final = r_buf.head(m);
    r_final.noalias() = b_ref - A_ref * x;
    result.residual_norm = r_final.norm();

    // Recompute dual on the final x for the caller.
    w.noalias() = A_ref.transpose() * r_final;

    return result;
}

}

#endif
