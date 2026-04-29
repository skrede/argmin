#ifndef HPP_GUARD_NABLAPP_DETAIL_NNLS_H
#define HPP_GUARD_NABLAPP_DETAIL_NNLS_H

// Lawson-Hanson non-negative least squares with incremental QR.
//
// Solves: min ||A*x - b||^2  s.t.  x >= 0
//
// Active-set iteration over a partition of the column index set into a
// "positive" set P (strictly positive components) and a "zero" set Z
// (bound-active at zero). On each outer step the largest positive dual
// variable in Z is swapped into P; the unconstrained LS subproblem
// restricted to P is solved by back-substitution on the maintained
// upper-triangular factor R; inner ratio tests move newly negative
// components back to Z.
//
// The factorization Q^T A = [R_P; 0] (with the columns of A permuted by
// indx so the active-P columns sit in the leading positions) is
// maintained incrementally:
//
//   * column-add: one Householder reflector applied to the rows below
//     the current pivot row, applied across all not-yet-active columns
//     and to the running b vector. O(m * |Z|) work.
//
//   * column-drop: shift indx leftward, then sweep Givens rotations
//     across the resulting upper-Hessenberg shape to restore upper
//     triangular form. O(nsetp^2) work in the worst case.
//
// vs. the prior implementation that recomputed `A_P.householderQr()`
// from scratch on every inner iteration.
//
// References:
//   Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares Problems.
//   Prentice-Hall. Chapter 23, Section 23.3 (NNLS).
//   Algorithm uses Householder for column-add and Givens for column-drop;
//   both are standard QR update primitives -- see Golub & Van Loan,
//   Matrix Computations 4e Algorithm 12.5.4 (Householder QR update) and
//   Algorithm 5.1.3 (Givens rotation).

#include "nablapp/detail/givens_qr_update.h"
#include "nablapp/types.h"

#include <Eigen/Core>

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
// A is (m x n), b is (m), x is (n), w is (n) dual vector. A and b may be
// modified by the routine; the caller must not rely on their post-call
// contents. x is set to the primal solution; w holds the final dual
// (w_j <= 0 for all j in Z at convergence).
//
// The iteration limit is fixed at 3*n, matching Lawson & Hanson and the
// NLopt / SciPy convention; on exhaustion the routine returns mode=3.
//
// Workspace: thread_local maintained-factorization buffers eliminate the
// per-inner-iter heap allocation of the prior fresh-QR path.
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

    // Maintained-factorization workspaces. A_buf holds Q^T A (rows below
    // active pivots are zero on P columns; nonzero on Z columns). b_buf
    // holds Q^T b. indx is the column permutation: positions [0, nsetp)
    // are P (in addition order), [nsetp, n) are Z.
    thread_local Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> b_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> z_buf;
    thread_local Eigen::Vector<int, Eigen::Dynamic> indx_buf;
    thread_local Eigen::Vector<int, Eigen::Dynamic> in_P_buf;
    thread_local Eigen::Vector<Scalar, Eigen::Dynamic> hh_buf;

    if(A_buf.rows() < m || A_buf.cols() < n) A_buf.resize(m, n);
    if(b_buf.size() < m) b_buf.resize(m);
    if(z_buf.size() < n) z_buf.resize(n);
    if(indx_buf.size() < n) indx_buf.resize(n);
    if(in_P_buf.size() < n) in_P_buf.resize(n);
    if(hh_buf.size() < m) hh_buf.resize(m);

    auto A_factored = A_buf.topLeftCorner(m, n);
    auto b_factored = b_buf.head(m);
    A_factored = A;
    b_factored = b;

    auto z = z_buf.head(n);
    auto indx = indx_buf.head(n);
    auto in_P = in_P_buf.head(n);

    x.setZero();
    w.setZero();
    in_P.setZero();
    for(int j = 0; j < n; ++j)
        indx[j] = j;
    int nsetp = 0;

    int iter = 0;
    for(; iter < max_iter; ++iter)
    {
        // --- Step 1-3: dual w over the zero set Z, find argmax. -----
        //
        // For j in P, the maintained factorization gives
        //   A_factored.col(j).tail(m - nsetp) == 0 exactly,
        // so the formula A_factored.col(j).tail.dot(b_factored.tail)
        // produces 0 for in-P columns automatically (modulo roundoff).
        // Z columns retain nonzero tails and yield the genuine dual.
        //
        // The first-block contribution A_factored.col(j).head(nsetp)
        // .dot(r_factored.head(nsetp)) is suppressed because at outer-
        // iter entry x is back-substituted (R x_p == b_factored.head),
        // so r_factored.head ~= 0 to roundoff.
        Scalar wmax = Scalar(0);
        int t = -1;
        for(int j = 0; j < n; ++j)
        {
            if(in_P[j])
            {
                w[j] = Scalar(0);
                continue;
            }
            const Scalar wj = A_factored.col(j).segment(nsetp, m - nsetp)
                                                .dot(b_factored.segment(nsetp, m - nsetp));
            w[j] = wj;
            if(wj > wmax)
            {
                wmax = wj;
                t = j;
            }
        }

        // KKT satisfied: all duals on Z are non-positive (within tol).
        if(t < 0 || wmax <= eps * Scalar(n))
        {
            result.mode = 1;
            break;
        }

        // --- Step 4-5: add column t to P via Householder reflector. ----
        //
        // Find t's position in indx (must be in [nsetp, n)). Compute the
        // Householder reflector H = I - tau v v^T that zeros rows
        // [nsetp+1, m) of A_factored.col(t) (acting on the segment
        // [nsetp, m)). Apply H to all Z-set columns and to b_factored.
        //
        // A relative-pivot stability check (column tail must contribute
        // a non-negligible fraction of the column's full norm) catches
        // the case where t is numerically dependent on the existing P
        // columns; in that rare case we zero w[t] and re-loop the
        // outer iteration to find a different argmax.
        int t_pos = -1;
        for(int k = nsetp; k < n; ++k)
        {
            if(indx[k] == t)
            {
                t_pos = k;
                break;
            }
        }

        const int tail_len = m - nsetp;
        const Scalar full_col_norm =
            A_factored.col(t).head(m).norm();
        const Scalar tail_col_norm =
            A_factored.col(t).segment(nsetp, tail_len).norm();

        // If the tail is dominated by the head (column dependent on P),
        // skip it: zero w[t], swap it temporarily out of contention by
        // restarting the outer loop. The eps factor matches NLopt's
        // h12_ skip threshold (slsqp.c h12_ "alpha=0.01*pivot" style).
        if(tail_col_norm <= eps * full_col_norm)
        {
            // Mark this column un-pickable for the rest of the run by
            // setting in_P[t] = 1 (it can't actually be added since it's
            // dependent, but the dual check uses in_P to skip Z scans).
            // This matches NLopt's behavior on degenerate columns.
            in_P[t] = 1;
            // Don't increment nsetp; this column never enters P
            // numerically. We still need to remove it from the live
            // search, which the in_P guard does on the next outer iter.
            continue;
        }

        // Compute Householder for the tail segment of column t.
        // Eigen API: makeHouseholderInPlace produces (essential, tau,
        // beta) such that H * tail = beta * e_0 and the essential
        // (rows [1, tail_len) of v) is stored in-place; the diagonal
        // entry is overwritten with beta.
        auto tail_t = A_factored.col(t).segment(nsetp, tail_len);
        Scalar tau = Scalar(0);
        Scalar beta = Scalar(0);
        // Build essential into hh_buf so we can apply to other columns.
        // makeHouseholder writes the essential vector to its first
        // argument (length tail_len - 1) and the (tau, beta) outputs.
        if(tail_len > 1)
        {
            auto essential = hh_buf.head(tail_len - 1);
            tail_t.makeHouseholder(essential, tau, beta);
            // Zero the column in-place: pivot at row nsetp = beta,
            // rows [nsetp+1, m) = 0.
            tail_t[0] = beta;
            tail_t.tail(tail_len - 1).setZero();

            // Apply H to all other Z-set columns at rows [nsetp, m).
            for(int k = nsetp; k < n; ++k)
            {
                const int j = indx[k];
                if(j == t) continue;
                auto col_seg = A_factored.col(j).segment(nsetp, tail_len);
                col_seg.applyHouseholderOnTheLeft(essential, tau, hh_buf.data() + (tail_len - 1));
            }

            // Apply H to b_factored at rows [nsetp, m).
            auto b_seg = b_factored.segment(nsetp, tail_len);
            b_seg.applyHouseholderOnTheLeft(essential, tau, hh_buf.data() + (tail_len - 1));
        }
        else
        {
            // tail_len == 1: Householder is identity; just record the
            // pivot value (no transformation needed).
            // tail_t[0] is already the pivot value.
        }

        // Move t into the leading P block at position nsetp.
        std::swap(indx[t_pos], indx[nsetp]);
        in_P[t] = 1;
        nsetp += 1;

        // --- Inner ratio loop: enforce non-negativity on P. ------------
        int inner_iter = 0;
        const int inner_max = 3 * n;
        bool inner_done = false;
        while(!inner_done && inner_iter++ < inner_max)
        {
            // Step 6: back-substitute z_p = R^{-1} b_factored.head(nsetp).
            // R is the upper-triangular block A_factored(i, indx[k]) for
            // 0 <= i <= k < nsetp.
            for(int k = 0; k < nsetp; ++k)
                z[indx[k]] = b_factored[k];
            for(int k = nsetp - 1; k >= 0; --k)
            {
                const int jk = indx[k];
                Scalar s = z[jk];
                for(int kp = k + 1; kp < nsetp; ++kp)
                    s -= A_factored(k, indx[kp]) * z[indx[kp]];
                z[jk] = s / A_factored(k, jk);
            }
            // Z components of z must be 0 for the ratio test below.
            for(int k = nsetp; k < n; ++k)
                z[indx[k]] = Scalar(0);

            // Step 7-10: ratio test on z over P.
            Scalar alpha = std::numeric_limits<Scalar>::max();
            int q = -1;
            for(int k = 0; k < nsetp; ++k)
            {
                const int j = indx[k];
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
                // No blocking constraint: accept z and exit inner loop.
                for(int k = 0; k < nsetp; ++k)
                    x[indx[k]] = z[indx[k]];
                for(int k = nsetp; k < n; ++k)
                    x[indx[k]] = Scalar(0);
                inner_done = true;
                break;
            }

            // Step 11: interpolate x <- x + alpha (z - x), drop columns
            // whose primal collapsed to ~0. We process drops in
            // descending order of position so each Givens sweep is
            // independent of subsequent drops.
            for(int j = 0; j < n; ++j)
                x[j] += alpha * (z[j] - x[j]);

            for(int drop_pos = nsetp - 1; drop_pos >= 0; --drop_pos)
            {
                const int j_drop = indx[drop_pos];
                if(x[j_drop] > eps * Scalar(10))
                    continue;

                // Drop column at drop_pos. Shift indx[drop_pos+1..n-1]
                // left by 1; push the dropped index into the tail
                // position of indx (so it lives in Z).
                in_P[j_drop] = 0;
                x[j_drop] = Scalar(0);
                for(int k = drop_pos; k < n - 1; ++k)
                    indx[k] = indx[k + 1];
                indx[n - 1] = j_drop;

                // Givens sweep: zero the spike A_factored(i+1, indx[i])
                // for i = drop_pos to nsetp-2 (using nsetp BEFORE the
                // decrement below; range collapses to empty when
                // drop_pos == nsetp-1).
                for(int i = drop_pos; i < nsetp - 1; ++i)
                {
                    const int j_pivot = indx[i];
                    auto g = givens_rotation<Scalar>::compute(
                        A_factored(i, j_pivot),
                        A_factored(i + 1, j_pivot));

                    // Apply to all n columns at rows i, i+1.
                    g.apply_left(A_factored, i, i + 1);

                    // Apply to b_factored at rows i, i+1.
                    Scalar bi = b_factored[i];
                    Scalar bi1 = b_factored[i + 1];
                    b_factored[i]     =  g.c * bi + g.s * bi1;
                    b_factored[i + 1] = -g.s * bi + g.c * bi1;
                }

                nsetp -= 1;
            }

            if(nsetp == 0)
            {
                inner_done = true;
                break;
            }
        }
    }

    result.iterations = iter;
    if(iter >= max_iter)
        result.mode = 3;

    // Final residual norm on the original system equals the norm of
    // the residual portion in the factored frame (Q is orthogonal):
    //   |r_orig| = |Q r_factored| = |r_factored|, and at convergence
    //   r_factored = [0; b_factored.tail(m - nsetp)].
    result.residual_norm = b_factored.segment(nsetp, m - nsetp).norm();

    // Recompute final dual w on the factored system. For j in P the
    // tail block is zero by construction, so w[j] = 0. For j in Z the
    // formula yields A_orig^T (b_orig - A_orig x).
    for(int j = 0; j < n; ++j)
    {
        w[j] = A_factored.col(j).segment(nsetp, m - nsetp)
                                 .dot(b_factored.segment(nsetp, m - nsetp));
    }

    return result;
}

}

#endif
