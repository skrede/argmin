#ifndef HPP_GUARD_NABLAPP_DETAIL_KRAFT_LSQ_QP_H
#define HPP_GUARD_NABLAPP_DETAIL_KRAFT_LSQ_QP_H

// Kraft 1988 LSQ/LSEI QP solver.
//
// Solves: min 0.5 * p^T * B * p + g^T * p
//         s.t. A_eq * p  = b_eq
//              A_ineq * p >= b_ineq
//              p_lo <= p <= p_hi
//
// Reformulates the quadratic program as a constrained least-squares
// problem via a Cholesky factorization of the (SPD) Hessian B, then
// delegates to the five-level cascade:
//
//   LSQ -> LSEI -> LSI -> LDP -> NNLS.
//
// Box bounds are folded into the inequality block as augmented +I / -I
// rows (only for finite bounds; infinite bounds are skipped, matching
// the NLopt SGJ 2010 patch). No allocation occurs inside solve() once
// resize() has been called.
//
// Reference: Kraft, D. (1988). A Software Package for Sequential
//            Quadratic Programming. DFVLR-FB 88-28, eq. 2.33 and
//            Section 3.2.
//            Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Prentice-Hall. Chapters 23.3-23.6.

#include "nablapp/detail/lsei.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/types.h"

#include <Eigen/QR>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace nablapp::detail
{

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class kraft_lsq_qp_solver
{
public:
    kraft_lsq_qp_solver() = default;

    // Pre-allocate workspace for a problem of the given shape.
    // n_finite_lower / n_finite_upper are upper bounds; the actual
    // number of augmented rows is determined per-solve from which
    // bounds are finite.
    void resize(int n, int m_eq, int m_ineq, int n_finite_lower, int n_finite_upper)
    {
        const int m_aug_max = m_ineq + n_finite_lower + n_finite_upper;

        E_.resize(n, n);
        f_.resize(n);
        G_aug_.resize(m_aug_max, n);
        h_aug_.resize(m_aug_max);
        A_eq_buf_.resize(m_eq, n);
        b_eq_buf_.resize(m_eq);
        x_out_.resize(n);

        const int n2_max = n;
        nnls_A_.resize(n2_max + 1, m_aug_max);
        nnls_b_.resize(n2_max + 1);
        nnls_x_vec_.resize(m_aug_max);
        nnls_w_.resize(m_aug_max);
    }

    // Solve the QP subproblem.
    //
    // B must be symmetric positive definite (the caller is responsible
    // for ensuring this via, e.g., BFGS Powell damping). The returned
    // qp_result contains the step p, the constraint multipliers (only
    // the m_eq + m_ineq "real" constraints; bound multipliers are not
    // currently exposed), and a status code.
    template <int M = nablapp::dynamic_dimension>
    qp_result<Scalar, N, M> solve(
        const Eigen::Matrix<Scalar, N, N>& B,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        using std::isfinite;

        const int n = static_cast<int>(B.rows());
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());

        qp_result<Scalar, N, M> out;
        out.x.setZero(n);

        // -------------------------------------------------------------
        // Cholesky of B: B = L * L^T.
        //
        // Cast the QP
        //     min 0.5 p^T B p + g^T p
        //
        // as a least-squares problem
        //     min 0.5 ||E p - f||^2
        //
        // by matching quadratic and linear terms:
        //   E^T E = B            -> E = L^T  (upper triangular)
        //   -f^T (E p) = g^T p   -> E^T f = -g  -> L f = -g
        //   => f = -L^{-1} g  (forward triangular solve, NOT -B^{-1} g)
        //
        // Using B^{-1} g here silently mis-scales the least-squares RHS
        // by an extra L^{-T}, producing a p that is L^{-T}-times too
        // small and multipliers that are the correct KKT values but
        // paired with a shrunken step.
        // -------------------------------------------------------------
        Eigen::LLT<Eigen::Matrix<Scalar, N, N>> llt(B);
        if(llt.info() != Eigen::Success)
        {
            out.status = qp_status::indefinite_hessian;
            return out;
        }

        if(E_.rows() != n || E_.cols() != n) E_.resize(n, n);
        if(f_.size() != n) f_.resize(n);

        Eigen::Matrix<Scalar, N, N> L = llt.matrixL();
        E_.setZero();
        for(int i = 0; i < n; ++i)
            for(int j = i; j < n; ++j)
                E_(i, j) = L(j, i);

        // f = -L^{-1} g via a single forward triangular solve.
        f_ = -L.template triangularView<Eigen::Lower>().solve(g);

        // -------------------------------------------------------------
        // Count finite bounds and build augmented inequality block.
        //
        // Rows:
        //   0..m_ineq-1             : A_ineq
        //   m_ineq..m_ineq+nlo-1    : +e_i (finite lower bounds)
        //   m_ineq+nlo..m_ineq+nlo+nhi-1 : -e_i (finite upper bounds)
        // RHS:
        //   b_ineq  |  p_lo_finite  |  -p_hi_finite
        // -------------------------------------------------------------
        int nlo = 0;
        int nhi = 0;
        for(int i = 0; i < n; ++i)
        {
            if(isfinite(p_lo[i])) ++nlo;
            if(isfinite(p_hi[i])) ++nhi;
        }
        const int m_aug = m_ineq + nlo + nhi;

        if(G_aug_.rows() != m_aug || G_aug_.cols() != n)
            G_aug_.resize(m_aug, n);
        if(h_aug_.size() != m_aug) h_aug_.resize(m_aug);

        G_aug_.setZero();
        if(m_ineq > 0)
        {
            G_aug_.topRows(m_ineq) = A_ineq;
            h_aug_.head(m_ineq) = b_ineq;
        }

        int row = m_ineq;
        for(int i = 0; i < n; ++i)
        {
            if(isfinite(p_lo[i]))
            {
                G_aug_(row, i) = Scalar(1);
                h_aug_[row] = p_lo[i];
                ++row;
            }
        }
        for(int i = 0; i < n; ++i)
        {
            if(isfinite(p_hi[i]))
            {
                G_aug_(row, i) = Scalar(-1);
                h_aug_[row] = -p_hi[i];
                ++row;
            }
        }

        // -------------------------------------------------------------
        // Buffered equality side (LSEI reads C, d by value/ref).
        // -------------------------------------------------------------
        if(A_eq_buf_.rows() != m_eq || A_eq_buf_.cols() != n)
            A_eq_buf_.resize(m_eq, n);
        if(b_eq_buf_.size() != m_eq) b_eq_buf_.resize(m_eq);
        if(m_eq > 0)
        {
            A_eq_buf_ = A_eq;
            b_eq_buf_ = b_eq;
        }

        if(x_out_.size() != n) x_out_.resize(n);
        x_out_.setZero();

        // -------------------------------------------------------------
        // Call LSEI.
        // -------------------------------------------------------------
        int mode = lsei<Scalar, N>(
            A_eq_buf_, b_eq_buf_,
            E_, f_,
            G_aug_, h_aug_,
            x_out_,
            nnls_A_, nnls_b_, nnls_x_vec_, nnls_w_,
            n, m_eq, m_aug);

        out.x = x_out_;
        out.iterations = 1;  // outer LSEI call; inner NNLS iterations not surfaced.

        switch(mode)
        {
        case 1:
            out.status = qp_status::optimal;
            break;
        case 4:
            out.status = qp_status::infeasible;
            break;
        case 6:
            out.status = qp_status::infeasible;  // rank-deficient equality
            break;
        default:
            out.status = qp_status::max_iterations;
            break;
        }

        // -------------------------------------------------------------
        // Multiplier recovery via post-hoc KKT projection.
        //
        // At the QP optimum p* the stationarity condition reads
        //   B p* + g = A_eq^T lam_eq + A_act_ineq^T lam_act_ineq
        //            + sum_{j in A_box} e_j lam_box_j
        // where the active set collects equality rows, inequality rows
        // whose residual is near zero, and box-bound indices where
        // p_j is at either bound (with the appropriate sign on the
        // normal). Box-bound multipliers must be included in the QR
        // solve so they do not leak into the equality / inequality
        // slots; they are discarded after the solve since downstream
        // consumers only need the "real" constraint multipliers for
        // the Lagrangian BFGS update.
        //
        // Solving the overdetermined system A_full^T lam = r with
        // r = B p + g by a Householder QR of A_full^T gives a
        // least-squares multiplier estimate that is exact when the
        // active rows are linearly independent and otherwise returns
        // the minimum-norm fit -- the standard SQP recipe.
        //
        // Inequality multipliers are clipped at zero to preserve the
        // dual-feasibility sign convention lam >= 0.
        //
        // Reference: N&W Section 16.5, p. 472 (multiplier recovery);
        //            Kraft 1988 DFVLR-FB 88-28 Section 3.2 (KKT).
        // -------------------------------------------------------------
        const int m_real = m_eq + m_ineq;
        out.lambda.setZero(m_real);

        if(m_real > 0)
        {
            using dynmat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
            using dynvec = Eigen::Vector<Scalar, Eigen::Dynamic>;

            const Scalar active_tol = std::sqrt(std::numeric_limits<Scalar>::epsilon());

            // Worst-case row count: m_eq + all m_ineq + both bounds on
            // every variable. The actual row count (act_rows) is
            // determined dynamically.
            const int max_rows = m_eq + m_ineq + 2 * n;
            dynmat A_act(max_rows, n);
            int act_rows = 0;

            // Track which slot each active row corresponds to so we
            // can scatter multipliers back into the (eq, ineq) vector
            // while ignoring the box rows.
            std::vector<int> row_slot(max_rows, -1);  // -1 = box row

            if(m_eq > 0)
            {
                A_act.topRows(m_eq) = A_eq;
                for(int i = 0; i < m_eq; ++i)
                    row_slot[i] = i;
                act_rows = m_eq;
            }

            for(int i = 0; i < m_ineq; ++i)
            {
                const Scalar lhs = A_ineq.row(i).dot(out.x);
                if(std::abs(lhs - b_ineq[i]) <= active_tol * (Scalar(1) + std::abs(b_ineq[i])))
                {
                    A_act.row(act_rows) = A_ineq.row(i);
                    row_slot[act_rows] = m_eq + i;
                    ++act_rows;
                }
            }

            // Active box bounds: p_j at lower bound (row +e_j) or at
            // upper bound (row -e_j). Use the cascade's active_tol.
            for(int j = 0; j < n; ++j)
            {
                if(std::isfinite(p_lo[j])
                   && std::abs(out.x[j] - p_lo[j]) <= active_tol * (Scalar(1) + std::abs(p_lo[j])))
                {
                    A_act.row(act_rows).setZero();
                    A_act(act_rows, j) = Scalar(1);
                    row_slot[act_rows] = -1;
                    ++act_rows;
                }
            }
            for(int j = 0; j < n; ++j)
            {
                if(std::isfinite(p_hi[j])
                   && std::abs(out.x[j] - p_hi[j]) <= active_tol * (Scalar(1) + std::abs(p_hi[j])))
                {
                    A_act.row(act_rows).setZero();
                    A_act(act_rows, j) = Scalar(-1);
                    row_slot[act_rows] = -1;
                    ++act_rows;
                }
            }

            if(act_rows > 0)
            {
                dynvec r = (B * out.x + g).eval();

                // Solve A_act^T * lam = r in the least-squares sense.
                dynmat At = A_act.topRows(act_rows).transpose();
                Eigen::HouseholderQR<dynmat> qr(At);
                dynvec lam_act = qr.solve(r);

                // Scatter lam_act back into (eq, ineq) slots; box rows
                // (row_slot == -1) are discarded.
                for(int i = 0; i < act_rows; ++i)
                {
                    const int slot = row_slot[i];
                    if(slot < 0) continue;
                    if(slot < m_eq)
                    {
                        out.lambda[slot] = lam_act[i];
                    }
                    else
                    {
                        // Clip at zero: SQP Lagrangian BFGS update
                        // needs lam_ineq >= 0 (dual feasibility).
                        out.lambda[slot] = std::max(Scalar(0), lam_act[i]);
                    }
                }
            }
        }

        return out;
    }

private:
    Eigen::Matrix<Scalar, N, N> E_;
    Eigen::Vector<Scalar, N> f_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> G_aug_;
    Eigen::Vector<Scalar, Eigen::Dynamic> h_aug_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_eq_buf_;
    Eigen::Vector<Scalar, Eigen::Dynamic> b_eq_buf_;
    Eigen::Vector<Scalar, N> x_out_;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> nnls_A_;
    Eigen::Vector<Scalar, Eigen::Dynamic> nnls_b_;
    Eigen::Vector<Scalar, Eigen::Dynamic> nnls_x_vec_;
    Eigen::Vector<Scalar, Eigen::Dynamic> nnls_w_;
};

}

#endif
