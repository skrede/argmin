#ifndef HPP_GUARD_NABLAPP_DETAIL_KRAFT_LSQ_QP_RECOVERY_H
#define HPP_GUARD_NABLAPP_DETAIL_KRAFT_LSQ_QP_RECOVERY_H

// Kraft 1988 §3.4 augmented-QP recovery wrapper around kraft_lsq_qp_solver.
//
// When the inner LSQ/LSEI cascade returns mode=4 (inequality infeasible)
// or mode=6 (rank-deficient equality), Kraft 1988 §3.4 prescribes adding
// a slack variable s in [0, 1] that relaxes the constraints, with a
// penalty term in the objective that grows across up to 5 retry
// iterations until the augmented QP becomes feasible.
//
// Augmented problem (per Kraft 1988 §3.4):
//   min 0.5 [p; s]^T diag(B, rho) [p; s] + [g; 0]^T [p; s]
//   s.t. [A_eq, b_eq] [p; s]            = b_eq
//        [A_ineq, max(b_ineq, 0)] [p; s] >= b_ineq
//        p_lo <= p <= p_hi
//        0 <= s <= 1
//
// The augmentation column is derived from the linearized constraint RHS
// passed to the QP. nablapp's outer SQP convention sets b = -c (constraint
// values), so the NLopt-style augmentation column "-c[j]" for equalities
// equals b_eq[j]; for inequalities it is "max(-c[j], 0)" = "max(b_ineq[j], 0)".
//
// At s=0 the augmented problem reduces to the original (feasible iff the
// linearization is feasible). At s=1 the relaxed constraints are
// trivially satisfied (set the original RHS to zero). The penalty rho on
// the diagonal of the augmented Hessian discourages large s; growth
// schedule rho_k = 100 * 10^k for k=0..4 matches NLopt's slsqpb_.
//
// On the optimal direct-path return the wrapper is bit-identical to
// kraft_lsq_qp_solver -- only the augmented retry adds work, and only on
// non-optimal returns from the direct path. Workspaces for both n and
// n+1 dimensions are pre-allocated in resize() so solve() does not
// allocate.
//
// Reference: Kraft, D. (1988). A Software Package for Sequential
//            Quadratic Programming. DFVLR-FB 88-28, §3.4.
//            NLopt slsqp.c slsqpb_ lines 1923-1963 (S.G. Johnson 2009
//            translation).

#include "nablapp/detail/kraft_lsq_qp.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

namespace recovery_detail
{

// Augmented dimension: N+1 for fixed N, Dynamic for Dynamic.
template <int N>
struct augmented_dim { static constexpr int value = N + 1; };

template <>
struct augmented_dim<Eigen::Dynamic> { static constexpr int value = Eigen::Dynamic; };

template <int N>
inline constexpr int augmented_dim_v = augmented_dim<N>::value;

}

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class kraft_lsq_qp_recovery_solver
{
public:
    static constexpr int N_aug = recovery_detail::augmented_dim_v<N>;

    kraft_lsq_qp_recovery_solver() = default;

    // Pre-allocate workspace for a problem of the given shape.
    // n_finite_lower / n_finite_upper are upper bounds on how many of the
    // n primal box bounds will be finite at solve-time; the augmented
    // solver always has one additional finite lower (s >= 0) and one
    // additional finite upper (s <= 1) bound, accounted for here.
    void resize(int n, int m_eq, int m_ineq, int n_finite_lower, int n_finite_upper)
    {
        solver_.resize(n, m_eq, m_ineq, n_finite_lower, n_finite_upper);
        solver_aug_.resize(n + 1, m_eq, m_ineq, n_finite_lower + 1, n_finite_upper + 1);

        const int n_aug = n + 1;
        B_aug_.resize(n_aug, n_aug);
        g_aug_.resize(n_aug);
        A_eq_aug_.resize(m_eq, n_aug);
        A_ineq_aug_.resize(m_ineq, n_aug);
        p_lo_aug_.resize(n_aug);
        p_hi_aug_.resize(n_aug);
    }

    // Solve the QP subproblem with augmented-QP recovery on infeasibility.
    //
    // Direct path: calls kraft_lsq_qp_solver::solve once. On optimal
    // status, returns immediately (bit-identical to direct kraft solve).
    //
    // Recovery path: on mode 4 (infeasible) or mode 6 (rank-deficient
    // equality), constructs the augmented (n+1)-dim QP per Kraft §3.4 and
    // retries with growing penalty up to max_attempts_ times. On augmented
    // success, returns the first n components of the augmented x as the
    // step; forwards the augmented multipliers (which approximate the
    // original problem multipliers when s is small).
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
        auto direct = solver_.template solve<M>(
            B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

        if(direct.status == qp_status::optimal)
            return direct;

        // -------------------------------------------------------------
        // Augmented retry path.
        // -------------------------------------------------------------
        const int n      = static_cast<int>(B.rows());
        const int m_eq   = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());
        const int n_aug  = n + 1;

        if(B_aug_.rows() != n_aug || B_aug_.cols() != n_aug)
            B_aug_.resize(n_aug, n_aug);
        if(g_aug_.size() != n_aug) g_aug_.resize(n_aug);
        if(A_eq_aug_.rows() != m_eq || A_eq_aug_.cols() != n_aug)
            A_eq_aug_.resize(m_eq, n_aug);
        if(A_ineq_aug_.rows() != m_ineq || A_ineq_aug_.cols() != n_aug)
            A_ineq_aug_.resize(m_ineq, n_aug);
        if(p_lo_aug_.size() != n_aug) p_lo_aug_.resize(n_aug);
        if(p_hi_aug_.size() != n_aug) p_hi_aug_.resize(n_aug);

        // Augmented A_eq: leftmost n cols = A_eq, last col = b_eq.
        if(m_eq > 0)
        {
            A_eq_aug_.leftCols(n) = A_eq;
            A_eq_aug_.col(n) = b_eq;
        }

        // Augmented A_ineq: leftmost n cols = A_ineq, last col = max(b_ineq, 0).
        if(m_ineq > 0)
        {
            A_ineq_aug_.leftCols(n) = A_ineq;
            for(int j = 0; j < m_ineq; ++j)
                A_ineq_aug_(j, n) = std::max(b_ineq[j], Scalar(0));
        }

        // Augmented box bounds: original bounds on p, [0, 1] on s.
        p_lo_aug_.head(n) = p_lo;
        p_lo_aug_[n] = Scalar(0);
        p_hi_aug_.head(n) = p_hi;
        p_hi_aug_[n] = Scalar(1);

        // Augmented gradient: extra component is zero (no linear cost on s).
        g_aug_.head(n) = g;
        g_aug_[n] = Scalar(0);

        // Augmented Hessian: block diagonal [B 0; 0 rho]. rho is updated
        // per attempt below. Off-diagonal blocks zeroed once.
        B_aug_.setZero();
        B_aug_.topLeftCorner(n, n) = B;

        Scalar rho = penalty_initial_;

        for(int attempt = 0; attempt < max_attempts_; ++attempt)
        {
            B_aug_(n, n) = rho;

            auto aug = solver_aug_.template solve<M>(
                B_aug_, g_aug_, A_eq_aug_, b_eq, A_ineq_aug_, b_ineq,
                p_lo_aug_, p_hi_aug_);

            if(aug.status == qp_status::optimal)
            {
                qp_result<Scalar, N, M> out;
                out.x.resize(n);
                out.x = aug.x.head(n);
                out.lambda = aug.lambda;
                out.status = qp_status::optimal;
                out.iterations = attempt + 2;  // 1 direct + (attempt+1) augmented
                out.relaxation_factor = aug.x[n];  // s in [0, 1]
                return out;
            }

            rho *= penalty_growth_;
        }

        // Augmentation failed across all attempts. Return the direct
        // path's failure status, with relaxation_factor = 1 to signal
        // that recovery was attempted but did not produce a feasible
        // augmented step.
        direct.relaxation_factor = Scalar(1);
        return direct;
    }

private:
    static constexpr Scalar penalty_initial_ = Scalar(100);
    static constexpr Scalar penalty_growth_  = Scalar(10);
    static constexpr int    max_attempts_    = 5;

    kraft_lsq_qp_solver<Scalar, N>     solver_;
    kraft_lsq_qp_solver<Scalar, N_aug> solver_aug_;

    Eigen::Matrix<Scalar, N_aug, N_aug>             B_aug_;
    Eigen::Vector<Scalar, N_aug>                    g_aug_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N_aug>    A_eq_aug_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N_aug>    A_ineq_aug_;
    Eigen::Vector<Scalar, N_aug>                    p_lo_aug_;
    Eigen::Vector<Scalar, N_aug>                    p_hi_aug_;
};

}

#endif
