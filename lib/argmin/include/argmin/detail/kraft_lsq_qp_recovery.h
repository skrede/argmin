#ifndef HPP_GUARD_ARGMIN_DETAIL_KRAFT_LSQ_QP_RECOVERY_H
#define HPP_GUARD_ARGMIN_DETAIL_KRAFT_LSQ_QP_RECOVERY_H

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
// passed to the QP. argmin's outer SQP convention sets b = -c (constraint
// values), so the §3.4 augmentation column "-c[j]" for equalities equals
// b_eq[j]; for inequalities it is "max(-c[j], 0)" = "max(b_ineq[j], 0)".
//
// At s=0 the augmented problem reduces to the original (feasible iff the
// linearization is feasible). At s=1 the relaxed constraints are
// trivially satisfied (the original RHS multiplied by (1-s) is zero).
// The penalty rho on the diagonal of the augmented Hessian discourages
// large s; per §3.4 rho is grown geometrically across retries until the
// augmented QP is feasible or the retry cap is exhausted.
//
// On the optimal direct-path return the wrapper is bit-identical to
// kraft_lsq_qp_solver -- only the augmented retry adds work, and only on
// non-optimal returns from the direct path. Workspaces for both n and
// n+1 dimensions are pre-allocated in resize() so solve() does not
// allocate.
//
// Reference: Kraft, D. (1988). A Software Package for Sequential
//            Quadratic Programming. DFVLR-FB 88-28, §3.4 (Inconsistent
//            Linearization).

#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/active_set_qp.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
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

template <typename Scalar = double, int N = argmin::dynamic_dimension>
class kraft_lsq_qp_recovery_solver
{
public:
    static constexpr int N_aug = recovery_detail::augmented_dim_v<N>;

    kraft_lsq_qp_recovery_solver() = default;

    // Sized for step boxes with the finite entries of [lower, upper], as kraft_lsq_qp_solver::resize.
    void resize(int n, int m_eq, int m_ineq, const Eigen::Vector<Scalar, N>& lower, const Eigen::Vector<Scalar, N>& upper)
    {
        augment_box_(n, lower, upper);
        solver_.resize(n, m_eq, m_ineq, lower, upper);
        solver_aug_.resize(n + 1, m_eq, m_ineq, p_lo_aug_, p_hi_aug_);

        const int n_aug = n + 1;
        B_aug_.resize(n_aug, n_aug);
        g_aug_.resize(n_aug);
        A_eq_aug_.resize(m_eq, n_aug);
        A_ineq_aug_.resize(m_ineq, n_aug);
        B_recover_.resize(n, n);
        aug_.x.setZero(n_aug);
        aug_.lambda.setZero(m_eq + m_ineq);
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
    template <int M = argmin::dynamic_dimension>
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
        qp_result<Scalar, N, M> out;
        solve_into<M>(out, B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
        return out;
    }

    // Fill-into sibling of solve(): a caller reusing the same qp_result at a stable problem shape incurs no per-solve
    // allocation on either path.
    template <int M = argmin::dynamic_dimension>
    void solve_into(
        qp_result<Scalar, N, M>& out,
        const Eigen::Matrix<Scalar, N, N>& B,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        out.relaxation_factor = Scalar(0);
        solver_.template solve_into<M>(out, B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
        if(out.status != qp_status::optimal)
            recover_<M>(out, B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
    }

    // Variant of solve() that consumes a pre-factored Hessian (E, f) for
    // the direct path, where E^T E = B and E^T f = -g. The caller
    // (typically a BFGS approximation maintained as packed L D L^T)
    // produces (E, f) at O(n^2) cost via dense_ldl_bfgs::factor_to_E_and_f,
    // replacing the O(n^3 / 3) LLT that solve(B, g, ...) runs internally.
    //
    // On the rare recovery path (Kraft §3.4 augmented retry), B is
    // reconstructed from E (O(n^3 / 2)) so the augmented solver can build
    // the block-diagonal augmented Hessian. Recovery fires only when the
    // direct LSEI cascade reports infeasibility / rank deficiency, so the
    // reconstruction cost is amortized away on optimal SQP runs.
    template <int M = argmin::dynamic_dimension>
    qp_result<Scalar, N, M> solve_with_factored_hessian(
        const Eigen::Matrix<Scalar, N, N>& E,
        const Eigen::Vector<Scalar, N>& f,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        qp_result<Scalar, N, M> out;
        solve_with_factored_hessian_into<M>(
            out, E, f, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
        return out;
    }

    // Fill-into sibling of solve_with_factored_hessian(): writes into the
    // caller-owned qp_result so a caller reusing the same result across
    // solves incurs no per-solve allocation.
    template <int M = argmin::dynamic_dimension>
    void solve_with_factored_hessian_into(
        qp_result<Scalar, N, M>& out,
        const Eigen::Matrix<Scalar, N, N>& E,
        const Eigen::Vector<Scalar, N>& f,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        // Reset the recovery telemetry so a reused qp_result does not carry
        // a stale relaxation_factor from a prior step's recovery into a
        // direct-optimal solve (the fresh-result overload defaults it to 0).
        out.relaxation_factor = Scalar(0);

        solver_.template solve_with_factored_hessian_into<M>(
            out, E, f, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

        if(out.status == qp_status::optimal)
            return;

        const int n = static_cast<int>(E.rows());
        if(B_recover_.rows() != n || B_recover_.cols() != n)
            B_recover_.resize(n, n);
        B_recover_.setZero();
        B_recover_.template selfadjointView<Eigen::Lower>()
            .rankUpdate(E.transpose());
        B_recover_.template triangularView<Eigen::StrictlyUpper>() =
            B_recover_.transpose();
        recover_<M>(out, B_recover_, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
    }

private:
    // Augmented retry over a direct result that was not optimal. On success the step is the first n components of the
    // augmented x and relaxation_factor the slack s; on exhaustion `out` keeps the direct result, flagged with
    // relaxation_factor = 1.
    template <int M>
    void recover_(
        qp_result<Scalar, N, M>& out,
        const Eigen::Matrix<Scalar, N, N>& B,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        const int n = static_cast<int>(B.rows());
        augment_rows_(n, A_eq, b_eq, A_ineq, b_ineq);
        augment_objective_(n, B, g, p_lo, p_hi);

        Scalar rho = penalty_initial_;
        for(int attempt = 0; attempt < max_attempts_; ++attempt)
        {
            B_aug_(n, n) = rho;
            solver_aug_.solve_into(aug_, B_aug_, g_aug_, A_eq_aug_, b_eq, A_ineq_aug_, b_ineq, p_lo_aug_, p_hi_aug_);
            if(aug_.status == qp_status::optimal)
            {
                out.x.resize(n);
                out.x = aug_.x.head(n);
                out.lambda = aug_.lambda;
                out.status = qp_status::optimal;
                out.iterations = attempt + 2;  // 1 direct + (attempt+1) augmented
                out.relaxation_factor = aug_.x[n];
                return;
            }
            rho *= penalty_growth_;
        }
        out.relaxation_factor = Scalar(1);
    }

    // Augmented A_eq: leftmost n cols = A_eq, last col = b_eq. Augmented
    // A_ineq: leftmost n cols = A_ineq, last col = max(b_ineq, 0).
    void augment_rows_(
        int n,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_eq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A_ineq,
        const Eigen::Vector<Scalar, Eigen::Dynamic>& b_ineq)
    {
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());
        if(A_eq_aug_.rows() != m_eq || A_eq_aug_.cols() != n + 1)
            A_eq_aug_.resize(m_eq, n + 1);
        if(A_ineq_aug_.rows() != m_ineq || A_ineq_aug_.cols() != n + 1)
            A_ineq_aug_.resize(m_ineq, n + 1);
        if(m_eq > 0)
        {
            A_eq_aug_.leftCols(n) = A_eq;
            A_eq_aug_.col(n) = b_eq;
        }
        if(m_ineq > 0)
        {
            A_ineq_aug_.leftCols(n) = A_ineq;
            for(int j = 0; j < m_ineq; ++j)
                A_ineq_aug_(j, n) = std::max(b_ineq[j], Scalar(0));
        }
    }

    // Box [p_lo, p_hi] on p and [0, 1] on s, no linear cost on s, and the block-diagonal Hessian [B 0; 0 rho] whose
    // rho entry each attempt sets.
    void augment_objective_(
        int n,
        const Eigen::Matrix<Scalar, N, N>& B,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Vector<Scalar, N>& p_lo,
        const Eigen::Vector<Scalar, N>& p_hi)
    {
        const int n_aug = n + 1;
        if(B_aug_.rows() != n_aug || B_aug_.cols() != n_aug)
            B_aug_.resize(n_aug, n_aug);
        if(g_aug_.size() != n_aug) g_aug_.resize(n_aug);
        augment_box_(n, p_lo, p_hi);
        g_aug_.head(n) = g;
        g_aug_[n] = Scalar(0);
        B_aug_.setZero();
        B_aug_.topLeftCorner(n, n) = B;
    }

    void augment_box_(int n, const Eigen::Vector<Scalar, N>& p_lo, const Eigen::Vector<Scalar, N>& p_hi)
    {
        if(p_lo_aug_.size() != n + 1) p_lo_aug_.resize(n + 1);
        if(p_hi_aug_.size() != n + 1) p_hi_aug_.resize(n + 1);
        p_lo_aug_.head(n) = p_lo;
        p_lo_aug_[n] = Scalar(0);
        p_hi_aug_.head(n) = p_hi;
        p_hi_aug_[n] = Scalar(1);
    }

    // Penalty schedule and retry cap per Kraft 1988 §3.4: rho_0 = 100,
    // rho_{k+1} = 10 * rho_k, k_max = 5. Adjusting these changes the
    // tradeoff between aggressive relaxation (smaller initial penalty,
    // more attempts) and tight constraint satisfaction (larger initial
    // penalty, fewer attempts before declaring failure).
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

    // Scratch for the recovery path of solve_with_factored_hessian: B
    // is reconstructed from E (B = E^T E) only when the direct LSEI
    // cascade reports infeasibility. Held as a member so the rare
    // recovery does not allocate.
    Eigen::Matrix<Scalar, N, N>                     B_recover_;
    qp_result<Scalar, N_aug>                        aug_;
};

}

#endif
