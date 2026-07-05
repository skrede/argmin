#ifndef HPP_GUARD_ARGMIN_DETAIL_SQP_COMMON_H
#define HPP_GUARD_ARGMIN_DETAIL_SQP_COMMON_H

// Cross-policy SQP scaffolding.
//
// Provides the shared pre-allocated workspace struct sqp_state_buffers
// adopted by the line-search SQP family (kraft_slsqp, nw_sqp,
// filter_slsqp, filter_nw_sqp), plus the shared helper functions
// (null_step_result, extract_qp_multipliers, compute_bfgs_pair_fused,
// equality_feasibility_warmstart, step_with_projection) that
// consolidate the duplicated patterns at the four SQP policy hot
// paths.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e, Section 18
//            (sequential quadratic programming);
//            Kraft 1988 DFVLR-FB 88-28 (line-search SLSQP).

#include "argmin/types.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/result/step_result.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/QR>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cstddef>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

// Adopted from: argmin/solver/kraft_slsqp_policy.h:135-192 (in-tree
//               precedent — the canonical *_buf state-resident layout).
// Reference: N&W 2e Section 18; Kraft 1988 DFVLR-FB 88-28.
//
// argmin variant: cross-policy struct consolidating the per-policy
//                 pre-allocated workspace; rationale: enables one
//                 shared allocation path across the SQP family.
template <typename Scalar = double, int N = argmin::dynamic_dimension>
struct sqp_state_buffers
{
    // Iterate / direction / box-aware step-direction buffers.
    Eigen::Vector<Scalar, N> x_trial_buf;
    Eigen::Vector<Scalar, N> p_buf;
    Eigen::Vector<Scalar, N> p_lo_buf;
    Eigen::Vector<Scalar, N> p_hi_buf;
    Eigen::Vector<Scalar, N> x_old_buf;
    Eigen::Vector<Scalar, N> g_old_buf;

    // BFGS curvature-pair buffers.
    Eigen::Vector<Scalar, N> grad_L_old_buf;
    Eigen::Vector<Scalar, N> grad_L_new_buf;
    Eigen::Vector<Scalar, N> sk_buf;
    Eigen::Vector<Scalar, N> yk_buf;

    // Constraint-axis buffers (m = n_eq + n_ineq dynamic).
    Eigen::VectorXd c_all;
    Eigen::VectorXd c_trial_buf;
    Eigen::VectorXd b_eq_workspace;
    Eigen::VectorXd b_ineq_workspace;
    Eigen::VectorXd b_eq_soc_buf;
    Eigen::VectorXd b_ineq_soc_buf;
    Eigen::VectorXd lam_buf;
    Eigen::VectorXd lam_eq_buf;
    Eigen::VectorXd lam_ineq_buf;
    Eigen::VectorXd kkt_lambda_eq_buf;
    Eigen::VectorXd kkt_mu_ineq_buf;

    // Constraint Jacobian (m x n). Typed Eigen::MatrixXd (dynamic rows AND
    // columns) to match the user-facing constraint_jacobian callback API
    // (which accepts Eigen::MatrixXd& by reference). Adopting Matrix<Scalar,
    // Dynamic, N> here would break the binding at policies' init() and
    // post-step Jacobian re-evaluation sites for fixed-N problems.
    Eigen::MatrixXd J_all;
    Eigen::MatrixXd J_all_old;

    // Pre-factored Hessian for kraft_lsq_qp's solve_with_factored_hessian path
    // (Kraft lineage only; N&W lineage leaves these zero-sized).
    Eigen::Matrix<Scalar, N, N> E_buf;
    Eigen::Vector<Scalar, N> f_buf;

    void resize(int n, int n_eq, int n_ineq);
};

template <typename Scalar, int N>
inline void sqp_state_buffers<Scalar, N>::resize(int n, int n_eq, int n_ineq)
{
    const int m = n_eq + n_ineq;

    x_trial_buf.resize(n);
    p_buf.resize(n);
    p_lo_buf.resize(n);
    p_hi_buf.resize(n);
    x_old_buf.resize(n);
    g_old_buf.resize(n);

    grad_L_old_buf.resize(n);
    grad_L_new_buf.resize(n);
    sk_buf.resize(n);
    yk_buf.resize(n);

    c_all.resize(m);
    c_trial_buf.resize(m);
    b_eq_workspace.resize(n_eq);
    b_ineq_workspace.resize(n_ineq);
    b_eq_soc_buf.resize(n_eq);
    b_ineq_soc_buf.resize(n_ineq);
    lam_buf.resize(m);
    lam_eq_buf.resize(n_eq);
    lam_ineq_buf.resize(n_ineq);
    kkt_lambda_eq_buf.resize(n_eq);
    kkt_mu_ineq_buf.resize(n_ineq);

    J_all.resize(m, n);
    J_all_old.resize(m, n);

    E_buf.resize(n, n);
    f_buf.resize(n);
}

// Adopted from: argmin/result/step_result.h schema (in-tree precedent —
//               the canonical step_result population pattern).
// Reference: N&W 2e Section 12.3 / eq. 12.34 (Lagrangian stationarity);
//            argmin/detail/kkt_residual.h (composite KKT residual).
//
// argmin variant: assemble a step_result for null/no-move steps with
//                 KKT residual computed at the current iterate;
//                 rationale: removes the duplicated null-step assembly
//                 across the four line-search SQP policies.
template <typename Scalar, int N, int Meq = argmin::dynamic_dimension,
          int Mineq = argmin::dynamic_dimension>
ARGMIN_FORCE_INLINE step_result<Scalar> null_step_result(
    Scalar f,
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& lam_eq,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& mu_ineq,
    const Eigen::Vector<Scalar, Meq>& c_eq,
    const Eigen::Vector<Scalar, Mineq>& c_ineq,
    Scalar x_norm,
    std::size_t bfgs_reset_count = 0,
    std::optional<solver_status> policy_status = std::nullopt)
{
    step_result<Scalar> r{};
    r.objective_value = f;
    r.step_size = Scalar(0);
    r.objective_change = Scalar(0);
    r.improved = false;
    r.is_null_step = true;
    r.x_norm = x_norm;
    r.constraint_violation = primal_feasibility_inf(c_eq, c_ineq);

    // Lagrangian gradient norm via two GEMVs (no allocation of stacked A).
    // grad_L = g - J_eq^T * lam_eq - J_ineq^T * mu_ineq.
    Eigen::Vector<Scalar, N> grad_L = g;
    if(J_eq.rows() > 0 && lam_eq.size() > 0)
        grad_L.noalias() -= J_eq.transpose() * lam_eq;
    if(J_ineq.rows() > 0 && mu_ineq.size() > 0)
        grad_L.noalias() -= J_ineq.transpose() * mu_ineq;
    r.gradient_norm = grad_L.size() > 0
        ? grad_L.template lpNorm<Eigen::Infinity>()
        : Scalar(0);

    // Composite KKT residual at the current iterate. Caller may overwrite
    // with a freshly recomputed value when a reset path needs the post-reset
    // residual instead.
    if(lam_eq.size() == J_eq.rows() && mu_ineq.size() == J_ineq.rows())
    {
        const Eigen::Vector<Scalar, Meq> lam_eq_typed = lam_eq;
        const Eigen::Vector<Scalar, Mineq> mu_ineq_typed = mu_ineq;
        r.kkt_residual = kkt_residual<Scalar, N, Meq, Mineq>(
            g, J_eq, J_ineq, lam_eq_typed, mu_ineq_typed, c_eq, c_ineq);
    }
    else
    {
        r.kkt_residual = std::nullopt;
    }

    r.diagnostics.bfgs_reset_count = bfgs_reset_count;
    r.policy_status = policy_status;

    return r;
}

// Adopted from: argmin/result/step_result.h schema (in-tree precedent
//               for partial-multiplier QP returns); argmin/detail/active_set_qp.h
//               solve() lambda layout.
// Reference: N&W 2e Section 16.5 (active-set QP duality);
//            argmin/detail/lagrangian.h estimate_multipliers_active_set.
//
// argmin variant: scatter qp_res.lambda into per-leg outputs handling
//                 partial-multiplier returns (qp.lambda.size() < m_total);
//                 rationale: removes the duplicated scatter pattern at
//                 four call sites across the SQP family.
template <typename Scalar>
ARGMIN_FORCE_INLINE void extract_qp_multipliers(
    const Eigen::Vector<Scalar, Eigen::Dynamic>& qp_lambda,
    int n_eq,
    int n_ineq,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> lam_eq_out,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> mu_ineq_out)
{
    const int m_total = n_eq + n_ineq;
    const int lam_size = static_cast<int>(qp_lambda.size());

    // Fast path: full-multiplier QP return. Skip the setZero pre-clear on
    // both legs (the assignments below cover every output element). This
    // is the dominant path on healthy SQP iterations; the partial-multiplier
    // fallback below explicitly setZero-s the legs it does not write.
    if(lam_size == m_total)
    {
        if(n_eq > 0) lam_eq_out = qp_lambda.head(n_eq);
        if(n_ineq > 0) mu_ineq_out = qp_lambda.tail(n_ineq);
        return;
    }

    // Partial-multiplier path: copy what is available head-first into the
    // equality leg, then the remainder into the inequality leg. Pre-clear
    // both legs so the un-written tail elements are zero.
    lam_eq_out.setZero();
    mu_ineq_out.setZero();

    const int eq_take = std::min(lam_size, n_eq);
    if(eq_take > 0) lam_eq_out.head(eq_take) = qp_lambda.head(eq_take);
    const int rem = lam_size - eq_take;
    if(rem > 0 && n_ineq > 0)
    {
        const int ineq_take = std::min(rem, n_ineq);
        mu_ineq_out.head(ineq_take) = qp_lambda.segment(eq_take, ineq_take);
    }
}

// Adopted from: argmin/solver/kraft_slsqp_policy.h:776-820 (in-tree
//               precedent — fused J^T*lambda subtraction over both legs);
//               NLopt slsqp.c slsqpb_'s u-update loop.
// Reference: N&W 2e Section 18.3 (Lagrangian gradient pair);
//            Kraft 1988 DFVLR-FB 88-28 §2.2.5 (BFGS update pair).
//
// argmin variant: build (sk, yk) via a single J^T*lambda GEMV per
//                 gradient (vs the prior eq/ineq split that ran two
//                 GEMVs); rationale: halves the gradient-walk cost
//                 on the curvature-pair construction hot path.
template <typename Scalar, int N>
ARGMIN_FORCE_INLINE void compute_bfgs_pair_fused(
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g_old,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g_new,
    const Eigen::Ref<const Eigen::MatrixXd>& J_all_old,
    const Eigen::Ref<const Eigen::MatrixXd>& J_all,
    const Eigen::Ref<const Eigen::Vector<Scalar, Eigen::Dynamic>>& lam_full,
    int m_total,
    Eigen::Ref<Eigen::Vector<Scalar, N>> grad_L_old,
    Eigen::Ref<Eigen::Vector<Scalar, N>> grad_L_new,
    Eigen::Ref<Eigen::Vector<Scalar, N>> sk_out,
    Eigen::Ref<Eigen::Vector<Scalar, N>> yk_out,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& x_new,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& x_old)
{
    // Operation order is bit-for-bit identical to kraft_slsqp_policy.h:776-820:
    //   1. seed grad_L_old / grad_L_new with the bare objective gradients;
    //   2. if the QP returned a full multiplier vector (lam_take == m_total),
    //      run a single fused J_all^T * lam_full GEMV per gradient against
    //      topRows(m_total) of the eq-then-ineq concatenated Jacobian;
    //   3. otherwise keep the bare gradients (kraft's eq-only fallback at
    //      :811-818 is structurally a no-op for the helper signature because
    //      the fallback used s.J_eq directly rather than J_all_old.topRows
    //      and is policy-internal; the helper exposes the m_total-only
    //      branch here and lets adopters keep the eq-only fallback inline
    //      where it touches policy-private state).
    grad_L_old = g_old;
    grad_L_new = g_new;

    const int lam_take = static_cast<int>(lam_full.size());
    if(lam_take == m_total && m_total > 0)
    {
        grad_L_old.noalias() -= J_all_old.topRows(m_total).transpose()
                               * lam_full.head(m_total);
        grad_L_new.noalias() -= J_all.topRows(m_total).transpose()
                               * lam_full.head(m_total);
    }

    sk_out.noalias() = x_new - x_old;
    yk_out.noalias() = grad_L_new - grad_L_old;
}

// Adopted from: argmin/solver/nw_sqp_policy.h:222-232, :297-300 (in-tree
//               precedent — LDLT-based equality-feasibility warm-start).
// Reference: N&W 2e Section 18.3 (equality-feasibility step);
//            argmin/detail/lagrangian.h (least-squares projection idiom).
//
// argmin variant: hoists nw_sqp's LDLT warm-start path into a shared
//                 helper with caller-owned workspace and output; rationale:
//                 enables filter_nw_sqp adoption without a per-step
//                 ColPivHouseholderQR (REF-05 close).
template <typename Scalar, int N, int Meq = argmin::dynamic_dimension>
ARGMIN_FORCE_INLINE void equality_feasibility_warmstart(
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
    Eigen::MatrixXd& AAt_workspace,
    Eigen::LDLT<Eigen::MatrixXd>& ldlt_workspace,
    Eigen::Vector<Scalar, Eigen::Dynamic>& w_workspace,
    Eigen::Ref<Eigen::Vector<Scalar, N>> p0_out)
{
    // Verbatim port of nw_sqp_policy.h:297-300:
    //   AAt = J_eq * J_eq^T, ldlt.compute(AAt), w = ldlt.solve(b_eq),
    //   p0 = J_eq^T * w. LDLT (not LLT) handles indefinite/singular
    //   J*J^T cases that LLT cannot.
    //
    // w_workspace is a caller-owned VectorXd pre-sized to b_eq.size()
    // (see init() in nw_sqp_policy / filter_nw_sqp_policy); writing into
    // it via the no-allocation Eigen::LDLT::solveInPlace path keeps the
    // hot-loop allocation count at zero.
    AAt_workspace.noalias() = J_eq * J_eq.transpose();
    ldlt_workspace.compute(AAt_workspace);
    w_workspace = b_eq;
    ldlt_workspace.solveInPlace(w_workspace);
    p0_out.noalias() = J_eq.transpose() * w_workspace;
}

// Orthogonal projection onto null(A) via the AA^T normal equations.
//
// Computes, in place, z = x - A^T (A A^T)^{-1} A x -- the orthogonal
// projector onto the null space of A (equivalently, x with its
// range(A^T) component removed). The caller supplies a prepared LDLT of
// A A^T (the same factorization the Byrd-Omojokun normal step already
// builds) and an m-sized scratch vector, so the full-rank path performs
// no heap allocation: one A x (m*n), one LDLT back-solve (m^2), one
// A^T y (m*n) per projection.
//
// Iterative refinement (Gould, Hribar, Nocedal 2001, "Solution of the
// Trust-Region Subproblem Using the Lanczos Method" / their projected-CG
// preconditioner, Algorithm 5.1-5.2): after the projection the residual
// orthogonality ||A z||_inf / max(1, ||A||_inf ||z||_inf) is measured
// and the projection is re-applied while it exceeds the tolerance, up to
// refine_passes additional passes. This absorbs the drift off null(A)
// that accumulated rounding introduces in the normal-equations
// projector at tight tolerances.
//
// Rank-deficient A (ldlt.info() != Success): fall back to the projector
// built from a rank-revealing ColPivHouseholderQR of A^T, z = x - Q1
// Q1^T x with Q1 the first rank(A) columns of Q. Correctness fallback
// only -- it allocates and is off the hot path (the composite step also
// flags normal_step_lsq_fallback on this branch).
//
// Reference: Gould, Hribar, Nocedal 2001 SIAM J. Sci. Comput. 23(6);
//            Nocedal and Wright 2e Section 16.3 Algorithm 16.2
//            (projected CG); scipy optimize/_trustregion_constr/
//            projections.py (normal-equations projection).
template <typename Scalar, int N>
ARGMIN_FORCE_INLINE void null_space_project(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, N>>& A,
    Eigen::LDLT<Eigen::MatrixXd>& ldlt,
    Eigen::Ref<Eigen::Vector<Scalar, N>> x,
    Eigen::VectorXd& m_scratch,
    int refine_passes)
{
    const int m = static_cast<int>(A.rows());
    if(m == 0)
        return;  // null(A) is all of R^n; nothing to remove.

    if(ldlt.info() != Eigen::Success)
    {
        // Rank-deficient AA^T: project off range(A^T) via pivoted QR.
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
            A.transpose());
        const int r = static_cast<int>(qr.rank());
        if(r == 0)
            return;
        const Eigen::MatrixXd Q =
            qr.householderQ()
            * Eigen::MatrixXd::Identity(A.cols(), r);
        x.noalias() -= Q * (Q.transpose() * x);
        return;
    }

    // Full-rank normal-equations projection.
    m_scratch.noalias() = A * x;
    ldlt.solveInPlace(m_scratch);
    x.noalias() -= A.transpose() * m_scratch;

    // Iterative refinement to hold ||A z|| at the projection tolerance.
    const Scalar A_inf = A.template lpNorm<Eigen::Infinity>();
    for(int pass = 0; pass < refine_passes; ++pass)
    {
        m_scratch.noalias() = A * x;
        const Scalar ortho = m_scratch.template lpNorm<Eigen::Infinity>();
        const Scalar scale =
            std::max(Scalar(1),
                     A_inf * x.template lpNorm<Eigen::Infinity>());
        if(ortho <= Scalar(1e-14) * scale)
            break;
        ldlt.solveInPlace(m_scratch);
        x.noalias() -= A.transpose() * m_scratch;
    }
}

// Adopted from: argmin/detail/bound_projection.h:19 (in-tree precedent
//               — detail::project clip-to-box utility).
// Reference: N&W 2e Section 17.3 (projected-step trial-iterate idiom).
//
// argmin variant: micro-helper composing x_trial = project(x + alpha*p)
//                 against a box; rationale: replaces four hand-rolled
//                 inline clip sites in kraft / filter_slsqp.
template <typename Scalar, int N>
ARGMIN_FORCE_INLINE void step_with_projection(
    const Eigen::Vector<Scalar, N>& x,
    Scalar alpha,
    const Eigen::Vector<Scalar, N>& p,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Eigen::Ref<Eigen::Vector<Scalar, N>> x_trial_out)
{
    // Compose x_trial = clamp(x + alpha*p, lower, upper) without an
    // intermediate copy. The previous pattern materialized tmp = x_trial_out
    // before forwarding to detail::project; using cwiseMax / cwiseMin keeps
    // the entire expression in expression-template space and lets Eigen
    // fuse the three vector ops into a single packet loop.
    x_trial_out.noalias() = x + alpha * p;
    x_trial_out = x_trial_out.cwiseMax(lower).cwiseMin(upper);
}

}

#endif
