#ifndef HPP_GUARD_ARGMIN_DETAIL_SQP_COMMON_H
#define HPP_GUARD_ARGMIN_DETAIL_SQP_COMMON_H

// Cross-policy SQP scaffolding.
//
// Provides the shared pre-allocated workspace struct sqp_state_buffers
// adopted by the line-search SQP family (kraft_slsqp, nw_sqp,
// filter_slsqp, filter_nw_sqp), plus forward declarations for the
// shared helper functions whose bodies live in this header's
// implementation companion.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e, Section 18
//            (sequential quadratic programming);
//            Kraft 1988 DFVLR-FB 88-28 (line-search SLSQP).

#include "argmin/types.h"
#include "argmin/result/step_result.h"

#include <Eigen/QR>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cstddef>
#include <optional>

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

    // Constraint Jacobian (m x n; row-axis dynamic, col-axis matches N).
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> J_all;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> J_all_old;

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
step_result<Scalar> null_step_result(
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
    std::optional<solver_status> policy_status = std::nullopt);

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
void extract_qp_multipliers(
    const Eigen::Vector<Scalar, Eigen::Dynamic>& qp_lambda,
    int n_eq,
    int n_ineq,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> lam_eq_out,
    Eigen::Ref<Eigen::Vector<Scalar, Eigen::Dynamic>> mu_ineq_out);

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
void compute_bfgs_pair_fused(
    const Eigen::Vector<Scalar, N>& g_old,
    const Eigen::Vector<Scalar, N>& g_new,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& J_all_old,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& J_all,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& lam_full,
    int m_total,
    Eigen::Ref<Eigen::Vector<Scalar, N>> grad_L_old,
    Eigen::Ref<Eigen::Vector<Scalar, N>> grad_L_new,
    Eigen::Ref<Eigen::Vector<Scalar, N>> sk_out,
    Eigen::Ref<Eigen::Vector<Scalar, N>> yk_out,
    const Eigen::Vector<Scalar, N>& x_new,
    const Eigen::Vector<Scalar, N>& x_old);

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
void equality_feasibility_warmstart(
    const Eigen::Matrix<Scalar, Meq, N>& J_eq,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& b_eq,
    Eigen::MatrixXd& AAt_workspace,
    Eigen::LDLT<Eigen::MatrixXd>& ldlt_workspace,
    Eigen::Ref<Eigen::Vector<Scalar, N>> p0_out);

// Adopted from: argmin/detail/bound_projection.h:19 (in-tree precedent
//               — detail::project clip-to-box utility).
// Reference: N&W 2e Section 17.3 (projected-step trial-iterate idiom).
//
// argmin variant: micro-helper composing x_trial = project(x + alpha*p)
//                 against a box; rationale: replaces four hand-rolled
//                 inline clip sites in kraft / filter_slsqp.
template <typename Scalar, int N>
void step_with_projection(
    const Eigen::Vector<Scalar, N>& x,
    Scalar alpha,
    const Eigen::Vector<Scalar, N>& p,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Eigen::Ref<Eigen::Vector<Scalar, N>> x_trial_out);

}

#endif
