#ifndef HPP_GUARD_ARGMIN_DETAIL_BYRD_OMOJOKUN_H
#define HPP_GUARD_ARGMIN_DETAIL_BYRD_OMOJOKUN_H

// Byrd-Omojokun composite step for trust-region SQP.
//
//   1. Normal step:   min ||A v + c||^2  s.t.  ||v|| <= zeta * delta
//                     via dogleg between the scaled Cauchy point and the
//                     minimum-norm LSQ Newton step.
//   2. Tangential:    min q_t(u) = (g + B v)^T u + 0.5 u^T B u
//                     s.t. ||v + u|| <= delta
//                     via the Steihaug-Toint truncated CG helper (see
//                     steihaug_cg.h).
//   3. Composite:     p = v + u.
//   4. Ratio test:    rho = actual_reduction / max(predicted_reduction, eps)
//                     with the augmented reduction (objective decrease
//                     plus linearized-feasibility decrease) per scipy
//                     convention.
//   5. Radius update: eta_1 = 0.1, eta_2 = 0.75, shrink x0.25,
//                     expand x2.0 (Nocedal-Wright 2e Section 4.1
//                     Algorithm 4.1 universal defaults).
//
// Adopted from: scipy/optimize/_trustregion_constr/qp_subproblem.py
//               (modified_dogleg + projected_cg);
//               scipy/optimize/_trustregion_constr/equality_constrained_sqp.py
//               (TR_FACTOR = 0.8 normal-step damping; ratio test and
//                radius update).
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 18.5 Algorithm 18.4 (Byrd-Omojokun composite step);
//            equations 18.45-18.50 (zeta damping; normal-step formulation);
//            Section 4.1 Algorithm 4.1 (TR ratio test and radius update);
//            Lalee, Nocedal, Plantenga 1998, SIAM J. Optim. 8(3):682-706
//              (concrete dogleg pseudocode for the normal step;
//               augmented-reduction ratio-test formula in Section 3.3
//               equations 3.4-3.5).
//
// argmin variant: callers assemble the joint residual with the
//                 inequality sign flipped to match the slack
//                 reformulation c_ineq + s = 0 with s >= 0; the helper
//                 consumes whatever c and A are passed and does not
//                 interpret sign convention. Bounds-projection inside
//                 the tangential Steihaug-CG handles box bounds and
//                 slack bounds uniformly via the displaced-box
//                 detail::project invocation. Caller is responsible
//                 for trial_eval finiteness; an unrecovered NaN from
//                 trial_eval propagates to a NaN ratio and the
//                 (NaN < eta_1) IEEE-754 branch accepts the step --
//                 caller-side clamping or validation is required.

#include "argmin/types.h"
#include "argmin/detail/steihaug_cg.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <utility>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace argmin::detail
{

// Normal-step damping factor.
//
// Reference: Nocedal and Wright 2e eq. 18.45 (zeta damping); Lalee,
//            Nocedal, Plantenga 1998 Section 3.2; scipy
//            equality_constrained_sqp.py TR_FACTOR (literature
//            universal value).
inline constexpr double zeta = 0.8;

// TR ratio thresholds and radius-update factors.
//
// Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
//            universal defaults (consistent with scipy trust-constr,
//            Ceres trust-region minimizer, KNITRO).
inline constexpr double tr_eta_1          = 0.1;   // shrink threshold
inline constexpr double tr_eta_2          = 0.75;  // expand threshold
inline constexpr double tr_shrink_factor  = 0.25;
inline constexpr double tr_expand_factor  = 2.0;
inline constexpr double tr_delta_max      = 1e10;
inline constexpr double tr_boundary_guard = 0.9;   // expand only when ||p|| >= 0.9 * delta

// Composite-step result POD returned by byrd_omojokun_composite_step.
//
// The caller decides whether to apply p_out to z_k based on `accepted`.
// `normal_step_lsq_fallback` is set when the LDLT factorization of
// AA^T failed (rank-deficient joint Jacobian) and the helper fell
// back to a pure-Cauchy step truncated to the zeta*delta boundary.
struct byrd_omojokun_result
{
    double rho;
    double new_delta;
    double actual_reduction;
    double predicted_reduction;
    bool accepted;
    bool normal_step_lsq_fallback;
    cg_exit_status tangential_cg_status;
};

// Byrd-Omojokun composite step.
//
// Inputs:
//   z_k              -- current iterate (joint primal variable).
//   g                -- joint Lagrangian gradient at z_k.
//   hessian_op       -- Hessian-vector-product callable; must accept a
//                       const Eigen::Ref to a step vector and return an
//                       Eigen::Vector of the same dimension.
//   A                -- joint constraint Jacobian (m_total x n_joint).
//   c                -- joint constraint residual at z_k (m_total).
//   delta            -- current trust-region radius.
//   eps              -- forcing tolerance (pre-resolved by the caller).
//   max_cg_iter      -- inner-iteration cap for the tangential
//                       Steihaug-CG leg.
//   lower_displaced  -- (lower - z_k) for the joint box.
//   upper_displaced  -- (upper - z_k) for the joint box.
//   f_old            -- objective value at z_k.
//   c_norm_old       -- ||c(z_k)||_2 (L2 merit feasibility term at z_k).
//   trial_eval       -- callable invoked as trial_eval(p_out) returning
//                       std::pair<Scalar, Scalar>{f_new, c_norm_new} at
//                       z_k + p_out. The caller owns evaluation; the
//                       helper just consumes the (f, c_norm) pair to
//                       form the augmented L2 merit difference.
//   AAt_workspace    -- caller-owned MatrixXd, m_total x m_total.
//   ldlt_workspace   -- caller-owned LDLT factorization object.
//   w_workspace      -- caller-owned VectorXd, m_total.
//   v_buf, u_buf     -- caller-owned step buffers in joint space; v_buf
//                       receives the normal step, u_buf receives the
//                       tangential step.
//   r_cg_buf,
//   d_cg_buf,
//   Bd_cg_buf        -- caller-owned Steihaug-CG scratch buffers in
//                       joint space.
//   p_out            -- caller-owned composite-step output (v + u).
template <typename Scalar, int N, typename HessianOp, typename TrialEval>
byrd_omojokun_result byrd_omojokun_composite_step(
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& z_k,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g,
    HessianOp&& hessian_op,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, N>>& A,
    const Eigen::Ref<const Eigen::VectorXd>& c,
    Scalar delta,
    Scalar eps,
    std::size_t max_cg_iter,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower_displaced,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper_displaced,
    Scalar f_old,
    Scalar c_norm_old,
    TrialEval&& trial_eval,
    Eigen::MatrixXd& AAt_workspace,
    Eigen::LDLT<Eigen::MatrixXd>& ldlt_workspace,
    Eigen::VectorXd& w_workspace,
    Eigen::Ref<Eigen::Vector<Scalar, N>> v_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> u_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> r_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> d_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> Bd_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> p_out)
{
    using std::sqrt;
    (void)z_k;  // z_k is documented as input; the helper operates in
                // step space (displaced bounds are pre-shifted by the
                // caller). Retained in the signature for clarity at
                // call sites and for future use by diagnostics.

    byrd_omojokun_result result{};
    result.normal_step_lsq_fallback = false;
    result.tangential_cg_status = cg_exit_status::forcing;

    const Scalar delta_n = zeta * delta;

    // ── Normal-step dogleg ─────────────────────────────────────────
    //
    // Reference: Nocedal and Wright 2e eq. 18.45-18.50; Lalee,
    //            Nocedal, Plantenga 1998 Section 3.2; scipy
    //            qp_subproblem.modified_dogleg.
    //
    // Solves min ||A v + c||^2 subject to ||v|| <= zeta * delta via
    // a dogleg interpolation between the scaled Cauchy point and the
    // minimum-norm LSQ Newton step.
    if(c.norm() == Scalar(0))
    {
        v_buf.setZero();
    }
    else
    {
        // Cauchy step: v_C = -alpha_C * g_n where g_n = A^T c and
        //              alpha_C = ||g_n||^2 / ||A g_n||^2.
        Eigen::Vector<Scalar, N> g_n = A.transpose() * c;
        Eigen::VectorXd Ag_n = A * g_n;
        const Scalar gn_sq = g_n.squaredNorm();
        const Scalar Agn_sq = Ag_n.squaredNorm();

        if(Agn_sq <= std::numeric_limits<Scalar>::epsilon())
        {
            // System locally consistent at v = 0; no normal step.
            v_buf.setZero();
        }
        else
        {
            const Scalar alpha_C = gn_sq / Agn_sq;
            Eigen::Vector<Scalar, N> v_C = -alpha_C * g_n;
            const Scalar v_C_norm = v_C.norm();

            if(v_C_norm >= delta_n)
            {
                // Cauchy direction truncated to the displaced TR
                // boundary along -g_n.
                const Scalar gn_norm = sqrt(gn_sq);
                v_buf.noalias() = -(delta_n / gn_norm) * g_n;
            }
            else
            {
                // LSQ Newton step via the shared LDLT warm-start
                // helper. equality_feasibility_warmstart computes
                //   p0 = A^T (A A^T)^{-1} b_eq  with  b_eq = c;
                // we want v_N = -p0.  LDLT (not LLT) is rank-
                // revealing via info() != Success on numerically
                // singular A A^T. Materialize a concrete Matrix and
                // VectorXd from the Eigen::Ref inputs because the
                // shared helper signature takes const lvalue refs to
                // concrete types (sqp_common.h:298-321).
                Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_dense = A;
                Eigen::VectorXd b_eq = c;
                Eigen::Vector<Scalar, N> v_N(g.size());

                detail::equality_feasibility_warmstart<Scalar, N, Eigen::Dynamic>(
                    A_dense, b_eq,
                    AAt_workspace, ldlt_workspace,
                    w_workspace, v_N);
                v_N *= Scalar(-1);

                if(ldlt_workspace.info() != Eigen::Success
                   || !v_N.allFinite())
                {
                    // Rank-deficient AA^T; LDLT flagged. Fall back
                    // to pure Cauchy truncated to the TR boundary.
                    const Scalar gn_norm = sqrt(gn_sq);
                    v_buf.noalias() = -(delta_n / gn_norm) * g_n;
                    result.normal_step_lsq_fallback = true;
                }
                else if(v_N.norm() <= delta_n)
                {
                    v_buf = v_N;
                }
                else
                {
                    // Dogleg interpolate between v_C and v_N along the
                    // segment, solving ||v_C + theta (v_N - v_C)|| =
                    // delta_n for theta in [0, 1]. Algebra and
                    // discriminant clamp transcribed verbatim from
                    // projected_gn_step.h:199-212 (a / d / disc / theta).
                    Eigen::Vector<Scalar, N> a = v_C;
                    Eigen::Vector<Scalar, N> d = v_N - v_C;
                    const Scalar a_sq = a.squaredNorm();
                    const Scalar d_sq = d.squaredNorm();
                    const Scalar a_dot_d = a.dot(d);
                    const Scalar delta_n_sq = delta_n * delta_n;
                    const Scalar disc =
                        a_dot_d * a_dot_d - d_sq * (a_sq - delta_n_sq);
                    const Scalar disc_clamped =
                        disc > Scalar(0) ? disc : Scalar(0);
                    Scalar theta = (-a_dot_d + sqrt(disc_clamped)) / d_sq;
                    theta = std::clamp(theta, Scalar(0), Scalar(1));
                    v_buf = a + theta * d;
                }
            }
        }
    }

    // ── Tangential-step Steihaug-CG ────────────────────────────────
    //
    // Reference: Nocedal and Wright 2e Section 7.3 Algorithm 7.2
    //            (truncated CG); Steihaug 1983 SIAM J. Numer. Anal.
    //            20(3):626-637.
    //
    // Tangential subproblem gradient at u = 0 is g_t = g + B v, per
    // the quadratic q_t(u) = (g + B v)^T u + 0.5 u^T B u. Steihaug-CG
    // operates with the tangential TR bound ||v + u|| <= delta; we
    // pass the FULL radius (not zeta * delta) and shift the displaced
    // box by -v_buf so the inner iterate starts from u = 0 on top of
    // v_buf.
    Eigen::Vector<Scalar, N> Bv = hessian_op(v_buf);
    Eigen::Vector<Scalar, N> g_t = g + Bv;

    // Compute the tangential radius: Steihaug-CG enforces ||u|| <=
    // delta_t where delta_t = sqrt(max(0, delta^2 - ||v||^2)) so
    // ||v + u||^2 = ||v||^2 + 2 v^T u + ||u||^2 stays below delta^2
    // when u is roughly orthogonal to v; in practice the conservative
    // delta_t below preserves the trust-region constraint on the
    // composite step for the common case where the tangential step is
    // approximately orthogonal to the normal step (Lalee-Nocedal-
    // Plantenga 1998 Section 3.3 derivation).
    const Scalar v_norm_sq = v_buf.squaredNorm();
    const Scalar delta_sq = delta * delta;
    const Scalar remainder = delta_sq - v_norm_sq;
    const Scalar delta_t =
        remainder > Scalar(0) ? sqrt(remainder) : Scalar(0);

    Eigen::Vector<Scalar, N> lower_t = lower_displaced - v_buf;
    Eigen::Vector<Scalar, N> upper_t = upper_displaced - v_buf;

    result.tangential_cg_status = detail::steihaug_cg<Scalar, N>(
        g_t, std::forward<HessianOp>(hessian_op),
        delta_t, eps,
        lower_t, upper_t,
        max_cg_iter,
        u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf);

    // ── Composite step ─────────────────────────────────────────────
    p_out = v_buf + u_buf;

    // ── Predicted reduction (augmented) ────────────────────────────
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3 eq. 3.4-3.5;
    //            scipy equality_constrained_sqp.py predicted-reduction
    //            formula.
    //
    //   pred = -g^T p - 0.5 * p^T B p + (||c||_2 - ||A p + c||_2)
    //
    // The two terms split the reduction into objective progress and
    // linearized-feasibility progress; this is the augmented-reduction
    // convention that handles v != 0 cleanly.
    Eigen::Vector<Scalar, N> Bp = hessian_op(p_out);
    const Scalar Bp_dot_p = p_out.dot(Bp);
    Eigen::VectorXd Ap_plus_c = A * p_out + c;
    const Scalar c_norm_lin = c.norm();
    const Scalar Ap_plus_c_norm = Ap_plus_c.norm();
    const Scalar predicted =
        -g.dot(p_out) - Scalar{0.5} * Bp_dot_p
        + (c_norm_lin - Ap_plus_c_norm);

    // ── Actual reduction ───────────────────────────────────────────
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3;
    //            scipy equality_constrained_sqp.py ratio test.
    //
    //   actual = (f_old + ||c(z_k)||_2)
    //          - (f_new + ||c(z_k + p)||_2)
    //
    // The L2 merit (objective + L2 constraint norm); matches scipy +
    // Nocedal-Wright convention.
    std::pair<Scalar, Scalar> trial = trial_eval(p_out);
    const Scalar f_new = trial.first;
    const Scalar c_norm_new = trial.second;
    const Scalar actual = (f_old + c_norm_old) - (f_new + c_norm_new);

    // ── Ratio and radius update ────────────────────────────────────
    //
    // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
    //            (universal defaults). max(pred, eps) guards 0/0 on
    //            degenerate steps; rho -> +/-inf is handled correctly
    //            by the standard threshold branches.
    const Scalar pred_guarded =
        std::max(predicted, std::numeric_limits<Scalar>::epsilon());
    const Scalar rho = actual / pred_guarded;
    const Scalar step_norm = p_out.norm();

    Scalar new_delta;
    bool accepted;
    if(rho < tr_eta_1)
    {
        accepted = false;
        new_delta = tr_shrink_factor * delta;
    }
    else
    {
        accepted = true;
        if(rho > tr_eta_2 && step_norm >= tr_boundary_guard * delta)
            new_delta = std::min(tr_expand_factor * delta, tr_delta_max);
        else
            new_delta = delta;
    }

    result.rho = rho;
    result.new_delta = new_delta;
    result.actual_reduction = actual;
    result.predicted_reduction = predicted;
    result.accepted = accepted;
    return result;
}

}

#endif
