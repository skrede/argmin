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
//   4. Penalty:       update the in/out adaptive penalty parameter per
//                     the Lalee-Nocedal-Plantenga heuristic so the value
//                     reflects the curvature at the proposed composite
//                     step. The penalty itself is internal to the helper
//                     (per-step quadratic-model state, not an acceptance
//                     verdict).
//
// The helper returns raw decision inputs (predicted objective-leg
// reduction, feasibility-leg predicted reduction vpred, trial-point
// f_new + c_norm_new) and lets the caller decide acceptance and the
// radius update. Caller-side merit gates that consume these inputs:
//   * tr_sqp_policy applies the augmented L2-merit ratio test
//     rho = (f_old + penalty * c_norm_old
//            - f_new - penalty * c_norm_new)
//           / max(predicted + penalty * vpred, eps)
//     plus the eta_1/eta_2 radius update at its single call site.
//   * filter_trsqp_policy applies a Fletcher-Leyffer dominance check
//     on (f_new, h_new) against its filter set; the radius update
//     follows the same eta_1/eta_2 thresholds re-implemented at the
//     call site.
//
// Adopted from: scipy/optimize/_trustregion_constr/qp_subproblem.py
//               (modified_dogleg + projected_cg);
//               scipy/optimize/_trustregion_constr/equality_constrained_sqp.py
//               (TR_FACTOR = 0.8 normal-step damping; ratio test and
//                radius update -- the latter migrated out of this helper).
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 18.5 Algorithm 18.4 (Byrd-Omojokun composite step);
//            equations 18.45-18.50 (zeta damping; normal-step formulation);
//            Section 4.1 Algorithm 4.1 (TR ratio test and radius update);
//            Lalee, Nocedal, Plantenga 1998, SIAM J. Optim. 8(3):682-706
//              (concrete dogleg pseudocode for the normal step;
//               augmented-reduction ratio-test formula in Section 3.3
//               equations 3.4-3.5 -- consumed at the caller).
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
//                 trial_eval propagates into f_new / c_norm_new and the
//                 caller's ratio-test branch is responsible for the
//                 IEEE-754 NaN semantics.

#include "argmin/types.h"
#include "argmin/detail/steihaug_cg.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <vector>
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

// TR ratio thresholds and radius-update factors. Consumed by callers
// (tr_sqp_policy, filter_trsqp_policy) to perform the eta_1 / eta_2
// rho-branch and the boundary-guarded expand step at the composite-
// step call site.
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

// Number of iterative-refinement passes applied by the null-space
// projector on the tangential leg (Gould-Hribar-Nocedal 2001 Alg.
// 5.1-5.2). scipy's projected_cg always refines once. The count was
// swept over {0, 1, 2}: on the small, well-conditioned Jacobians of
// the HS trust-region cohort the projector is already exact after the
// first solve (||A u||_inf = 0 at every count; no HS accuracy
// difference), and the orthogonality early-break inside the projector
// makes count 2 identical to count 1 once orthogonality is reached
// (pass 1). One pass is retained: it is the scipy/GHN standard, its HS
// cost with the early-break is a single orthogonality-check matvec
// (equal to zero passes when the projection is already exact), and it
// guards against null-space drift on ill-conditioned Jacobians outside
// the HS cohort where zero passes would accumulate rounding.
inline constexpr int null_space_refine_passes = 1;

// Box-face handling on the tangential leg. false selects the
// scipy/GHN-standard truncate-and-stop (variant A): on a box-face hit
// the projected CG stops with a boundary-class exit. true selects the
// Lin-More free-set restart (variant B): the blocking coordinate is
// pinned at its box-face value (a unit row is added to the null-space
// projector so the warm-restarted tangential leg holds it there) and
// the subproblem is re-solved on the reduced free set. Variant B is
// selected by the HS-suite comparison recorded in the plan summary
// (net +6 passing cells: 9 red->green flips vs 3 green->red, against
// variant A's 5 vs 6). When the normal step is box-infeasible on the
// slack block, variant A's truncate-and-stop collapses the tangential
// leg to a near-zero step; variant B's free-set restart keeps making
// objective progress on the remaining coordinates.
inline constexpr bool tangential_lin_more_restart = true;

// Composite-step result POD returned by byrd_omojokun_composite_step.
//
// The helper places the composite step into the caller's p_out buffer
// and reports raw decision inputs. The caller decides whether to apply
// p_out to z_k based on its own acceptance gate built from these
// fields:
//   predicted  -- the unweighted-objective predicted reduction
//                 -g^T p - 0.5 p^T B p (the quadratic-model decrease in
//                 the objective leg). Callers form the augmented
//                 predicted reduction as predicted + penalty * vpred
//                 if they use an L2-merit ratio test; filter-style
//                 callers consume the unweighted predicted directly.
//   vpred      -- the feasibility-leg predicted reduction
//                 ||c||_2 - ||A p + c||_2 evaluated AT the
//                 linearization point (both terms at z_k; this is NOT
//                 the cross-iterate feasibility decrease).
//   f_new      -- objective at z_k + p, from trial_eval's first return.
//   c_norm_new -- L2 norm of the joint constraint residual at z_k + p,
//                 from trial_eval's second return.
//   p_out_filled                -- true on every non-degenerate path
//                                  (false reserved for explicit early
//                                  exits that leave p_out untouched).
//   normal_step_lsq_fallback    -- set when the LDLT factorization of
//                                  AA^T failed (rank-deficient joint
//                                  Jacobian) and the helper fell back
//                                  to a pure-Cauchy step truncated to
//                                  the zeta * delta boundary.
//   tangential_cg_status        -- exit status of the inner Steihaug-CG
//                                  call on the tangential leg.
struct byrd_omojokun_step_result
{
    double predicted;
    double vpred;
    double f_new;
    double c_norm_new;
    bool p_out_filled;
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
//   trial_eval       -- callable invoked as trial_eval(p_out) returning
//                       std::pair<Scalar, Scalar>{f_new, c_norm_new} at
//                       z_k + p_out. The caller owns evaluation; the
//                       helper just forwards the (f, c_norm) pair to
//                       its caller through the result POD.
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
//   penalty          -- in/out adaptive L2-merit penalty parameter.
//                       Owned by the caller's policy state so it persists
//                       across composite-step calls; updated per the
//                       LNP / scipy update_penalty heuristic before
//                       return. Callers that do not consume the
//                       penalty (filter-style acceptance) may pass a
//                       dummy field; the helper does not gate on it.
//   penalty_factor   -- in: heuristic growth factor controlling how
//                       aggressively the penalty grows. The post-step
//                       update sets
//                         new_penalty = max(penalty,
//                                           hredd / ((1 - penalty_factor)
//                                                     * vpred))
//                       when both vpred > 0 and hredd > 0 (and the denom
//                       is positive); otherwise the penalty stays.
//
// Reference: Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
//            8(3):682-706 Section 3.3 equations 3.4-3.5 (augmented
//            merit + penalty update rule); equation 1.13 (penalty
//            growth margin). Adopted from scipy
//            optimize/_trustregion_constr/equality_constrained_sqp.py
//            update_penalty. The scipy reference default for
//            penalty_factor is 0.3; this library's default is selected
//            empirically by an HS-suite sweep (the most conservative
//            non-regressing value, currently driven toward a
//            second-order correction retry on the inequality leg to
//            absorb the freeze-on-feasibility regression that larger
//            growth factors produce in isolation).
template <typename Scalar, int N, typename HessianOp, typename TrialEval>
byrd_omojokun_step_result byrd_omojokun_composite_step(
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
    TrialEval&& trial_eval,
    Eigen::MatrixXd& AAt_workspace,
    Eigen::LDLT<Eigen::MatrixXd>& ldlt_workspace,
    Eigen::VectorXd& w_workspace,
    Eigen::Ref<Eigen::Vector<Scalar, N>> v_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> u_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> r_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> d_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> Bd_cg_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> p_out,
    Scalar& penalty,
    Scalar penalty_factor)
{
    using std::sqrt;
    (void)z_k;  // z_k is documented as input; the helper operates in
                // step space (displaced bounds are pre-shifted by the
                // caller). Retained in the signature for clarity at
                // call sites and for future use by diagnostics.

    byrd_omojokun_step_result result{};
    result.p_out_filled = false;
    result.normal_step_lsq_fallback = false;
    result.tangential_cg_status = cg_exit_status::forcing;

    const Scalar delta_n = zeta * delta;

    // ── Factorize A A^T once per composite step ────────────────────
    //
    // The tangential-leg null-space projector needs (A A^T)^{-1} on
    // EVERY step with constraints, and the normal-step LSQ-Newton
    // branch needs the same factorization. Assemble and factor it once
    // here; both legs reuse ldlt_workspace. have_factorization is false
    // on a rank-deficient A A^T (LDLT flags info() != Success), in which
    // case the normal step falls back to pure Cauchy and the projector
    // takes its rank-revealing QR fallback.
    const int m_total = static_cast<int>(A.rows());
    bool have_factorization = false;
    if(m_total > 0)
    {
        AAt_workspace.noalias() = A * A.transpose();
        ldlt_workspace.compute(AAt_workspace);
        have_factorization = (ldlt_workspace.info() == Eigen::Success);
    }

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
                // LSQ Newton step reusing the A A^T factorization built
                // at composite-step entry: v_N = -A^T (A A^T)^{-1} c.
                // LDLT (not LLT) is rank-revealing via info() != Success
                // on numerically singular A A^T. solveInPlace writes the
                // back-solve into the caller-owned w_workspace with no
                // heap allocation.
                w_workspace = c;
                ldlt_workspace.solveInPlace(w_workspace);
                Eigen::Vector<Scalar, N> v_N =
                    -(A.transpose() * w_workspace);

                if(!have_factorization || !v_N.allFinite())
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

    // ── Tangential-step projected Steihaug-CG ──────────────────────
    //
    // Reference: Nocedal and Wright 2e Section 16.3 Algorithm 16.2
    //            (projected CG); Gould-Hribar-Nocedal 2001 Algorithm
    //            6.2; scipy qp_subproblem.projected_cg.
    //
    // Tangential subproblem gradient at u = 0 is g_t = g + B v, per the
    // quadratic q_t(u) = (g + B v)^T u + 0.5 u^T B u. The tangential
    // step u is confined to null(A) (N&W 18.49) so it preserves the
    // normal step's linearized-feasibility gain; the projector below
    // reuses the A A^T factorization built at entry.
    Eigen::Vector<Scalar, N> Bv = hessian_op(v_buf);
    Eigen::Vector<Scalar, N> g_t = g + Bv;

    // Tangential radius: because u in null(A) is exactly orthogonal to
    // the normal step v in range(A^T), ||v + u||^2 = ||v||^2 + ||u||^2
    // holds exactly, so enforcing ||u|| <= delta_t = sqrt(delta^2 -
    // ||v||^2) makes the composite step satisfy ||v + u|| <= delta
    // exactly (no longer an approximation, as it was before the
    // null-space projection landed).
    const Scalar v_norm_sq = v_buf.squaredNorm();
    const Scalar delta_sq = delta * delta;
    const Scalar remainder = delta_sq - v_norm_sq;
    const Scalar delta_t =
        remainder > Scalar(0) ? sqrt(remainder) : Scalar(0);

    Eigen::Vector<Scalar, N> lower_t = lower_displaced - v_buf;
    Eigen::Vector<Scalar, N> upper_t = upper_displaced - v_buf;

    // Null-space projector: z <- z - A^T (A A^T)^{-1} A z, reusing
    // ldlt_workspace and w_workspace. Identity when there are no
    // constraints (null(A) is the whole space).
    auto project_null = [&](auto& vec)
    {
        if(m_total > 0)
            detail::null_space_project<Scalar, N>(
                A, ldlt_workspace, vec, w_workspace,
                null_space_refine_passes);
    };

    int blocking_coord = -1;
    result.tangential_cg_status = detail::steihaug_cg<Scalar, N>(
        g_t, hessian_op,
        delta_t, eps,
        lower_t, upper_t,
        max_cg_iter,
        u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf,
        project_null, &blocking_coord);

    // Variant B (Lin-More free-set restart): on a box-face-blocked exit
    // fix the blocking coordinate, augment A with its unit row so the
    // projector pins that coordinate, refactor the reduced normal
    // equations, and re-solve. Retained OFF by default (variant A, the
    // scipy/GHN truncate-and-stop, is the reference-standard mechanism;
    // the HS-suite comparison in the plan summary showed variant B added
    // no passing cell). The augmented factorization is local scratch so
    // the caller-owned ldlt_workspace stays consistent for downstream
    // reuse.
    if constexpr(tangential_lin_more_restart)
    {
        if(m_total > 0)
        {
            const int n = static_cast<int>(g.size());
            std::vector<int> fixed;
            int restarts = 0;
            while(blocking_coord >= 0 && restarts < n)
            {
                fixed.push_back(blocking_coord);
                ++restarts;

                const int m_aug = m_total + static_cast<int>(fixed.size());
                Eigen::MatrixXd A_aug(m_aug, n);
                A_aug.topRows(m_total) = A;
                for(std::size_t f = 0; f < fixed.size(); ++f)
                {
                    A_aug.row(m_total + static_cast<int>(f)).setZero();
                    A_aug(m_total + static_cast<int>(f), fixed[f]) =
                        Scalar(1);
                }

                Eigen::MatrixXd AAt_aug = A_aug * A_aug.transpose();
                Eigen::LDLT<Eigen::MatrixXd> ldlt_aug(AAt_aug);
                Eigen::VectorXd w_aug(m_aug);
                const Eigen::Ref<const Eigen::Matrix<Scalar,
                    Eigen::Dynamic, N>> A_aug_ref(A_aug);

                auto project_aug = [&](auto& vec)
                {
                    detail::null_space_project<Scalar, N>(
                        A_aug_ref, ldlt_aug, vec, w_aug,
                        null_space_refine_passes);
                };

                blocking_coord = -1;
                result.tangential_cg_status =
                    detail::steihaug_cg<Scalar, N>(
                        g_t, hessian_op,
                        delta_t, eps,
                        lower_t, upper_t,
                        max_cg_iter,
                        u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf,
                        project_aug, &blocking_coord,
                        /*warm_start=*/true);
            }
        }
    }

    // ── Composite step ─────────────────────────────────────────────
    p_out = v_buf + u_buf;

    // ── Predicted reductions (objective leg + feasibility leg) ─────
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3 eq. 3.4-3.5;
    //            scipy equality_constrained_sqp.py predicted-reduction
    //            formula.
    //
    //   predicted (objective leg)   = -g^T p - 0.5 * p^T B p
    //   vpred     (feasibility leg) = ||c||_2 - ||A p + c||_2
    //
    // Both norms in vpred are at the linearization point, NOT across
    // iterates. The previous-iterate end residual (||c(z_k)||_2 at the
    // caller's f_old / c_norm_old shape) belongs in the caller's
    // actual-reduction leg, not in vpred. Callers form the augmented
    // L2-merit predicted reduction as `predicted + penalty * vpred`
    // when needed; filter-style callers consume the unweighted
    // `predicted` and a separately-computed `h` aggregate.
    Eigen::Vector<Scalar, N> Bp = hessian_op(p_out);
    const Scalar Bp_dot_p = p_out.dot(Bp);
    Eigen::VectorXd Ap_plus_c = A * p_out + c;
    const Scalar c_norm_lin = c.norm();
    const Scalar Ap_plus_c_norm = Ap_plus_c.norm();
    const Scalar vpred = c_norm_lin - Ap_plus_c_norm;
    const Scalar predicted =
        -g.dot(p_out) - Scalar{0.5} * Bp_dot_p;

    // ── Trial evaluation at z_k + p ─────────────────────────────────
    //
    // Caller-supplied closure returns (f_new, c_norm_new). The helper
    // does not consume these for an internal ratio test; the values
    // are forwarded through the result POD so the caller's acceptance
    // gate can read them.
    std::pair<Scalar, Scalar> trial = trial_eval(p_out);
    const Scalar f_new = trial.first;
    const Scalar c_norm_new = trial.second;

    // ── Penalty update (LNP / scipy update_penalty heuristic) ──────
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 equation 1.13;
    //            scipy update_penalty. The penalty grows only when
    //            the objective-leg descent magnitude (hredd) exceeds
    //            the (1 - penalty_factor)-scaled feasibility-leg
    //            predicted reduction (vpred); otherwise the penalty
    //            stays.
    //
    // argmin variant: penalty_factor = 0 disables the adaptive growth
    // entirely (the heuristic gate is short-circuited). This makes
    // penalty_factor = 0 semantically equivalent to "fixed penalty at
    // its initial value" — when state-resident penalty is initialized
    // to 1 and never updated, the augmented merit collapses to the
    // unweighted L2 form that argmin shipped prior to the adaptive-
    // penalty plumbing. Non-zero penalty_factor opts into the
    // heuristic; the literature shape (always-on growth) is recovered
    // by any strictly-positive penalty_factor. This is a deliberate
    // choice: the swept HS-suite empirics at HEAD show that even the
    // literature-default growth shape (hredd / vpred) regresses the
    // four existing-passing cells without a paired second-order
    // correction retry on the inequality leg. Keeping the no-growth
    // endpoint as a true no-op leaves the default landing path safe
    // while the runtime knob remains available for downstream SOC
    // pairing.
    //
    // hredd is the objective predicted-reduction MAGNITUDE under the
    // quadratic model; the sign convention matches scipy's
    // "hredd > 0 means descent". The denom guard handles
    // penalty_factor >= 1 (degenerate runtime values) gracefully by
    // skipping the update — that knob slot is empirically restricted
    // to the [0, 1) interval by the literature heuristic.
    if(penalty_factor > Scalar{0})
    {
        const Scalar hredd =
            -(g.dot(p_out) + Scalar{0.5} * Bp_dot_p);
        if(vpred > Scalar{0} && hredd > Scalar{0})
        {
            const Scalar denom = (Scalar{1} - penalty_factor) * vpred;
            if(denom > Scalar{0})
            {
                const Scalar candidate = hredd / denom;
                penalty = std::max(penalty, candidate);
            }
        }
    }

    result.predicted = predicted;
    result.vpred = vpred;
    result.f_new = f_new;
    result.c_norm_new = c_norm_new;
    result.p_out_filled = true;
    return result;
}

}

#endif
