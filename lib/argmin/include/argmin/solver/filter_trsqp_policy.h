#ifndef HPP_GUARD_ARGMIN_SOLVER_FILTER_TRSQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_FILTER_TRSQP_POLICY_H

// Filter-acceptance trust-region SQP policy on the Byrd-Omojokun
// composite step.
//
// Replaces the inline L2-merit augmented ratio test (which lives in
// tr_sqp_policy at its composite-step call site) with a Fletcher-Leyffer
// dominance check on the (f, h) pair where
//   h(x) = sum |c_eq(x)| + sum max(0, -c_ineq(x))
// matches the filter_slsqp / filter_nw_sqp convention. The acceptance
// gate ships in two reject-mode variants behind a runtime knob:
//   * tr_shrink (Fletcher-Leyffer-Toint 2002 Section 3): the filter is
//     a Boolean acceptance check; on accept the step is always added to
//     the filter; on reject the trust-region radius shrinks and the
//     step is not added.
//   * switching_condition (Wachter-Biegler 2006 Section 2.3, eqs.
//     18-20, Algorithm A): acceptance requires the filter memory check
//     AND the eq. 18 current-iterate margin; a step is classified
//     f-type when the eq. 19 switching condition (on grad_f^T p, gated
//     by the theta_min floor) and the eq. 20 sufficient-objective-
//     reduction condition both hold. f-type accepted steps are NOT
//     added to the filter (Algorithm A step A-7 -- they are reversible
//     and must not pollute the filter); every other accepted step is
//     added. Rejection path is identical to tr_shrink.
//
// Inequalities are reformulated via slacks (c_ineq(x) + s = 0, s >= 0).
// The joint primal is z = (x, s) in R^{n + n_ineq}; the Lagrangian
// Hessian approximation lives on the joint space via
// dense_ldl_bfgs<double, N>.
//
// Feasibility-restoration prototype is wired in behind a four-way
// gate (trust_radius below floor, filter rejected the candidate,
// restoration_max_iter > 0, and h_current above the feasibility
// tolerance). When the gate fires, the policy delegates to the
// Levenberg-Marquardt feasibility-restoration helper in
// detail/restoration.h to drive ||c(x)||_2 below
// restoration_feasibility_tolerance; on convergence the policy
// resumes composite-step from the restored iterate with a BFGS reset
// and a refreshed (f, h) seed in the filter. On non-convergence the
// policy falls through to the trust-radius-collapse null-step
// emission and the stall-window machinery in basic_solver decides
// termination. The argmin variant is the minimal-viable LM
// simplification of the full IPOPT restoration phase (no slack
// reformulation, no multiplier leg, no in-restoration filter
// integration).
//
// argmin variant: filter acceptance on the Byrd-Omojokun composite step.
//                 Replaces the inline L2-merit augmented ratio test
//                 (which lives in tr_sqp_policy) with a Fletcher-Leyffer
//                 dominance check on (f, h) where h =
//                 sum|c_eq| + sum max(0, -c_ineq) matches the
//                 filter_slsqp / filter_nw_sqp convention. Dual
//                 reject-gate (TR-shrink vs switching-condition) runtime
//                 knob; the LM feasibility-restoration hook is opt-in
//                 via restoration_max_iter and delegates to the helper
//                 in detail/restoration.h.
//
// Reference: Fletcher and Leyffer 2002 Math. Programming 91:239-269
//            Section 2.1 (filter dominance);
//            Fletcher, Leyffer, Toint 2002 SIAM J. Optim. 13(1):44-59
//            Section 3 (filter-TR convergence theory; pure TR-shrink
//            gate);
//            Wachter and Biegler 2005 SIAM J. Optim. 16(1):1-31
//            Section 2.3 (switching condition; kappa, s defaults);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            Section 2.3 eq. 6 (filter envelope; gamma_f, gamma_h)
//            and Section 3.3 (IPOPT feasibility-restoration phase;
//            the LM helper in detail/restoration.h is a minimal
//            simplification thereof);
//            Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim. 8(3):682-706
//            Section 3.1 (v-optimal restoration; structurally similar
//            alternative to the LM restoration prototype used here);
//            scipy/optimize/_trustregion_constr (public-API reference).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/restoration.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/byrd_omojokun.h"
#include "argmin/detail/steihaug_cg.h"
#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/filter_acceptance.h"

#include "argmin/result/step_result.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace argmin
{

// argmin variant: closed-set Mode NTTP threaded through
//                 filter_trsqp_policy. `rebind<M>` preserves Mode on N
//                 rebind. Per-mode Steihaug-CG inner-iter cap,
//                 forcing-sequence variant, BFGS-skip-on-non-positive-
//                 curvature gate, and multiplier-reest stride are
//                 exposed as static-constexpr members. TR ratio
//                 thresholds and radius-update factors are uniform
//                 across modes per literature consensus. `accurate` is
//                 the default to preserve cross-family consistency with
//                 the line-search SQP policies.
//
// Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Section 7.3 Algorithm 7.2 (truncated CG);
//            Eisenstat and Walker 1996 SIAM J. Sci. Comput. 17(1):16-32
//            (accurate-mode adaptive forcing);
//            Dembo, Eisenstat, Steihaug 1982 SIAM J. Numer. Anal.
//            19(2):400-408 (fast-mode forcing sequence).
template <int N = dynamic_dimension, sqp_mode Mode = sqp_mode::accurate>
struct filter_trsqp_policy
{
    using scalar_type = double;
    static constexpr sqp_mode mode_ = Mode;

    template <int M>
    using rebind = filter_trsqp_policy<M, Mode>;

    // Filter-reject acceptance gate variant. tr_shrink follows
    // Fletcher-Leyffer-Toint 2002 Section 3 (filter is a Boolean
    // acceptance check; on accept always add to filter, on reject
    // shrink the trust region and do not add). switching_condition
    // follows Wachter-Biegler 2006 Section 2.3 / Algorithm A:
    // acceptance additionally requires the eq. 18 margin against the
    // current iterate; an accepted step is f-type when the eq. 19
    // switching condition on grad_f^T p (gated by theta_min) and the
    // eq. 20 sufficient-f-reduction condition both hold; f-type
    // accepted steps are NOT added to the filter (allowing revisits);
    // all other accepted steps are added (step A-7).
    //
    // Reference: Fletcher, Leyffer, Toint 2002 SIAM J. Optim.
    //            13(1):44-59 Section 3;
    //            Wachter and Biegler 2006 Math. Programming 106:25-57
    //            Section 2.3 eqs. 18-20, Algorithm A step A-7.
    enum class filter_reject_mode : std::uint8_t
    {
        tr_shrink,
        switching_condition
    };

    // Cross-cutting convergence tolerances (cross-family consistency
    // with kraft_slsqp, nw_sqp, filter_slsqp, filter_nw_sqp). A consumer
    // racing line-search vs trust-region policies under
    // basic_solver_group sees a uniform tolerance contract.
    static constexpr double default_gradient_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-8;
    static constexpr double default_step_tolerance_rel =
        (Mode == sqp_mode::fast) ? 1e-6 : 1e-12;
    static constexpr double default_feasibility_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-6;

    // BFGS-skip-on-non-positive-curvature gate (policy-level; the
    // dense_ldl_bfgs helper itself is mode-blind). Fast mode skips the
    // Powell-damping path when s^T y <= 0 and increments
    // diagnostics.bfgs_skip_count; accurate mode lets dense_ldl_bfgs
    // apply Powell damping per N&W eq. 18.22-18.24.
    //
    // Reference: Nocedal and Wright 2e Procedure 18.2 (damping guard);
    //            eq. 18.22-18.24 (Powell damping, accurate path).
    static constexpr bool default_bfgs_skip_on_nonpositive_curvature =
        (Mode == sqp_mode::fast);

    // Active-set Lagrange-multiplier re-estimation stride. The post-step
    // KKT-leg invokes detail::compute_kkt_multipliers_active_set only
    // when (s.iteration % multiplier_reest_every_k == 0); on the skip
    // path the prior step's kkt_lambda_eq_buf / kkt_mu_ineq_buf are
    // reused (state-resident, zero-alloc).
    //
    // Reference: Bertsekas 1996 Section 4.2 (stale-multiplier reuse
    //            rationale);
    //            Nocedal and Wright 2e Section 18.3 + Algorithm 18.3
    //            (working-set identification).
    static constexpr std::size_t default_multiplier_reest_every_k =
        (Mode == sqp_mode::fast) ? std::size_t{5} : std::size_t{1};

    // Steihaug-CG inner-iteration cap MULTIPLIER (resolved against
    // n + n_ineq at policy init time). Fast caps at the textbook
    // exact-arithmetic bound; accurate doubles it to absorb inexact-
    // Hessian (BFGS) effects.
    //
    // Reference: Nocedal and Wright 2e Section 7.3 (truncated CG
    //            converges in <= n exact-arithmetic iterations on SPD B).
    static constexpr std::size_t default_max_cg_iterations_multiplier =
        (Mode == sqp_mode::fast) ? std::size_t{1} : std::size_t{2};

    // Forcing-sequence dispatch is resolved at the step() site via
    // `if constexpr (Mode == sqp_mode::fast)`:
    //   accurate: Eisenstat-Walker 1996  eps_k = min(0.5, sqrt(||grad L||))
    //   fast:     Dembo-Eisenstat-Steihaug 1982  eps_k = min(0.1, ||grad L||)
    // Both forcing sequences are documented in Nocedal and Wright 2e
    // Section 7.3 + Conn, Gould, Toint 2000 Chapter 7 Section 7.4.

    // Uniform-across-modes trust-region parameters.
    //
    // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
    //            (universal trust-region defaults; consistent across
    //            scipy trust-constr, Ceres trust-region minimizer,
    //            KNITRO).
    static constexpr double default_initial_trust_radius = 1.0;
    static constexpr double default_min_trust_radius     = 1e-12;

    // Second-order correction retry budget. On a rejected primary
    // composite step whose trial violation did not decrease (the
    // Maratos regime), the policy iterates the SOC retry up to this
    // many times: each retry assembles the SOC residual
    // c(z + p) - A_joint p at the trial point with the original
    // Jacobian (detail::assemble_joint_soc), then re-calls the
    // composite-step helper with the corrected residual. The trust-
    // region radius is held fixed across retries (the retry exploits
    // curvature information from the trial-point residual, not radius
    // expansion); between retries the violation must contract by
    // detail::kappa_soc or the loop aborts.
    //
    // The zero default is re-derived (2026-07-05, lockstep with
    // tr_sqp_policy) against the corrected SOC residual
    // c(z+p) - A p. With the residual assembled correctly the filter
    // acceptance closes its HS-suite closure targets without a SOC
    // retry, so the retry budget re-locks to 0 as an opt-in knob
    // rather than a default; a Maratos-curvature workload opts in via
    // `policy.options.soc_max_iterations`.
    //
    // Reference: Nocedal and Wright 2e Section 18.3 (Maratos effect
    //            and SOC pairing); Wachter and Biegler 2006 Math.
    //            Programming 106:25-57 Section 2.4, Algorithm A steps
    //            A-5.7..A-5.9.
    static constexpr std::size_t default_soc_max_iterations =
        std::size_t{0};

    // Empirically-selected per-mode defaults from an HS-suite sweep
    // across the filter envelope (gamma_f, gamma_h), the reject-gate
    // variant (tr_shrink vs switching_condition), the switching-
    // condition parameters (since re-based onto the Wachter-Biegler
    // eq. 19/20 form at the literature-fixed Section 4 constants;
    // they sit on the post-fix re-sweep register), and the
    // restoration budget.
    // All knobs except `default_initial_trust_radius` and
    // `default_min_trust_radius` are per-mode dispatched through the
    // ternary block below; the trust-radius defaults are uniform
    // across modes per the empirical sweep outcome (the picked
    // initial radius coincided across modes and was lifted out of
    // the per-mode block above). The per-mode ternary follows the
    // cross-policy mode-dispatch convention: `accurate` prioritizes
    // convergence quality, `fast` prioritizes per-step wall. The two
    // modes happen to coincide on the current pick for the
    // empirically-swept axes; the ternary shape is preserved so the
    // per-mode independence convention is structurally visible and a
    // future re-tune can split them without a code-shape change.
    //
    // Reference: Fletcher and Leyffer 2002 Math. Programming
    //            91:239-269 Section 2.1 (filter dominance);
    //            Fletcher, Leyffer, Toint 2002 SIAM J. Optim.
    //            13(1):44-59 Section 3 (filter-TR convergence theory;
    //            tr_shrink gate);
    //            Wachter and Biegler 2005 SIAM J. Optim. 16(1):1-31
    //            Section 2.3 (kappa, s switching condition);
    //            Wachter and Biegler 2006 Math. Programming
    //            106:25-57 Section 2.3 eq. 6 (gamma envelope) and
    //            Section 3.3 (restoration phase; the argmin
    //            implementation is a minimal-viable Levenberg-
    //            Marquardt simplification, see
    //            detail/restoration.h).

    // Filter envelope margins (Wachter and Biegler 2006 Section 2.3,
    // eq. 6).
    static constexpr double default_filter_gamma_f =
        (Mode == sqp_mode::fast) ? 1e-2 : 1e-2;
    static constexpr double default_filter_gamma_h =
        (Mode == sqp_mode::fast) ? 1e-2 : 1e-2;

    // Filter-reject acceptance gate variant. tr_shrink follows
    // Fletcher-Leyffer-Toint 2002 Section 3; switching_condition is a
    // Wachter-Biegler 2006 Section 2.3 / Algorithm A port to the
    // trust-region composite step and remains exposed as a runtime
    // knob.
    static constexpr filter_reject_mode default_filter_reject_mode =
        (Mode == sqp_mode::fast) ? filter_reject_mode::tr_shrink
                                 : filter_reject_mode::tr_shrink;

    // Switching-condition parameters in the Wachter-Biegler form
    // (eq. 19: (-grad_f^T p)^s_phi > delta * h_current^s_theta;
    // eq. 20: f_trial <= f_current + eta_phi * grad_f^T p). Values
    // are the literature-fixed W-B Section 4 defaults (delta = 1,
    // s_theta = 1.1, s_phi = 2.3, eta_phi = 1e-4); they sit on the
    // post-fix re-sweep register but ship at the cited constants.
    // Consumed only when reject_mode == switching_condition. The
    // per-mode ternary shape is preserved per the cross-policy
    // mode-dispatch convention (both modes currently coincide).
    //
    // Reference: Wachter and Biegler 2006 Math. Programming
    //            106:25-57 Section 4 (default constants).
    static constexpr double default_filter_switching_delta =
        (Mode == sqp_mode::fast) ? 1.0 : 1.0;
    static constexpr double default_filter_switching_s_theta =
        (Mode == sqp_mode::fast) ? 1.1 : 1.1;
    static constexpr double default_filter_switching_s_phi =
        (Mode == sqp_mode::fast) ? 2.3 : 2.3;
    static constexpr double default_filter_switching_eta_phi =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-4;

    // Feasibility-restoration knob defaults (minimal-viable
    // Levenberg-Marquardt prototype). The per-mode default budget is
    // the smallest restoration_max_iter that closes the L2-merit
    // overshoot family on the cross-cell sweep; callers that need a
    // larger budget can still set the option directly.
    //
    // Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-
    //            Marquardt for least-squares);
    //            Wachter and Biegler 2006 Math. Programming
    //            106:25-57 Section 3.3 (IPOPT restoration phase;
    //            full version).
    static constexpr std::size_t default_restoration_max_iter =
        (Mode == sqp_mode::fast) ? std::size_t{10} : std::size_t{10};
    static constexpr double default_restoration_lambda_init = 1e-3;

    struct options_type
    {
        // Direct-field-default form: literature-defaulted scalars are
        // brace-initialized from the per-mode default_* static-constexpr
        // members above.
        //
        // Reference: Nocedal and Wright 2e Section 4.1 (universal TR
        //            radius defaults).
        double initial_trust_radius{default_initial_trust_radius};
        double min_trust_radius{default_min_trust_radius};

        // Steihaug-CG inner-iter cap MULTIPLIER; resolved against
        // n + n_ineq at the step() call site.
        std::size_t max_cg_iterations_multiplier{
            default_max_cg_iterations_multiplier};

        // Stall window carried over from the line-search SQP family for
        // basic_solver convergence-framework parity.
        std::uint16_t stall_window{50};

        // Active-set multiplier re-estimation stride. A value of 0 is
        // treated as 1 (re-estimate every step) by the read-site clamp.
        std::size_t multiplier_reest_every_k{
            default_multiplier_reest_every_k};

        // Maximum number of second-order correction retries on the
        // inequality leg of the composite step. Zero disables SOC retry.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.1;
        //            kraft_slsqp_policy line-search SOC analog.
        std::size_t soc_max_iterations{default_soc_max_iterations};

        // Filter envelope margins (Wachter and Biegler 2006 Section 2.3,
        // eq. 6). Independent asymmetric margins; gamma_f scales the
        // objective dominance slack, gamma_h scales the constraint
        // violation dominance slack.
        double gamma_f{default_filter_gamma_f};
        double gamma_h{default_filter_gamma_h};

        // Filter-reject gate variant. tr_shrink follows
        // Fletcher-Leyffer-Toint 2002 Section 3 (always add accepted
        // steps to the filter; shrink the trust region on reject).
        // switching_condition follows Wachter-Biegler 2006 Section 2.3
        // / Algorithm A (eq. 18 current-iterate margin required for
        // acceptance; f-type accepted steps -- eq. 19 switching AND
        // eq. 20 sufficient reduction -- are NOT added; all other
        // accepted steps are added; rejection path identical to
        // tr_shrink).
        filter_reject_mode reject_mode{default_filter_reject_mode};

        // Switching-condition parameters in the W-B form (consumed
        // only when reject_mode == switching_condition). delta and
        // s_theta shape the eq. 19 right-hand side
        // delta * h_current^s_theta, s_phi is the eq. 19 exponent on
        // the directional term -grad_f^T p, and eta_phi is the eq. 20
        // Armijo-type sufficient-reduction fraction. Defaults are the
        // literature-fixed Wachter-Biegler 2006 Section 4 constants.
        double filter_switching_delta{default_filter_switching_delta};
        double filter_switching_s_theta{
            default_filter_switching_s_theta};
        double filter_switching_s_phi{default_filter_switching_s_phi};
        double filter_switching_eta_phi{
            default_filter_switching_eta_phi};

        // Feasibility-restoration knobs (minimal-viable Levenberg-
        // Marquardt prototype). The restoration hook fires only on the
        // three-way conjunction
        //   (trust_radius < min_trust_radius) AND
        //   (filter rejected the most recent candidate)         AND
        //   (restoration_max_iter > 0);
        // any single conjunct false leaves the policy's
        // pre-restoration null-step emission and filter-reject paths
        // unchanged. The opt-in-zero default keeps the policy behavior
        // bit-identical for callers that do not explicitly enable
        // restoration.
        //
        // Buffer-aliasing rule: the restoration helper uses
        // DEDICATED state buffers (restoration_c, restoration_c_trial,
        // restoration_J, restoration_JtJ, restoration_rhs,
        // restoration_dx, restoration_xtrial, restoration_ldlt) sized
        // at init() time when restoration_max_iter > 0. The shared
        // sqp_state_buffers c_all / c_trial_buf are NOT passed into the
        // helper -- they have live readers in the trial-evaluation
        // closure and the filter h-evaluation block, and the
        // restoration helper's inner-loop mutations during the LM
        // iteration would corrupt step-internal state.
        //
        // Init-time invariant: restoration_max_iter MUST be set BEFORE
        // basic_solver construction. The lazy-init guard in init()
        // reads the option once at policy construction; mid-solve
        // changes do not re-size the restoration_* buffers and the
        // hook would dereference unsized buffers.
        //
        // Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-
        //            Marquardt for least-squares);
        //            Wachter and Biegler 2006 Math. Programming
        //            106:25-57 Section 3.3 (IPOPT restoration phase;
        //            this prototype is a minimal-viable LM
        //            simplification).
        std::size_t restoration_max_iter{default_restoration_max_iter};
        double restoration_lambda_init{default_restoration_lambda_init};
        double restoration_feasibility_tolerance{
            default_feasibility_tolerance};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;

        // Cross-policy state-resident buffer struct. Filter-TRSQP uses
        // the x-only-typed bufs for the constraint-axis buffers (c_all,
        // c_trial_buf, J_all, kkt_lambda_eq_buf, kkt_mu_ineq_buf) sized
        // at the x-only dimension n; the joint-space (x, s) buffers are
        // dynamic-dimension fields below (joint_*) because the joint
        // primal width n + n_ineq is a runtime quantity even when the
        // policy's N template parameter is a fixed compile-time
        // constant for the x-only dimension.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        // Joint-space (x, s) dynamic buffers. Sized at n + n_ineq on
        // init; the joint primal dimension is runtime even when N is
        // fixed.
        Eigen::VectorXd joint_x_old_buf;
        Eigen::VectorXd joint_grad_L_old_buf;
        Eigen::VectorXd joint_grad_L_new_buf;
        Eigen::VectorXd joint_sk_buf;
        Eigen::VectorXd joint_yk_buf;

        // Slack vector for the inequality reformulation
        // c_ineq(x) + s = 0, s >= 0.
        Eigen::VectorXd s_slack;

        // Current trust-region radius; updated by the filter-acceptance
        // dispatch inside step(). The skeleton seeds it from
        // options.initial_trust_radius at init / reset time.
        double trust_radius{default_initial_trust_radius};

        // BO-internal state; updated by the LNP heuristic inside
        // byrd_omojokun_composite_step ONLY when penalty_factor > 0.
        // filter_trsqp passes penalty_factor = 0.0 at the composite-step
        // call site (byrd_omojokun.h penalty_factor > Scalar{0}
        // strict-positive guard), so the state-resident penalty stays
        // inert. The acceptance gate here is (f, h) filter dominance,
        // not an augmented-merit ratio test.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3.
        double penalty{1.0};

        // LDLT workspace pair used by
        // detail::equality_feasibility_warmstart on the joint Jacobian
        // inside the byrd_omojokun normal-step leg.
        Eigen::MatrixXd AAt_workspace;
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;
        Eigen::VectorXd w_workspace;

        // Restoration helper buffers. Allocated only when
        // options.restoration_max_iter > 0 at policy construction time
        // (lazy-init guard keeps zero-restoration callers at the
        // pre-restoration memory footprint). The c and c_trial buffers
        // are DEDICATED to the restoration helper -- the shared
        // sqp_state_buffers c_all / c_trial_buf cannot be aliased here
        // because they have live readers in the trial_eval closure and
        // the filter h-evaluation block, and the restoration helper's
        // inner-loop mutations would corrupt step-internal state on the
        // failed-restoration fall-through path.
        //
        // Reference: Nocedal and Wright 2e Section 10.3.
        Eigen::VectorXd              restoration_c;
        Eigen::VectorXd              restoration_c_trial;
        Eigen::MatrixXd              restoration_J;
        Eigen::MatrixXd              restoration_JtJ;
        Eigen::Vector<double, N>     restoration_rhs;
        Eigen::Vector<double, N>     restoration_dx;
        Eigen::Vector<double, N>     restoration_xtrial;
        Eigen::LDLT<Eigen::MatrixXd> restoration_ldlt;

        // Snapshot of the iterate at restoration entry. The LM helper
        // mutates s.x in place across its accepted inner steps, so a
        // failed (or filter-unacceptable) restoration must restore the
        // entry iterate to keep s.x consistent with the state-resident
        // f / g / c / J evaluations that were NOT refreshed on the
        // failure path.
        Eigen::Vector<double, N>     restoration_x_entry;

        // Restoration budget latch. Set when a restoration attempt
        // exhausts its budget (or exits without producing an iterate
        // acceptable to the augmented filter); while set, the
        // restoration entry gate short-circuits so a failed
        // restoration does not re-burn its full budget on every
        // subsequent step of a terminal stall window. Cleared on any
        // ACCEPTED step (x moved, so the local feasibility landscape
        // changed) and on reset()/reset_clear(). Parameter-free by
        // design: the latch is a pure state machine on events the
        // policy already observes, so there is no numeric knob to
        // sweep.
        bool restoration_exhausted{false};

        double objective_value{};
        // Lagrangian-Hessian approximation on the JOINT (x, s) space;
        // sized n + n_ineq.
        //
        // Reference: Fletcher and Powell 1974 Math. Computation 28:1067-
        //            1078 (LDL^T rank-1 update);
        //            Nocedal and Wright 2e Section 7.2 (limited-memory
        //            BFGS);
        //            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (BFGS in
        //            the SQP outer loop).
        detail::dense_ldl_bfgs<double, Eigen::Dynamic> hessian;

        // Filter-set member; the acceptance gate at the composite-step
        // call site is dominance-based, not augmented-merit-based.
        //
        // Reference: Fletcher and Leyffer 2002 Math. Programming
        //            91:239-269 Section 2.1;
        //            Wachter and Biegler 2006 Math. Programming
        //            106:25-57 Section 2.3 eq. 6.
        argmin::detail::filter_set<double> filter;

        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "filter_trsqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "filter_trsqp_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int m = s.n_eq + s.n_ineq;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // Cross-policy state-resident buffer struct sized on the x-only
        // dimension n; the constraint-axis buffers hold un-reformulated
        // x-only quantities cross-family consistent with the
        // line-search SQP family. The slack-augmented joint-space
        // buffers are separate dynamic-dimension fields sized below.
        s.bufs.resize(n, s.n_eq, s.n_ineq);

        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);

        // Joint (x, s) dynamic buffers sized on n + n_ineq.
        const int n_joint = n + s.n_ineq;
        s.joint_x_old_buf.setZero(n_joint);
        s.joint_grad_L_old_buf.setZero(n_joint);
        s.joint_grad_L_new_buf.setZero(n_joint);
        s.joint_sk_buf.setZero(n_joint);
        s.joint_yk_buf.setZero(n_joint);

        // LDLT workspace for the normal-step LSQ leg. Sized on the
        // joint constraint count m_joint = n_eq + n_ineq.
        const int m_joint = s.n_eq + s.n_ineq;
        if(m_joint > 0)
        {
            s.AAt_workspace.resize(m_joint, m_joint);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(m_joint);
            s.w_workspace.resize(m_joint);
        }

        // Lazy-init restoration helper buffers. Allocated only when
        // restoration is opted in; the zero-default keeps the
        // post-init memory footprint identical to the
        // pre-restoration ship-state for callers that leave the knob
        // at its default. The x-only constraint count
        // m_xonly = n_eq + n_ineq is consumed by the helper (not the
        // joint slack-augmented count).
        if(options.restoration_max_iter > 0)
        {
            const int m_xonly = s.n_eq + s.n_ineq;
            if(m_xonly > 0)
            {
                s.restoration_c.resize(m_xonly);
                s.restoration_c_trial.resize(m_xonly);
                s.restoration_J.resize(m_xonly, n);
                s.restoration_JtJ.resize(n, n);
            }
            s.restoration_rhs.resize(n);
            s.restoration_dx.resize(n);
            s.restoration_xtrial.resize(n);
            s.restoration_x_entry.resize(n);
            s.restoration_ldlt = Eigen::LDLT<Eigen::MatrixXd>(n);
        }

        // Constraint evaluation on the un-reformulated x-only shape.
        Eigen::VectorXd c_eval(m);
        if(m > 0)
            problem.constraints(x0, c_eval);
        s.c_eq = c_eval.head(s.n_eq);
        s.c_ineq = c_eval.tail(s.n_ineq);

        Eigen::MatrixXd J_eval(m, n);
        if(m > 0)
            problem.constraint_jacobian(x0, J_eval);
        s.J_eq = J_eval.topRows(s.n_eq);
        s.J_ineq = J_eval.bottomRows(s.n_ineq);

        // Box bounds: detected via concept.
        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(n, inf);
        }

        // Slack initialization: s_0 = max(c_ineq(x_0), 0) enforces
        // s_0 >= 0 and zeroes the joint residual c_ineq - s at every
        // feasible initial iterate.
        s.s_slack = s.c_ineq.cwiseMax(0.0).eval();

        // Lagrangian-Hessian on the joint (x, s) space. dense_ldl_bfgs
        // initializes to identity; the slack block picks up curvature
        // through (s_k, y_k) pairs naturally on subsequent push() calls.
        s.hessian = detail::dense_ldl_bfgs<double, Eigen::Dynamic>(
            n + s.n_ineq);

        s.lambda = Eigen::VectorXd::Zero(m);

        s.trust_radius = options.initial_trust_radius;
        s.iteration = 0;

        // Initialize the filter with h_max = 1e4 * max(1, h_0) per the
        // Wachter-Biegler 2006 eq. 8 ceiling and the entry floor
        // theta_min = 1e-4 * max(1, h_0) per W-B 2006 Section 4 (so an
        // exactly-feasible visited point cannot turn the filter into a
        // permanent monotone-f gate), thread the configured envelope
        // margins onto it, and seed it with (f_0, h_0) so the first
        // trial's dominance check is well-defined. h_0 follows the L1
        // aggregate convention (sum |c_eq| + sum max(0, -c_ineq))
        // verbatim from the filter_slsqp closure-capture pattern.
        //
        // Reference: Wachter and Biegler 2006 eq. (8) and Section 4
        //            (theta_max / theta_min scaling);
        //            Fletcher and Leyffer 2002 Section 2.1 (the
        //            initial filter is seeded with the iter-0 point so
        //            dominance is well-defined on trial 0).
        double h_0 = 0.0;
        if(s.n_eq > 0)
            h_0 += s.c_eq.cwiseAbs().sum();
        if(s.n_ineq > 0)
            h_0 += (-s.c_ineq).cwiseMax(0.0).sum();
        s.filter.initialize(1e4 * std::max(1.0, h_0),
                            1e-4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f, options.gamma_h);
        s.filter.add(s.objective_value, h_0);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Trust-region SQP step with filter acceptance: Byrd-Omojokun
        // composite step on the slack-reformulated joint variable
        // z = (x, s); a Fletcher-Leyffer dominance check on (f, h)
        // replaces the L2-merit augmented ratio test at the call site.
        //
        // Reference: Fletcher and Leyffer 2002 Section 2.1
        //            (filter dominance);
        //            Fletcher, Leyffer, Toint 2002 Section 3
        //            (filter-TR convergence theory; TR-shrink gate);
        //            Wachter and Biegler 2005 Section 2.3
        //            (switching condition);
        //            Wachter and Biegler 2006 Section 2.3 eq. 6
        //            (filter envelope);
        //            Nocedal and Wright 2e Section 18.5 Algorithm 18.4
        //            (Byrd-Omojokun composite step);
        //            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
        //            8(3):682-706.

        const int n = static_cast<int>(s.x.size());
        const int n_ineq = s.n_ineq;
        const int n_total = n + n_ineq;
        const int m_total = s.n_eq + n_ineq;
        std::size_t bfgs_skip_count = 0;

        constexpr double inf = std::numeric_limits<double>::infinity();

        // Runtime envelope retune so options.gamma_f / options.gamma_h
        // changes between solve() calls take effect on the next step
        // without requiring a reset_clear().
        s.filter.set_envelope(options.gamma_f, options.gamma_h);

        // Iteration-entry constraint-violation aggregate h_current
        // (Fletcher-Leyffer L1 aggregate). Captured BEFORE the BO call
        // and re-used across the primary step and any SOC retry: the
        // iterate s.x has not moved between attempts, so the "current"
        // violation is the same scalar across both.
        double h_current = 0.0;
        if(s.n_eq > 0)
            h_current += s.c_eq.cwiseAbs().sum();
        if(n_ineq > 0)
            h_current += (-s.c_ineq).cwiseMax(0.0).sum();

        // Section B -- Joint Lagrangian gradient at the current iterate.
        //
        // x-block: g - J_eq^T lambda_eq - J_ineq^T mu_ineq
        // s-block: +mu_ineq (the joint multiplier consistent with the
        //          x-block is nu = -mu on the slack-equality rows, so
        //          grad_s L = -nu = +mu; derivation at
        //          detail::assemble_joint_lagrangian_gradient. This
        //          block does NOT vanish at a KKT point -- it equals
        //          the slack-bound multiplier zeta = mu -- which is
        //          why stationarity is measured with the box-projected
        //          form in Section E, not the raw norm).
        argmin::detail::assemble_joint_lagrangian_gradient<double, N>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            n, n_ineq, s.joint_grad_L_new_buf);

        // Bound-aware composite KKT residual at the current iterate
        // (Section S and the null-step emission paths). The
        // stationarity leg is the box-projected form, so kkt-gated
        // convergence can fire at solutions with an active box bound
        // (the raw leg equals the unmodeled bound multiplier there).
        auto bound_aware_kkt = [&s]() -> double {
            return argmin::detail::kkt_residual<double,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x, s.lower, s.upper);
        };

        // Section C -- Joint Jacobian A and joint residual c.
        //
        // The slack reformulation casts inequalities c_ineq(x) >= 0 as
        // c_ineq(x) - s = 0 with s >= 0; in argmin's external sign
        // convention (c_ineq >= 0 feasible) the helper-internal
        // residual is -c_ineq + s, with the joint Jacobian's inequality
        // row block negated to match the linearization.
        //
        // Reference:
        //  scipy/optimize/_trustregion_constr/canonical_constraint.py.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A_joint;
        Eigen::VectorXd c_joint;
        A_joint.setZero(m_total, n_total);
        c_joint.resize(m_total);
        if(s.n_eq > 0)
        {
            A_joint.topLeftCorner(s.n_eq, n) = s.J_eq;
            c_joint.head(s.n_eq) = s.c_eq;
        }
        if(n_ineq > 0)
        {
            A_joint.bottomLeftCorner(n_ineq, n) = -s.J_ineq;
            A_joint.bottomRightCorner(n_ineq, n_ineq).setIdentity();
            c_joint.tail(n_ineq) = -s.c_ineq + s.s_slack;
        }

        // Section D -- Displaced joint bounds: (lower - z_k, upper - z_k).
        // Steihaug-CG projects on the displaced box at each inner
        // iteration, and the box-projected stationarity measure in
        // Section E reads the same displaced box.
        Eigen::VectorXd lower_joint(n_total);
        Eigen::VectorXd upper_joint(n_total);
        lower_joint.head(n) = (s.lower - s.x).template cast<double>();
        upper_joint.head(n) = (s.upper - s.x).template cast<double>();
        if(n_ineq > 0)
        {
            lower_joint.tail(n_ineq) = -s.s_slack;
            upper_joint.tail(n_ineq).setConstant(inf);
        }

        // Section E -- Forcing tolerance eps_k.
        //
        // Reference: Eisenstat and Walker 1996 (accurate-mode);
        //            Dembo, Eisenstat, Steihaug 1982 (fast-mode).
        // The forcing sequence reads the BOX-PROJECTED joint
        // stationarity measure ||P(z - grad_L, l, u) - z||_inf rather
        // than the raw joint gradient norm: the raw norm is floored at
        // mu on inequality-active problems (the +mu slack block of the
        // joint gradient), which would cap the attainable inner-solve
        // accuracy at sqrt(mu_max) under the Eisenstat-Walker schedule.
        // At interior points the two measures coincide. The forcing
        // formulas themselves are unchanged; only the driving norm is.
        const double proj_stationarity =
            argmin::detail::projected_stationarity_displaced<double>(
                s.joint_grad_L_new_buf, lower_joint, upper_joint);
        double eps_k;
        if constexpr (Mode == sqp_mode::fast)
            eps_k = std::min(0.1, proj_stationarity);
        else
            eps_k = std::min(0.5, std::sqrt(proj_stationarity));

        // Section F -- Steihaug-CG inner-iteration cap.
        const std::size_t max_cg_iter =
            options.max_cg_iterations_multiplier
            * static_cast<std::size_t>(n_total);

        // Section G -- Hessian-vector-product closure.
        auto hessian_op = [&s](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            return s.hessian.multiply(v);
        };

        // Section H -- Trial-evaluation closure. Materializes z + p as
        // x_trial / s_trial; queries the problem callbacks at x_trial;
        // populates s.bufs.c_trial_buf with the ORIGINAL constraint
        // values [c_eq(x_trial); c_ineq(x_trial)] so the caller's
        // filter h_trial computation can read it via VectorBlock
        // head/tail (filter_slsqp_policy idiom; no per-call VectorXd
        // materialization on the filter path); assembles the joint
        // residual under the slack reformulation sign convention;
        // returns (f_new, ||c_joint_new||_2).
        auto trial_eval = [&s, n, n_ineq](const Eigen::VectorXd& p)
            -> std::pair<double, double>
        {
            Eigen::Vector<double, N> x_trial = s.x + p.head(n);
            Eigen::VectorXd s_trial(n_ineq);
            if(n_ineq > 0)
                s_trial = s.s_slack + p.tail(n_ineq);

            const double f_new = s.problem->value(x_trial);

            if(s.n_eq + n_ineq > 0)
                s.problem->constraints(x_trial, s.bufs.c_trial_buf);

            Eigen::VectorXd c_joint_new(s.n_eq + n_ineq);
            if(s.n_eq > 0)
                c_joint_new.head(s.n_eq)
                    = s.bufs.c_trial_buf.head(s.n_eq);
            if(n_ineq > 0)
                c_joint_new.tail(n_ineq)
                    = -s.bufs.c_trial_buf.tail(n_ineq) + s_trial;
            return {f_new, c_joint_new.norm()};
        };

        // Section I -- Baseline scalars (the f_old leg of the
        // unweighted-objective predicted-vs-actual comparison used by
        // the radius rho rule; c_norm_old is not consumed here because
        // the filter does not need an L2 cross-iterate diff).
        const double f_old = s.objective_value;

        // Section J -- Per-step workspaces in joint primal space.
        Eigen::VectorXd z_k(n_total);
        z_k.head(n) = s.x;
        if(n_ineq > 0)
            z_k.tail(n_ineq) = s.s_slack;

        // Joint OBJECTIVE gradient: x-block grad f, slack block zero
        // (the slack variables do not enter f). Consumed by the
        // helper's predicted objective-leg reduction -- an f-model
        // quantity, matching the f-measured actual_obj in the radius
        // rho rule and the W-B eq. (14) f-model reduction consumed by
        // the switching-condition classification. The joint LAGRANGIAN
        // gradient (Section B) keeps driving the quadratic step model.
        Eigen::VectorXd g_obj_joint(n_total);
        g_obj_joint.head(n) = s.g;
        if(n_ineq > 0)
            g_obj_joint.tail(n_ineq).setZero();

        Eigen::VectorXd v_buf(n_total);
        Eigen::VectorXd u_buf(n_total);
        Eigen::VectorXd r_cg_buf(n_total);
        Eigen::VectorXd d_cg_buf(n_total);
        Eigen::VectorXd Bd_cg_buf(n_total);
        Eigen::VectorXd p_out(n_total);

        // Section K -- Byrd-Omojokun composite step. The helper returns
        // raw decision inputs and we apply the filter dispatch locally
        // below. filter_trsqp passes penalty_factor = 0.0 because the
        // LNP merit update is not the acceptance gate here; filter
        // dominance is. The BO-internal penalty state remains for
        // symmetry with tr_sqp_policy but is inert at this call site:
        // byrd_omojokun.h gates the update on penalty_factor > Scalar{0}
        // (strict-positive), so 0.0 is a documented no-op.
        //
        // Snapshot the pre-step trust radius so the SOC retry can
        // re-invoke the helper without compounding the radius shrink
        // (Lalee, Nocedal, Plantenga 1998 Section 3.1: the retry
        // exploits curvature from the trial-point residual, not radius
        // expansion).
        const double delta_pre_step = s.trust_radius;
        auto bo_step = argmin::detail::byrd_omojokun_composite_step<
            double, Eigen::Dynamic>(
            z_k, s.joint_grad_L_new_buf, g_obj_joint,
            hessian_op,
            A_joint, c_joint,
            s.trust_radius, eps_k, max_cg_iter,
            lower_joint, upper_joint,
            trial_eval,
            s.AAt_workspace, s.ldlt_feasibility, s.w_workspace,
            v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
            s.penalty, 0.0);

        // Directional objective-model term grad_f^T p of the PRIMARY
        // composite step, consumed by the switching-condition
        // classification (W-B eq. 19 is defined on the directional
        // term, an f-model quantity; the slack block of the joint
        // objective gradient is zero so only the x-block contributes).
        // Computed ONCE from the primary step and NOT refreshed across
        // SOC retries: W-B Section 2.4 keeps the original direction in
        // the switching test (19) and the right-hand side of (20) for
        // second-order-corrected trials.
        //
        // Reference: Wachter and Biegler 2006 Section 2.3 eq. (19),
        //            Section 2.4 (SOC keeps the uncorrected direction
        //            in the switching and Armijo tests).
        const double grad_f_dot_p = s.g.dot(p_out.head(n));

        // Filter dispatch lambda. Computes h_trial from the
        // post-trial_eval s.bufs.c_trial_buf via the VectorBlock
        // head/tail closure-capture idiom (filter_slsqp_policy:639-643;
        // no per-call VectorXd materialization), runs the filter
        // dominance check, and applies the eta_1 / eta_2 rho-rule
        // radius update against the UNWEIGHTED-OBJECTIVE predicted
        // reduction (Fletcher-Leyffer-Toint 2002 Section 3: the
        // filter-TR radius rule uses unweighted f-predicted, NOT the
        // augmented-merit form). h_current is captured by reference
        // from the enclosing scope; it is the iteration-entry
        // violation, NOT refreshed between primary step and SOC retry,
        // because s.x has not moved.
        // Trial-point violation aggregate (Fletcher-Leyffer L1 form,
        // same measure as h_current) read from the c_trial_buf that
        // trial_eval populated during the most recent helper call.
        // Consumed by the filter dispatch below AND by the Section K'
        // SOC trigger / continuation gates (the filter family's theta
        // measure).
        auto compute_h_trial = [&s]() -> double
        {
            double h_trial = 0.0;
            if(s.n_eq > 0)
                h_trial += s.bufs.c_trial_buf.head(s.n_eq)
                              .cwiseAbs().sum();
            if(s.n_ineq > 0)
                h_trial += (-s.bufs.c_trial_buf.tail(s.n_ineq))
                              .cwiseMax(0.0).sum();
            return h_trial;
        };

        auto compute_accept_and_radius = [&s, &h_current, &p_out,
                                          &compute_h_trial,
                                          grad_f_dot_p,
                                          f_old, this](
            const argmin::detail::byrd_omojokun_step_result& bo,
            double delta_input)
            -> std::pair<bool, double>
        {
            const double h_trial = compute_h_trial();

            // Acceptance + augmentation verdict per reject-gate
            // variant. tr_shrink (default): plain filter memory check,
            // always add on accept (FLT-style TR filter practice, made
            // safe by the theta_min entry floor -- an accepted exactly-
            // feasible point enters at violation theta_min instead of
            // turning the filter into a monotone-f gate).
            // switching_condition: W-B Algorithm A -- the shared
            // helper enforces the eq. 18 current-iterate margin on top
            // of the filter memory check and classifies f-type steps
            // (eq. 19 switching AND eq. 20 sufficient reduction, both
            // on the primary step's grad_f^T p); genuine f-type
            // accepts leave the filter unchanged, all other accepts
            // augment it (step A-7). f_old is the current iterate's
            // objective (s.x has not moved between attempts).
            //
            // Reference: Fletcher, Leyffer, Toint 2002 Section 3
            //            (tr_shrink); Wachter and Biegler 2006
            //            Section 2.3 eqs. 18-20, Algorithm A steps
            //            A-5.3, A-5.4, A-7 (switching_condition).
            bool acceptable;
            bool add_to_filter;
            if(options.reject_mode
               == filter_reject_mode::switching_condition)
            {
                const argmin::detail::wb_switching_params<double>
                    wb_params{options.filter_switching_delta,
                              options.filter_switching_s_theta,
                              options.filter_switching_s_phi,
                              options.filter_switching_eta_phi};
                const auto verdict =
                    argmin::detail::wb_switching_acceptance<double>(
                        s.filter, f_old, h_current,
                        bo.f_new, h_trial, grad_f_dot_p,
                        options.gamma_f, options.gamma_h,
                        s.filter.theta_min(), wb_params);
                acceptable    = verdict.accept;
                add_to_filter = verdict.augment;
            }
            else
            {
                acceptable = s.filter.is_acceptable(bo.f_new, h_trial);
                add_to_filter = acceptable;
            }

            if(!acceptable)
            {
                return {false,
                        argmin::detail::tr_shrink_factor * delta_input};
            }

            // Filter-accepted step: radius rule is decoupled from
            // acceptance. Expand on a very-successful objective-leg
            // prediction (rho > tr_eta_2 with the step at the trust
            // boundary); otherwise hold the radius. The shrink path
            // belongs only on filter rejection (handled above at the
            // !acceptable branch), not on h-type feasibility-progress
            // accepts where the objective worsens.
            //
            // Reference: Fletcher, Leyffer, Toint 2002 Section 3 --
            //            filter-TR decouples acceptance from radius
            //            update.
            // Explicit no-expand branch on a non-positive predicted
            // reduction: N&W Algorithm 4.1's ratio test assumes
            // pred > 0. Forming rho = actual / max(pred, eps) here
            // would produce |rho| ~ 1e16 on a legitimately negative
            // prediction and expand the radius on a model-increasing
            // step. The radius is held; only a genuinely very-
            // successful objective-leg prediction expands.
            const double actual_obj = f_old - bo.f_new;
            const double step_norm = p_out.norm();

            double new_delta = delta_input;
            if(bo.predicted > 0.0)
            {
                const double rho = actual_obj / bo.predicted;
                if(rho > argmin::detail::tr_eta_2
                   && step_norm
                      >= argmin::detail::tr_boundary_guard
                         * delta_input)
                    new_delta = std::min(
                        argmin::detail::tr_expand_factor * delta_input,
                        argmin::detail::tr_delta_max);
            }

            // Filter augmentation per the verdict computed above. The
            // classification used the ITERATION-ENTRY h_current (NOT
            // refreshed across primary step and SOC retry, because
            // s.x has not moved).
            if(add_to_filter)
                s.filter.add(bo.f_new, h_trial);

            return {true, new_delta};
        };

        auto [accepted, new_delta] =
            compute_accept_and_radius(bo_step, s.trust_radius);

        // Section K' -- Second-order correction retry. On a rejected
        // primary composite step in the MARATOS REGIME (trial filter
        // violation did not decrease: h_trial >= h_current, the
        // theta(z + p) >= theta(z_k) gate of Wachter-Biegler A-5.7),
        // re-assemble the SOC residual
        //   c_soc = c(z_k + p) - A_joint p
        // at the trial point with the ORIGINAL Jacobian A_joint via
        // the shared detail::assemble_joint_soc helper: subtracting
        // A_joint p preserves the linear constraint model and leaves
        // only the curvature remainder for the corrective re-solve
        // (N&W Section 18.3; Wachter-Biegler 2006 eq. 24/27 in the
        // full-step convention). The helper is re-called with
        // c_joint_soc; the trust radius input is held at
        // delta_pre_step across retries (no chained shrinks). An
        // ordinary rejection whose violation already fell is driven
        // by the objective leg; the retry stays out (a NaN h_trial
        // fails the >= comparison and skips the retry too).
        //
        // Continuation: between retries the trial violation must
        // contract by detail::kappa_soc (Wachter-Biegler A-5.9) or
        // the loop aborts; a NaN retry violation fails the negated
        // contraction test and aborts.
        //
        // The retry's filter dispatch reads the SAME h_current
        // captured at iteration entry (BEFORE the original BO call):
        // the iterate s.x has not moved between attempts. Refreshing
        // h_current would be both a no-op and a code-smell-inviting
        // bug.
        //
        // Reference: Nocedal and Wright 2e Section 18.3 (Maratos
        //            effect and SOC pairing); Wachter and Biegler
        //            2006 Math. Programming 106:25-57 Section 2.4,
        //            Algorithm A steps A-5.7..A-5.9.
        std::size_t soc_retry_count = 0;
        if(!accepted && options.soc_max_iterations > 0
           && compute_h_trial() >= h_current)
        {
            Eigen::Vector<double, N> x_trial(n);
            Eigen::VectorXd s_trial(n_ineq);
            Eigen::VectorXd c_all_trial(m_total);
            Eigen::VectorXd c_trial_joint(m_total);
            Eigen::VectorXd c_joint_soc(m_total);

            // Filter-measure violation of the most recent trial,
            // driving the kappa_soc contraction test across retries.
            double theta_trial_prev = compute_h_trial();

            for(std::size_t soc_iter = 0;
                soc_iter < options.soc_max_iterations;
                ++soc_iter)
            {
                ++soc_retry_count;

                x_trial.noalias() = s.x + p_out.head(n);
                if(n_ineq > 0)
                    s_trial.noalias() = s.s_slack + p_out.tail(n_ineq);

                if(m_total > 0)
                    s.problem->constraints(x_trial, c_all_trial);
                if(s.n_eq > 0)
                    c_trial_joint.head(s.n_eq) =
                        c_all_trial.head(s.n_eq);
                if(n_ineq > 0)
                    c_trial_joint.tail(n_ineq) =
                        -c_all_trial.tail(n_ineq) + s_trial;

                // SOC residual through the shared assembly helper:
                // c_soc = c_trial_joint - A_joint * p_out.
                argmin::detail::assemble_joint_soc<double,
                                                   Eigen::Dynamic>(
                    c_trial_joint, A_joint, p_out, c_joint_soc);

                bo_step = argmin::detail::byrd_omojokun_composite_step<
                    double, Eigen::Dynamic>(
                    z_k, s.joint_grad_L_new_buf, g_obj_joint,
                    hessian_op,
                    A_joint, c_joint_soc,
                    delta_pre_step, eps_k, max_cg_iter,
                    lower_joint, upper_joint,
                    trial_eval,
                    s.AAt_workspace, s.ldlt_feasibility, s.w_workspace,
                    v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
                    s.penalty, 0.0);

                std::tie(accepted, new_delta) =
                    compute_accept_and_radius(bo_step, delta_pre_step);

                if(accepted)
                    break;

                // Continuation gate (Wachter-Biegler A-5.9): abort
                // unless the retry's trial violation contracted by
                // kappa_soc relative to the previous trial. The
                // negated form routes a NaN violation to the abort
                // branch.
                const double theta_trial_new = compute_h_trial();
                if(!(theta_trial_new
                     < argmin::detail::kappa_soc * theta_trial_prev))
                    break;
                theta_trial_prev = theta_trial_new;
            }
        }

        // Section L -- Radius update consumes the caller-side verdict.
        s.trust_radius = new_delta;

        // Feasibility-restoration hook. Fires only on the five-way
        // conjunction
        //   (trust_radius < min_trust_radius)                  AND
        //   (filter rejected the most recent candidate)        AND
        //   (restoration_max_iter > 0)                         AND
        //   (h_current > restoration_feasibility_tolerance)    AND
        //   (no exhausted restoration attempt latched).
        // Any single conjunct false leaves the original null-step
        // emission path (Section M) and the filter-reject path
        // (Section N) untouched. The fourth conjunct is the
        // feasibility guard: restoration is a feasibility-restoration
        // phase, vacuous on an already-feasible iterate (Fletcher,
        // Leyffer, Toint 2002 Section 3 -- restoration applies on
        // infeasibility-driven stalls only). A feasible-iterate
        // Maratos-type stall is NOT restoration's problem: with the
        // theta_min entry floor on the filter (exactly-feasible
        // entries no longer force strict monotone-f) and the
        // corrected SOC retry, that stall escapes through the
        // second-order correction instead, so the gate keeps the
        // h_current > tol condition without needing to fire at
        // feasible points. The fifth conjunct is the budget latch: a
        // restoration attempt that exhausted its budget (or exited
        // without an iterate acceptable to the augmented filter) does
        // not re-burn restoration_max_iter LM iterations on every
        // subsequent step of a terminal stall window; the latch
        // clears when an accepted step moves x or on reset().
        //
        // Filter-augmentation order (W-B Algorithm A step A-9): the
        // iterate that TRIGGERED restoration is added to the filter
        // BEFORE restoration runs, and the restoration result must be
        // acceptable to the AUGMENTED filter; the restoration output
        // itself is never added (W-B practice -- adding it would
        // anchor the filter at the restored point).
        //
        // On converged-and-acceptable restoration the policy resumes
        // composite-step at the restored iterate with a BFGS reset
        // (the restoration step is a feasibility step, not a
        // Lagrangian step; the prior B approximation is stale). On a
        // failed attempt (budget exhausted, degenerate, or output not
        // acceptable to the augmented filter) the entry iterate is
        // restored from the snapshot -- the LM helper mutates s.x in
        // place across its accepted inner steps, and the state-
        // resident f / g / c / J are NOT refreshed on the failure
        // path -- the latch is set, and control falls through to the
        // Section M emission below; the iters_used count is
        // propagated to the diagnostics field so a downstream sweep
        // can attribute the outcome.
        //
        // Reference: Nocedal and Wright 2e Section 10.3;
        //            Wachter and Biegler 2006 Math. Programming
        //            106:25-57 Section 3.3 and Algorithm A steps
        //            A-7, A-9 (augment-then-restore order);
        //            Fletcher, Leyffer, Toint 2002 SIAM J. Optim.
        //            13(1):44-59 Section 3 (feasibility-restoration
        //            applicability gate).
        std::size_t restoration_iters_used_out = 0;
        if(s.trust_radius < options.min_trust_radius
           && !accepted
           && options.restoration_max_iter > 0
           && h_current > options.restoration_feasibility_tolerance
           && !s.restoration_exhausted)
        {
            // W-B A-9: augment the filter with the rejected current
            // iterate (f_k, h_k) BEFORE entering restoration. The
            // restoration exit below is checked against this
            // augmented filter.
            s.filter.add(s.objective_value, h_current);

            // Snapshot the entry iterate; the LM helper mutates s.x
            // in place and the failure path must undo that to keep
            // s.x consistent with the un-refreshed state evaluations.
            s.restoration_x_entry = s.x;

            // x-space (not joint slack-augmented) constraint and
            // jacobian closures. The LM helper operates on the
            // original x with the original c(x); slack lifting is a
            // filter_trsqp internal that does not belong in the
            // feasibility-restoration objective.
            auto cfn = [&s](const Eigen::Vector<double, N>& xv,
                            Eigen::VectorXd&                 cv)
            {
                s.problem->constraints(xv, cv);
            };
            auto jfn = [&s](const Eigen::Vector<double, N>& xv,
                            Eigen::MatrixXd&                 Jv)
            {
                s.problem->constraint_jacobian(xv, Jv);
            };

            const auto restoration_outcome =
                argmin::detail::feasibility_restoration<double, N>(
                    s.x, cfn, jfn, s.lower, s.upper,
                    options.restoration_max_iter,
                    options.restoration_lambda_init,
                    options.restoration_feasibility_tolerance,
                    s.restoration_c, s.restoration_c_trial,
                    s.restoration_J, s.restoration_JtJ,
                    s.restoration_rhs, s.restoration_dx,
                    s.restoration_xtrial, s.restoration_ldlt);
            restoration_iters_used_out =
                restoration_outcome.iterations_used;

            // Restoration exit acceptability (W-B A-9): the restored
            // iterate must be acceptable to the filter as augmented
            // with the triggering iterate above. Probed on locals
            // BEFORE committing any state so the reject path only has
            // to undo the helper's in-place x mutation. c_trial_buf is
            // free here: the trial-evaluation closure has no further
            // readers this iteration.
            bool restoration_acceptable = false;
            double restoration_h = 0.0;
            double restoration_f = 0.0;
            if(restoration_outcome.status
               == argmin::detail::restoration_status::converged)
            {
                restoration_f = s.problem->value(s.x);
                if(m_total > 0)
                    s.problem->constraints(s.x, s.bufs.c_trial_buf);
                if(s.n_eq > 0)
                    restoration_h += s.bufs.c_trial_buf.head(s.n_eq)
                                        .cwiseAbs().sum();
                if(n_ineq > 0)
                    restoration_h +=
                        (-s.bufs.c_trial_buf.tail(n_ineq))
                            .cwiseMax(0.0).sum();
                restoration_acceptable =
                    s.filter.is_acceptable(restoration_f,
                                           restoration_h);
            }

            if(restoration_acceptable)
            {
                // BFGS reset on the post-restoration iterate. The
                // restoration step is a feasibility step, not a
                // Lagrangian step; the prior LDL B approximation is
                // stale.
                s.hessian.reset();
                s.trust_radius = options.initial_trust_radius;

                // Clear the active-set multiplier buffers. They
                // correspond to the pre-restoration (J_eq, J_ineq)
                // linearization and would otherwise contaminate the
                // next iter's grad_L = grad_f - J^T lambda
                // computation (Section B reads them directly). Under
                // fast mode the active-set re-estimate stride is 5,
                // so a stale multiplier can persist for up to 5
                // iters post-restoration; zeroing here forces the
                // next-iter grad_L to collapse to grad_f and lets
                // Section P re-estimate from scratch.
                if(s.n_eq > 0)
                    s.bufs.kkt_lambda_eq_buf.head(s.n_eq).setZero();
                if(n_ineq > 0)
                    s.bufs.kkt_mu_ineq_buf.head(n_ineq).setZero();

                // Re-evaluate f, g, c, J at the restored s.x into the
                // SHARED sqp_state_buffers; the restoration helper
                // buffers are no longer consumed after this point in
                // the iteration.
                s.objective_value = s.problem->value(s.x);
                s.problem->gradient(s.x, s.g);
                if(m_total > 0)
                {
                    s.problem->constraints(s.x, s.bufs.c_all);
                    s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                    s.c_eq   = s.bufs.c_all.head(s.n_eq);
                    s.c_ineq = s.bufs.c_all.tail(n_ineq);
                    s.J_eq   = s.bufs.J_all.topRows(s.n_eq);
                    s.J_ineq = s.bufs.J_all.bottomRows(n_ineq);
                }

                // Refresh the slack vector to keep the joint residual
                // c_ineq - s nominally zero at the restored iterate
                // (mirroring the init()-time s_slack seeding).
                if(n_ineq > 0)
                    s.s_slack = s.c_ineq.cwiseMax(0.0).eval();

                // The restoration OUTPUT is deliberately NOT added to
                // the filter (W-B practice: the filter was augmented
                // with the TRIGGERING iterate before restoration ran,
                // step A-9; adding the output would anchor the filter
                // at the restored point and block the very region the
                // solver is about to re-enter).
                ++s.iteration;
                auto r = argmin::detail::null_step_result<
                    double, N, Eigen::Dynamic, Eigen::Dynamic>(
                    s.objective_value, s.g, s.J_eq, s.J_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                    s.c_eq, s.c_ineq, s.x.norm(),
                    /*bfgs_reset_count=*/std::size_t{1},
                    solver_status::running);
                r.kkt_residual = bound_aware_kkt();
                r.diagnostics.bfgs_skip_count = bfgs_skip_count;
                r.diagnostics.soc_retry_count = soc_retry_count;
                r.diagnostics.restoration_iters_used =
                    restoration_iters_used_out;
                return r;
            }
            // Failed attempt: budget exhausted / degenerate LM exit,
            // or a converged output that is not acceptable to the
            // augmented filter. Undo the helper's in-place x mutation
            // (the state-resident f / g / c / J still describe the
            // entry iterate) and set the budget latch so subsequent
            // steps of the stall window do not re-burn the budget.
            // Control falls through to the Section M null-step
            // emission below; the iters_used count is captured into
            // diagnostics on that emission path so the failed-
            // restoration case still surfaces the iter count to
            // downstream sweep consumers.
            s.x = s.restoration_x_entry;
            s.restoration_exhausted = true;
        }

        // Section M -- Trust-radius collapse path. Below the policy-
        // level floor the policy emits a null-step carrying the new
        // solver_status entry; the convergence framework's stall
        // window then declares termination after stall_window
        // consecutive null-steps.
        if(s.trust_radius < options.min_trust_radius)
        {
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), /*bfgs_reset_count=*/0,
                solver_status::trust_region_step_rejected);
            r.kkt_residual = bound_aware_kkt();
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            r.diagnostics.soc_retry_count = soc_retry_count;
            r.diagnostics.restoration_iters_used =
                restoration_iters_used_out;
            ++s.iteration;
            return r;
        }

        // Section N -- Single-step rejection branch (filter rejects but
        // radius above the floor). Iterate unchanged; the radius has
        // already been shrunk by the filter dispatch above.
        if(!accepted)
        {
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), /*bfgs_reset_count=*/0);
            r.kkt_residual = bound_aware_kkt();
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            r.diagnostics.soc_retry_count = soc_retry_count;
            ++s.iteration;
            return r;
        }

        // Section O -- Step acceptance.
        //
        // An accepted step moves x, so the local feasibility landscape
        // a latched failed restoration was stuck in has changed: clear
        // the budget latch so a future stall may attempt restoration
        // again.
        s.restoration_exhausted = false;

        // Save the joint old iterate plus the OLD-point objective
        // gradient and Jacobians for the fixed-multiplier BFGS pair
        // (N&W 18.13 evaluates BOTH gradient legs against the NEW
        // multipliers, so the old-point quantities must survive the
        // re-evaluation below; mirrors the line-search family's
        // J_all_old retention consumed by compute_bfgs_pair_fused).
        s.joint_x_old_buf.head(n) = s.x;
        if(n_ineq > 0)
            s.joint_x_old_buf.tail(n_ineq) = s.s_slack;
        s.bufs.g_old_buf = s.g;
        if(s.n_eq > 0)
            s.bufs.J_all_old.topRows(s.n_eq) = s.J_eq;
        if(n_ineq > 0)
            s.bufs.J_all_old.bottomRows(n_ineq) = s.J_ineq;

        s.x.noalias() += p_out.head(n);
        if(n_ineq > 0)
            s.s_slack.noalias() += p_out.tail(n_ineq);

        s.objective_value = s.problem->value(s.x);
        s.problem->gradient(s.x, s.g);
        if(m_total > 0)
        {
            s.problem->constraints(s.x, s.bufs.c_all);
            s.problem->constraint_jacobian(s.x, s.bufs.J_all);
            s.c_eq = s.bufs.c_all.head(s.n_eq);
            s.c_ineq = s.bufs.c_all.tail(n_ineq);
            s.J_eq = s.bufs.J_all.topRows(s.n_eq);
            s.J_ineq = s.bufs.J_all.bottomRows(n_ineq);
        }

        // Section P -- Active-set Lagrange-multiplier re-estimation at
        // x_{k+1}.
        //
        // Reference: Bertsekas 1996 Section 4.2;
        //            Nocedal and Wright 2e Section 18.3 + Algorithm
        //            18.3.
        ++s.iteration;
        if(m_total > 0)
        {
            const std::size_t reest_stride = std::max<std::size_t>(
                options.multiplier_reest_every_k, std::size_t{1});
            if(s.iteration % reest_stride == 0)
            {
                s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
                s.bufs.kkt_mu_ineq_buf.setZero(n_ineq);
                argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                                  Eigen::Dynamic,
                                                                  Eigen::Dynamic>(
                    s.g, s.J_eq, s.J_ineq, s.c_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
            }
        }
        else
        {
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(n_ineq);
        }

        // Section Q -- Joint BFGS curvature pair (s_k, y_k) on the
        // (x, s) space via the shared fixed-multiplier helper. Both
        // gradient legs are evaluated against the NEW multiplier
        // estimate (N&W 2e eq. 18.13): the old leg re-uses the
        // retained old-point gradient / Jacobians from Section O, the
        // new leg lands in joint_grad_L_new_buf for the stationarity
        // report in Section S. The slack block of y is structurally
        // zero (grad_s L = +mu is z-independent under fixed
        // multipliers).
        z_k.head(n) = s.x;
        if(n_ineq > 0)
            z_k.tail(n_ineq) = s.s_slack;
        argmin::detail::compute_joint_bfgs_pair<double, N>(
            s.bufs.g_old_buf, s.g,
            s.bufs.J_all_old.topRows(s.n_eq),
            s.bufs.J_all_old.bottomRows(n_ineq),
            s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            z_k, s.joint_x_old_buf, n, n_ineq,
            s.joint_grad_L_old_buf, s.joint_grad_L_new_buf,
            s.joint_sk_buf, s.joint_yk_buf);

        // Section R -- BFGS curvature-pair push under the per-mode
        // skip-on-non-positive-curvature gate.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            Nocedal and Wright 2e Procedure 18.2 + eq.
        //            18.22-18.24.
        const double sTy = s.joint_sk_buf.dot(s.joint_yk_buf);
        if(s.joint_sk_buf.norm() > 1e-15)
        {
            if constexpr (Mode == sqp_mode::fast)
            {
                if(sTy > 0.0)
                    s.hessian.push(s.joint_sk_buf, s.joint_yk_buf);
                else
                    ++bfgs_skip_count;
            }
            else
            {
                if(sTy > 0.0)
                    s.hessian.push(s.joint_sk_buf, s.joint_yk_buf);
            }
        }

        // Section S -- Bound-aware KKT residual on the x-only
        // un-reformulated problem. gradient_norm reports the
        // box-projected joint stationarity measure
        // ||P(z - grad_L, l, u) - z||_inf (the raw joint norm is
        // floored at mu by the +mu slack block at inequality-active
        // solutions and is not a stationarity measure there; at
        // interior points the two coincide).
        //
        // Reference: Nocedal and Wright 2e Definition 12.1;
        //            Section 12.3 + eq. 12.34;
        //            Section 16.7 (projected-gradient optimality).
        const double kkt = bound_aware_kkt();

        // Re-displace the joint box at the accepted iterate for the
        // projected stationarity report (the Section D box was
        // displaced at the pre-step iterate; the upper slack leg stays
        // +inf).
        lower_joint.head(n) = (s.lower - s.x).template cast<double>();
        upper_joint.head(n) = (s.upper - s.x).template cast<double>();
        if(n_ineq > 0)
            lower_joint.tail(n_ineq) = -s.s_slack;
        const double grad_L_norm_new =
            argmin::detail::projected_stationarity_displaced<double>(
                s.joint_grad_L_new_buf, lower_joint, upper_joint);

        return step_result<double>{
            .objective_value      = s.objective_value,
            .gradient_norm        = grad_L_norm_new,
            .step_size            = p_out.norm(),
            .objective_change     = s.objective_value - f_old,
            // Acceptance branch: the filter accepted the step. The
            // improved bit reports objective progress on the
            // unweighted f; constraint progress is implicit in the
            // filter add.
            .improved             = s.objective_value < f_old,
            .constraint_violation = detail::primal_feasibility_inf(
                                        s.c_eq, s.c_ineq),
            .x_norm               = s.x.norm(),
            .kkt_residual         = kkt,
            .diagnostics = {
                .bfgs_reset_count = 0,
                .bfgs_skip_count  = bfgs_skip_count,
                .nan_eval_count   = 0,
                .soc_retry_count  = soc_retry_count,
            },
        };
    }

    // Hot start -- preserves the BFGS Hessian. The filter is cleared
    // and RESEEDED here (clear-then-seed, exactly as reset_clear
    // seeds): filter_set::clear() erases the init()-time seed along
    // with everything else, so a reset without a reseed would leave
    // the next solve's first dominance check against an empty filter
    // with a stale h_max ceiling. Stale cross-run state is zeroed as
    // well: the KKT multiplier buffers (so the first post-reset joint
    // Lagrangian gradient does not consume multipliers estimated at
    // an unrelated point), the inert BO-internal penalty, and the
    // restoration budget latch.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = static_cast<int>(x0.size());
        const int m = s.n_eq + s.n_ineq;
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);

        Eigen::VectorXd c_eval(m);
        if(m > 0)
            s.problem->constraints(x0, c_eval);
        s.c_eq = c_eval.head(s.n_eq);
        s.c_ineq = c_eval.tail(s.n_ineq);

        Eigen::MatrixXd J_eval(m, n);
        if(m > 0)
            s.problem->constraint_jacobian(x0, J_eval);
        s.J_eq = J_eval.topRows(s.n_eq);
        s.J_ineq = J_eval.bottomRows(s.n_ineq);

        s.s_slack = s.c_ineq.cwiseMax(0.0).eval();
        s.trust_radius = options.initial_trust_radius;
        // Inert at this policy's composite-step call site
        // (penalty_factor = 0.0 keeps the LNP update a no-op) but
        // re-seeded anyway so no cross-run value survives a reset.
        s.penalty = 1.0;
        s.iteration = 0;
        s.restoration_exhausted = false;

        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);

        // Clear-then-seed at the new x0 (same seeding as init /
        // reset_clear): h_max = 1e4 * max(1, h_0), entry floor
        // theta_min = 1e-4 * max(1, h_0).
        //
        // Reference: Wachter and Biegler 2006 eq. (8) and Section 4.
        s.filter.clear();
        double h_0 = 0.0;
        if(s.n_eq > 0)
            h_0 += s.c_eq.cwiseAbs().sum();
        if(s.n_ineq > 0)
            h_0 += (-s.c_ineq).cwiseMax(0.0).sum();
        s.filter.initialize(1e4 * std::max(1.0, h_0),
                            1e-4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f, options.gamma_h);
        s.filter.add(s.objective_value, h_0);
    }

    // Cold restart -- additionally clears the BFGS Hessian and the
    // multiplier estimate; the filter reseed is performed by reset().
    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        s.lambda.setZero();
    }
};

template <int N = dynamic_dimension>
using filter_trsqp_policy_fast = filter_trsqp_policy<N, sqp_mode::fast>;

template <int N = dynamic_dimension>
using filter_trsqp_policy_accurate =
    filter_trsqp_policy<N, sqp_mode::accurate>;

// In-header concept gate: surfaces nlp_solver concept-satisfaction
// failures at the first compile of any TU that includes this header
// rather than waiting for the downstream compile-test TU. Mirrors the
// pattern in tests/compile/sqp_mode_racing_test.cpp.
static_assert(nlp_solver<step_budget_solver<filter_trsqp_policy<dynamic_dimension, sqp_mode::accurate>>>);
static_assert(nlp_solver<step_budget_solver<filter_trsqp_policy<dynamic_dimension, sqp_mode::fast>>>);

}

#endif
