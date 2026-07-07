#ifndef HPP_GUARD_ARGMIN_SOLVER_TR_SQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_TR_SQP_POLICY_H

// Trust-region SQP policy implementing the Byrd-Omojokun composite step.
//
// The policy avoids the L1-merit Maratos failure mode by construction: no
// merit function, no Armijo backtracker, no penalty-parameter calibration.
// Step acceptance is the canonical actual-vs-predicted reduction ratio.
//
// Inequalities are reformulated via slacks (c_ineq(x) + s = 0, s >= 0).
// The joint primal is z = (x, s) in R^{n + n_ineq}; the Lagrangian Hessian
// approximation lives on the joint space via dense_ldl_bfgs<double, N>.
//
// argmin variant: bounds projection inside Steihaug-CG handles box bounds
//                 on x and slack bounds on s uniformly; TR ratio thresholds
//                 and radius update factors are uniform across modes per
//                 literature consensus; per-mode dispatch (CG inner-iter cap,
//                 forcing-sequence variant, BFGS-skip-on-non-positive-
//                 curvature, multiplier-reest stride) is policy-local via
//                 `if constexpr (Mode == sqp_mode::fast)`.
//
// Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim. 8(3):682-706;
//            Conn, Gould, Toint 2000 MOS-SIAM Trust-Region Methods
//            Chapters 7, 12, 17;
//            Nocedal and Wright 2e Section 4.1 (TR ratio test and radius
//            update);
//            scipy/optimize/_trustregion_constr (public-API reference
//            shape).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/byrd_omojokun.h"
#include "argmin/detail/steihaug_cg.h"
#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/bound_projection.h"

#include "argmin/result/step_result.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"

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

// argmin variant: closed-set Mode NTTP threaded through tr_sqp_policy.
//                 `rebind<M>` preserves Mode on N rebind. Per-mode
//                 Steihaug-CG inner-iter cap, forcing-sequence variant,
//                 BFGS-skip-on-non-positive-curvature gate, and
//                 multiplier-reest stride are exposed as static-constexpr
//                 members. TR ratio thresholds and radius-update factors
//                 are uniform across modes per literature consensus.
//                 `accurate` is the default to preserve cross-family
//                 consistency with the line-search SQP policies.
//
// Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Section 7.3 Algorithm 7.2 (truncated CG);
//            Eisenstat and Walker 1996 SIAM J. Sci. Comput. 17(1):16-32
//            (accurate-mode adaptive forcing);
//            Dembo, Eisenstat, Steihaug 1982 SIAM J. Numer. Anal.
//            19(2):400-408 (fast-mode forcing sequence).
template <int N = dynamic_dimension, sqp_mode Mode = sqp_mode::accurate>
struct tr_sqp_policy
{
    using scalar_type = double;
    static constexpr sqp_mode mode_ = Mode;

    template <int M>
    using rebind = tr_sqp_policy<M, Mode>;

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
    // Reference: Bertsekas 1996 §4.2 (stale-multiplier reuse rationale);
    //            Nocedal and Wright 2e Section 18.3 + Algorithm 18.3
    //            (working-set identification).
    static constexpr std::size_t default_multiplier_reest_every_k =
        (Mode == sqp_mode::fast) ? std::size_t{5} : std::size_t{1};

    // Steihaug-CG inner-iteration cap MULTIPLIER (resolved against
    // n + n_ineq at policy init time). Fast caps at the textbook
    // exact-arithmetic bound; accurate doubles it to absorb inexact-
    // Hessian (BFGS) effects. This is the primary wall-time lever for
    // fast mode — Steihaug-CG iterations dominate per-step cost in TRSQP
    // (each iteration runs one O((n+n_ineq)^2) Hessian-vector product
    // through dense_ldl_bfgs).
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

    // Uniform-across-modes trust-region parameters. Mirrored from the
    // detail::byrd_omojokun constexpr block so callers can read tunable
    // defaults from the policy's options_type while the helper continues
    // to receive the resolved value as a function argument.
    //
    // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
    //            (universal trust-region defaults; consistent across
    //            scipy trust-constr, Ceres trust-region minimizer,
    //            KNITRO).
    static constexpr double default_initial_trust_radius = 1.0;
    static constexpr double default_min_trust_radius     = 1e-12;

    // Adaptive L2-merit penalty parameter defaults. The penalty is
    // persistent state on the joint Byrd-Omojokun ratio test (the
    // augmented merit weights the feasibility leg by `penalty`); the
    // penalty_factor is the LNP heuristic growth factor that fires only
    // when the model predicts a positive violation decrease (qm > 0).
    //
    // default_penalty_factor is re-derived (2026-07-05) from a fresh
    // sweep over penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} on the
    // HS-suite closure axis (HS024/035/039/040/043/050) gated on the
    // non-regression reference axis (HS026/028/071/076), both sqp_mode
    // variants, under the qm > 0 growth gate. On this corrected gate the
    // baseline penalty_factor = 0 closes 8 of 12 closure-target tuples,
    // while every non-zero entry holds the full reference axis and
    // closes all 12; 0.3 is the lexicographic winner (full closure, no
    // reference regression, fewest closure-target iterations of the
    // full-closure entries) and coincides with the scipy update_penalty
    // literature default. The value is a re-derived winner, not a
    // carried-forward lock.
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3 (initial
    //            penalty; growth factor); scipy update_penalty
    //            (penalty_factor = 0.3 default).
    static constexpr double default_initial_penalty = 1.0;
    static constexpr double default_penalty_factor  = 0.3;

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
    // Default re-derived (2026-07-05) from the joint
    // (penalty_factor, soc_max_iterations) sweep across
    // penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} and
    // soc_max_iterations in {0, 1, 2, 4}, on:
    //   - reference cells (non-regression gate): HS026, HS028, HS071,
    //     HS076 in both sqp_mode variants;
    //   - closure-target cells: HS024, HS035, HS039, HS040, HS043,
    //     HS050 in both sqp_mode variants.
    //
    // On the corrected kernel (SOC residual c(z+p) - A p with the
    // -A p subtraction present, LNP penalty growth gated on qm > 0,
    // and the grad-L-predicted / f-measured ratio test) the corrected
    // penalty growth at default_penalty_factor closes every closure-
    // target tuple at soc_max_iterations = 0. Adding SOC retries on top
    // (soc in {1, 2, 4}) does not improve the closure count or the
    // reference axis and only spends iterations, so the SOC retry
    // budget re-locks to 0: the retry is an opt-in knob, not a default.
    // The per-mode strict bar is 1% f_err and 1e-4 cv for accurate,
    // 5% f_err and 1e-2 cv for fast.
    //
    // A workload whose curvature exhibits the Maratos effect can opt in
    // with `policy.options.soc_max_iterations = 2` (paired with the
    // default penalty growth); the user assigns the field directly.
    //
    // soc_max_iterations = 0 disables the retry entirely; the
    // helper's existing single-step rejection branch returns a
    // null-step result on the first rejection (pre-SOC behavior).
    //
    // Reference: Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
    //            8(3):682-706 Section 3.1 (v-optimal restoration shape);
    //            Nocedal and Wright 2e Section 18.3 (Maratos effect and
    //            the line-search second-order correction analog at
    //            kraft_slsqp_policy.h Section 2.2.4).
    static constexpr std::size_t default_soc_max_iterations =
        std::size_t{0};

    struct options_type
    {
        // Direct-field-default form: literature-defaulted scalars are
        // brace-initialized from the per-mode default_* static-constexpr
        // members above. Step site reads the field directly (no
        // value_or / has_value indirection). Callers who want a non-
        // default value assign the field directly.
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
        // step_budget_solver convergence-framework parity.
        std::uint16_t stall_window{50};

        // Active-set multiplier re-estimation stride. A value of 0 is
        // treated as 1 (re-estimate every step) by the read-site clamp.
        std::size_t multiplier_reest_every_k{
            default_multiplier_reest_every_k};

        // Initial value for the adaptive L2-merit penalty parameter.
        // The state's `penalty` field is seeded from this at init() /
        // reset() time and then updated in place by the LNP / scipy
        // update_penalty heuristic on every composite-step call.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3.
        double initial_penalty{default_initial_penalty};

        // LNP heuristic growth factor for the adaptive L2-merit penalty.
        // Selected empirically across {0.0, 0.01, 0.05, 0.1, 0.3} as
        // the largest non-regressing value on the existing HS-suite
        // acceptance bars; larger values may be selected once a paired
        // second-order correction retry on the inequality leg absorbs
        // the freeze-on-feasibility regression that they produce in
        // isolation.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 equation 1.13;
        //            scipy update_penalty.
        double penalty_factor{default_penalty_factor};

        // Maximum number of second-order correction retries on the
        // inequality leg of the composite step. Zero disables SOC retry
        // (matches the pre-SOC single-step rejection shape). Direct-
        // value form per the project memory rule on options-struct
        // fields with literature defaults.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.1;
        //            kraft_slsqp_policy line-search SOC analog.
        std::size_t soc_max_iterations{default_soc_max_iterations};
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

        // Cross-policy state-resident buffer struct. TRSQP uses the
        // x-only-typed bufs for the constraint-axis buffers (c_all, J_all,
        // kkt_lambda_eq_buf, kkt_mu_ineq_buf) sized at the x-only dimension
        // n; the joint-space (x, s) buffers are dynamic-dimension fields
        // below (joint_*) because the joint primal width n + n_ineq is a
        // runtime quantity even when the policy's N template parameter is
        // a fixed compile-time constant for the x-only dimension.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        // Joint-space (x, s) dynamic buffers. Sized at n + n_ineq on init;
        // the joint primal dimension is runtime even when N is fixed.
        Eigen::VectorXd joint_x_old_buf;
        Eigen::VectorXd joint_grad_L_old_buf;
        Eigen::VectorXd joint_grad_L_new_buf;
        Eigen::VectorXd joint_sk_buf;
        Eigen::VectorXd joint_yk_buf;

        // TRSQP-specific slack vector for the inequality reformulation
        // c_ineq(x) + s = 0, s >= 0. The full step() integration in a
        // later commit operates on the joint (x, s) primal.
        Eigen::VectorXd s_slack;

        // Current trust-region radius; updated by the ratio test inside
        // step(). The skeleton seeds it from options.initial_trust_radius
        // at init / reset time.
        double trust_radius{default_initial_trust_radius};

        // Adaptive L2-merit penalty parameter (in/out across composite-
        // step calls). Seeded from options.initial_penalty at init() /
        // reset() time; the byrd_omojokun helper updates it in place per
        // the LNP / scipy update_penalty heuristic on every step.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 Section 3.3.
        double penalty{default_initial_penalty};

        // LDLT workspace pair used by detail::equality_feasibility_warmstart
        // on the joint Jacobian inside the byrd_omojokun normal-step leg.
        Eigen::MatrixXd AAt_workspace;
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;
        Eigen::VectorXd w_workspace;

        double objective_value{};
        // Lagrangian-Hessian approximation on the JOINT (x, s) space;
        // sized n + n_ineq. The compile-time dimension is Eigen::Dynamic
        // because the joint primal width is a runtime quantity even when
        // the policy's N template parameter pins the x-only dimension.
        //
        // Reference: Fletcher and Powell 1974 Math. Computation 28:1067-
        //            1078 (LDL^T rank-1 update);
        //            Nocedal and Wright 2e Section 7.2 (limited-memory
        //            BFGS);
        //            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (BFGS in
        //            the SQP outer loop).
        detail::dense_ldl_bfgs<double, Eigen::Dynamic> hessian;
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
                      "tr_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "tr_sqp_policy requires constrained<Problem>");

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
        // dimension n; the constraint-axis buffers (c_all, J_all,
        // kkt_lambda_eq_buf, kkt_mu_ineq_buf) hold un-reformulated
        // x-only quantities cross-family consistent with the line-search
        // SQP family. The slack-augmented joint-space buffers are
        // separate dynamic-dimension fields sized below.
        s.bufs.resize(n, s.n_eq, s.n_ineq);

        // Section B of step() reads kkt_lambda_eq_buf / kkt_mu_ineq_buf
        // on the first call to assemble the joint Lagrangian gradient
        // before the post-step active-set re-estimation has run; without
        // an explicit zero-init the read is on uninitialized Eigen-managed
        // memory (resize() does not clear). Zero is the correct
        // information-free default at iter 0 — the joint Lagrangian
        // gradient collapses to the bare objective gradient, matching
        // the line-search SQP family's first-step KKT-leg semantics.
        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);

        // Joint (x, s) dynamic buffers sized on n + n_ineq.
        const int n_joint = n + s.n_ineq;
        s.joint_x_old_buf.setZero(n_joint);
        s.joint_grad_L_old_buf.setZero(n_joint);
        s.joint_grad_L_new_buf.setZero(n_joint);
        s.joint_sk_buf.setZero(n_joint);
        s.joint_yk_buf.setZero(n_joint);

        // LDLT workspace for the normal-step LSQ leg. Sized on the joint
        // constraint count m_joint = n_eq + n_ineq (the normal step
        // solves min ||A v + c||^2 against the slack-augmented A).
        const int m_joint = s.n_eq + s.n_ineq;
        if(m_joint > 0)
        {
            s.AAt_workspace.resize(m_joint, m_joint);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(m_joint);
            s.w_workspace.resize(m_joint);
        }

        // Constraint evaluation. The state holds c_eq / c_ineq /
        // J_eq / J_ineq in the un-reformulated (x-only) shape so the
        // public-API-facing constraint_violation diagnostic continues
        // to report on the original problem; the slack-augmented joint
        // (A, c) is assembled at the step() site in a later commit.
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

        // Box bounds: detected via concept. The slack bounds [0, +inf]
        // are implicit and applied inside Steihaug-CG by the joint-
        // dimension displaced box in the later step() integration.
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
        // feasible initial iterate. argmin's external sign convention
        // is `c_ineq(x) >= 0 feasible`; the slack-equality `c_ineq - s = 0`
        // with `s >= 0` is satisfied by `s_0 = c_ineq(x_0)` whenever
        // c_ineq(x_0) >= 0, and by `s_0 = 0` with residual `c_ineq(x_0)`
        // when c_ineq(x_0) < 0 (the magnitude of infeasibility).
        //
        // Reference: scipy/optimize/_trustregion_constr/canonical_constraint.py
        //            (slack init matched to the inequality sign
        //             convention).
        s.s_slack = s.c_ineq.cwiseMax(0.0).eval();

        // Lagrangian-Hessian on the joint (x, s) space. dense_ldl_bfgs
        // initializes to identity; the slack block picks up curvature
        // through (s_k, y_k) pairs naturally on subsequent push() calls.
        // The compile-time dimension is Eigen::Dynamic so the joint
        // primal width is a runtime quantity.
        s.hessian = detail::dense_ldl_bfgs<double, Eigen::Dynamic>(
            n + s.n_ineq);

        // Multiplier vector sized on the un-reformulated constraint
        // count (cross-policy consistency with the line-search SQP
        // family; the slack-equality multipliers are policy-internal
        // and live inside bufs).
        s.lambda = Eigen::VectorXd::Zero(m);

        s.trust_radius = options.initial_trust_radius;
        s.penalty      = options.initial_penalty;
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Trust-region SQP step: Byrd-Omojokun composite step on the
        // slack-reformulated joint variable z = (x, s).
        //
        // Reference: Nocedal and Wright 2e Section 18.5 Algorithm 18.4
        //            (Byrd-Omojokun composite step);
        //            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
        //            8(3):682-706 Section 3 (concrete dogleg and ratio-
        //            test pseudocode);
        //            Eisenstat and Walker 1996 SIAM J. Sci. Comput.
        //            17(1):16-32 (accurate-mode forcing sequence);
        //            Dembo, Eisenstat, Steihaug 1982 SIAM J. Numer.
        //            Anal. 19(2):400-408 (fast-mode forcing sequence).
        //
        // argmin variant: slack reformulation makes the joint Jacobian
        //                 block-diagonal in the slack block; bounds
        //                 projection inside Steihaug-CG handles box
        //                 bounds on x and the s >= 0 bound on slacks
        //                 uniformly via detail::project.

        const int n = static_cast<int>(s.x.size());
        const int n_ineq = s.n_ineq;
        const int n_total = n + n_ineq;
        const int m_total = s.n_eq + n_ineq;
        std::size_t bfgs_skip_count = 0;

        constexpr double inf = std::numeric_limits<double>::infinity();

        // Section B -- Joint Lagrangian gradient at the current iterate.
        //
        // x-block: g - J_eq^T lambda_eq - J_ineq^T mu_ineq
        //          (the standard x-only Lagrangian gradient against the
        //           active-set multiplier estimate carried in
        //           s.bufs.kkt_*).
        // s-block: +mu_ineq
        //          (the joint multiplier on the slack-equality rows
        //           consistent with the x-block is nu = -mu, so
        //           grad_s L = -nu = +mu; derivation at
        //           detail::assemble_joint_lagrangian_gradient. This
        //           block does NOT vanish at a KKT point -- it equals
        //           the slack-bound multiplier zeta = mu -- which is
        //           why stationarity is measured with the box-projected
        //           form in Section E, not the raw norm).
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
        // convention (c_ineq >= 0 feasible) the helper-internal residual
        // is -c_ineq + s, with the joint Jacobian's inequality row block
        // negated to match the linearization.
        //
        // Reference: scipy/optimize/_trustregion_constr/canonical_constraint.py
        //            (the same internal sign flip on c and J for the
        //             inequality leg).
        //
        // Hot-path optimization candidate: hoist A_joint / c_joint into
        // s.bufs as state-resident scratch in a later milestone.
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
        // The s-block carries the slack lower bound s >= 0 and an
        // infinite upper bound; Steihaug-CG projects on the displaced
        // box at each inner iteration, and the box-projected
        // stationarity measure in Section E reads the same displaced
        // box.
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

        // Section F -- Steihaug-CG inner-iteration cap. Resolves the
        // policy-level multiplier against the joint primal width.
        const std::size_t max_cg_iter =
            options.max_cg_iterations_multiplier
            * static_cast<std::size_t>(n_total);

        // Section G -- Hessian-vector-product closure. The joint
        // Lagrangian Hessian approximation lives in s.hessian (sized
        // n + n_ineq at init); the helper consumes it as a generic
        // callable so steihaug_cg / byrd_omojokun stay BFGS-blind.
        auto hessian_op = [&s](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            return s.hessian.multiply(v);
        };

        // Section H -- Trial-evaluation closure. Materializes z + p as
        // x_trial / s_trial; queries the problem callbacks at x_trial;
        // assembles the joint residual under the slack reformulation
        // sign convention; returns (f_new, ||c_joint_new||_2).
        auto trial_eval = [&s, n, n_ineq](const Eigen::VectorXd& p)
            -> std::pair<double, double>
        {
            Eigen::Vector<double, N> x_trial = s.x + p.head(n);
            Eigen::VectorXd s_trial(n_ineq);
            if(n_ineq > 0)
                s_trial = s.s_slack + p.tail(n_ineq);

            const double f_new = s.problem->value(x_trial);

            Eigen::VectorXd c_eq_new(s.n_eq);
            Eigen::VectorXd c_ineq_new(n_ineq);
            if(s.n_eq + n_ineq > 0)
            {
                Eigen::VectorXd c_all_new(s.n_eq + n_ineq);
                s.problem->constraints(x_trial, c_all_new);
                if(s.n_eq > 0)
                    c_eq_new = c_all_new.head(s.n_eq);
                if(n_ineq > 0)
                    c_ineq_new = c_all_new.tail(n_ineq);
            }

            Eigen::VectorXd c_joint_new(s.n_eq + n_ineq);
            if(s.n_eq > 0)
                c_joint_new.head(s.n_eq) = c_eq_new;
            if(n_ineq > 0)
                c_joint_new.tail(n_ineq) = -c_ineq_new + s_trial;
            return {f_new, c_joint_new.norm()};
        };

        // Section I -- Baseline scalars for the ratio test.
        const double c_norm_old = c_joint.norm();
        const double f_old = s.objective_value;

        // Section J -- Per-step workspaces in joint primal space.
        // Hot-path optimization candidate: hoist these into s.bufs once
        // benchmarks identify the byrd_omojokun call site as a bottleneck.
        Eigen::VectorXd z_k(n_total);
        z_k.head(n) = s.x;
        if(n_ineq > 0)
            z_k.tail(n_ineq) = s.s_slack;

        // Joint OBJECTIVE gradient: x-block grad f, slack block zero
        // (the slack variables do not enter f). Consumed by the
        // helper's predicted objective-leg reduction and the LNP
        // penalty gate -- both f-model quantities, matching the
        // f-measured actual reduction in the ratio test below. The
        // joint LAGRANGIAN gradient (Section B) keeps driving the
        // quadratic step model.
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

        // The AAt / LDLT workspaces for the normal-step LSQ leg are
        // pre-sized at init() time on the joint constraint count
        // m_joint = n_eq + n_ineq; the byrd_omojokun helper reuses them
        // directly with no resize on the hot path.

        // Section K -- Byrd-Omojokun composite step. The helper
        // returns raw decision inputs (predicted objective-leg
        // reduction, vpred feasibility-leg predicted reduction,
        // f_new and c_norm_new at the trial point); the augmented
        // L2-merit ratio test and the eta_1 / eta_2 radius update
        // are computed locally below against these inputs. The
        // helper is templated on Eigen::Dynamic because the joint
        // primal dimension is a runtime quantity even when the policy
        // template N is a fixed compile-time constant for the x-only
        // dim.
        //
        // Snapshot the pre-step trust radius and penalty so the SOC
        // retry (Section K') can re-invoke the helper without
        // compounding the radius shrink the primary call's rejection
        // performs (Lalee, Nocedal, Plantenga 1998 Section 3.1: the SOC
        // retry exploits curvature information from the trial-point
        // residual, not radius expansion) and without inheriting any
        // penalty inflation the LNP heuristic applied on the rejected
        // primary step.
        //
        // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
        //            (rho branches on eta_1 / eta_2; boundary-guarded
        //             expansion; the ratio test implicitly assumes
        //             pred > 0 -- a non-positive augmented predicted
        //             reduction takes the explicit reject branch
        //             below instead of forming rho).
        const double delta_pre_step   = s.trust_radius;
        const double penalty_pre_step = s.penalty;
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
            s.penalty, options.penalty_factor);

        // Caller-side augmented-merit ratio test + radius update.
        // The helper updates s.penalty in place per the LNP heuristic
        // BEFORE returning, so the post-call s.penalty value is the
        // one that must weight both legs of the augmented merit on
        // the primary step (matches the pre-refactor BO behavior).
        // The radius input below is `s.trust_radius` -- the pre-step
        // radius snapshot lives in `delta_pre_step` and is consumed
        // by the SOC retry, not by the primary branch.
        auto compute_acceptance = [](const argmin::detail::
                                         byrd_omojokun_step_result& bo,
                                     double f_old_,
                                     double c_norm_old_,
                                     double penalty_,
                                     double delta_in,
                                     double step_norm)
            -> std::pair<bool, double>
        {
            const double actual =
                (f_old_ + penalty_ * c_norm_old_)
                - (bo.f_new + penalty_ * bo.c_norm_new);
            const double pred_aug =
                bo.predicted + penalty_ * bo.vpred;
            // Explicit reject branch on a non-positive augmented
            // predicted reduction: N&W Algorithm 4.1's ratio test
            // assumes pred > 0. Forming rho = actual / max(pred, eps)
            // here would produce |rho| ~ 1e16 on a legitimately
            // negative prediction and accept any actual reduction
            // >= 0.1 * eps -- degenerating sufficient decrease to
            // simple decrease exactly in the degenerate cases the
            // guard was meant to protect. The negated comparison also
            // routes a NaN prediction (NaN-producing trial_eval) to
            // the reject branch.
            if(!(pred_aug > 0.0))
                return {false,
                        argmin::detail::tr_shrink_factor * delta_in};
            const double rho = actual / pred_aug;
            if(rho < argmin::detail::tr_eta_1)
                return {false,
                        argmin::detail::tr_shrink_factor * delta_in};
            if(rho > argmin::detail::tr_eta_2
               && step_norm
                  >= argmin::detail::tr_boundary_guard * delta_in)
                return {true,
                        std::min(
                            argmin::detail::tr_expand_factor * delta_in,
                            argmin::detail::tr_delta_max)};
            return {true, delta_in};
        };

        auto [accepted, new_delta] = compute_acceptance(
            bo_step,
            f_old,
            c_norm_old,
            s.penalty,
            s.trust_radius,
            p_out.norm());

        // Section K' -- Second-order correction retry. On a rejected
        // primary composite step in the MARATOS REGIME (trial
        // violation did not decrease), re-assemble the SOC residual
        //   c_soc = c(z_k + p) - A_joint p
        // at the trial point with the ORIGINAL Jacobian A_joint via
        // the shared detail::assemble_joint_soc helper: subtracting
        // A_joint p preserves the linear constraint model and leaves
        // only the curvature remainder for the corrective re-solve
        // (N&W Section 18.3; Wachter-Biegler 2006 eq. 24/27 in the
        // full-step convention). The helper is re-called with
        // c_joint_soc; the trust radius is held fixed at
        // delta_pre_step (no further shrink) and the penalty is reset
        // to its pre-step value so a single failed primary step does
        // not inflate the penalty input to the retry.
        //
        // Trigger: only when the primary rejection left the trial
        // violation at or above the current one (c_norm_new >=
        // c_norm_old, the theta(z + p) >= theta(z_k) Maratos gate of
        // Wachter-Biegler A-5.7). An ordinary rejection whose
        // violation already fell is driven by the objective leg; a
        // corrected re-solve cannot help there and the retry would
        // only burn constraint evaluations. A NaN trial violation
        // fails the >= comparison and skips the retry.
        //
        // Continuation: between retries the trial violation must
        // contract by detail::kappa_soc (Wachter-Biegler A-5.9) or
        // the loop aborts; a NaN retry violation fails the negated
        // contraction test and aborts.
        //
        // Acceptance: if any retry produces an accepted step, p_out is
        // overwritten with the retry's step and the policy falls
        // through to Section O (step acceptance). The helper's
        // new_delta from the accepted retry governs the post-step
        // radius update.
        //
        // Rejection: if the retries are exhausted or the continuation
        // aborts, the policy falls through to Sections L/M/N with the
        // LAST retry's verdict; the trust radius from that retry
        // reflects exactly one tr_shrink_factor application (not
        // chained shrinks), because each retry is invoked with
        // delta_pre_step as input.
        //
        // Reference: Nocedal and Wright 2e Section 18.3 (Maratos
        //            effect and SOC pairing); Wachter and Biegler
        //            2006 Math. Programming 106:25-57 Section 2.4,
        //            Algorithm A steps A-5.7..A-5.9.
        std::size_t soc_retry_count = 0;
        if(!accepted && options.soc_max_iterations > 0
           && bo_step.c_norm_new >= c_norm_old)
        {
            Eigen::Vector<double, N> x_trial(n);
            Eigen::VectorXd s_trial(n_ineq);
            Eigen::VectorXd c_all_trial(m_total);
            Eigen::VectorXd c_trial_joint(m_total);
            Eigen::VectorXd c_joint_soc(m_total);

            // Violation of the most recent trial, driving the
            // kappa_soc contraction test across retries.
            double theta_trial_prev = bo_step.c_norm_new;

            for(std::size_t soc_iter = 0;
                soc_iter < options.soc_max_iterations;
                ++soc_iter)
            {
                ++soc_retry_count;

                // Trial point at the rejected step (state x and s_slack
                // are still the pre-step values; only p_out has been
                // populated by the helper).
                x_trial.noalias() = s.x + p_out.head(n);
                if(n_ineq > 0)
                    s_trial.noalias() = s.s_slack + p_out.tail(n_ineq);

                // Re-evaluate constraints at the trial point. The
                // equality block uses c_eq(x_trial); the inequality
                // block uses the slack-reformulated joint residual
                // -c_ineq(x_trial) + s_trial under argmin's sign
                // convention (c_ineq >= 0 feasible).
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

                // Reset penalty + trust-radius inputs to pre-step
                // values: the retry re-enters the helper with the same
                // starting conditions as the primary step, but with a
                // corrected residual.
                s.penalty = penalty_pre_step;
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
                    s.penalty, options.penalty_factor);

                // Re-run the caller-side acceptance gate against this
                // retry's raw outputs. The radius input is held at
                // delta_pre_step (no chained shrinks across retries).
                std::tie(accepted, new_delta) = compute_acceptance(
                    bo_step,
                    f_old,
                    c_norm_old,
                    s.penalty,
                    delta_pre_step,
                    p_out.norm());

                if(accepted)
                    break;

                // Continuation gate (Wachter-Biegler A-5.9): abort
                // unless the retry's trial violation contracted by
                // kappa_soc relative to the previous trial. The
                // negated form routes a NaN violation to the abort
                // branch.
                if(!(bo_step.c_norm_new
                     < argmin::detail::kappa_soc * theta_trial_prev))
                    break;
                theta_trial_prev = bo_step.c_norm_new;
            }
        }

        // Section L -- Radius update consumes the caller-side verdict.
        // After the SOC retry path, `new_delta` is either the accepted
        // retry's radius (delta_pre_step or expanded) or the last
        // rejected retry's radius (shrunk from delta_pre_step). Either
        // way, new_delta is the authoritative post-step radius.
        s.trust_radius = new_delta;

        // Section M -- Trust-radius collapse path. Below the policy-
        // level floor the policy emits a null-step carrying the new
        // solver_status entry; the convergence framework's stall window
        // then declares termination after stall_window consecutive
        // null-steps. Note: the Δ-collapse path triggers ONLY when the
        // post-update radius is below floor; this is the
        // "shrink-shrink-shrink ... rejected" terminal pattern, not the
        // single-rejection branch handled in Section N.
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
            ++s.iteration;
            return r;
        }

        // Section N -- Single-step rejection branch (ratio < eta_1 but
        // radius above the floor). Iterate unchanged; the helper has
        // already shrunk s.trust_radius. The null-step status is
        // unset; step_budget_solver's convergence framework reads
        // is_null_step + the stall window to decide termination.
        //
        // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
        //            (rejection contracts the radius, leaves the
        //             iterate unchanged).
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
        // Save the joint old iterate plus the OLD-point objective
        // gradient and Jacobians for the fixed-multiplier BFGS pair
        // (N&W 18.13 evaluates BOTH gradient legs against the NEW
        // multipliers, so the old-point quantities must survive the
        // re-evaluation below; mirrors the line-search family's
        // J_all_old retention consumed by compute_bfgs_pair_fused),
        // then apply the step and re-evaluate problem callbacks at
        // the new x.
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
        // x_{k+1}. The stride dispatch reads the policy-level constexpr
        // resolved at template instantiation. On the skip path the
        // prior step's kkt_lambda_eq_buf / kkt_mu_ineq_buf are reused
        // (state-resident; matches the line-search SQP family pattern).
        //
        // Reference: Bertsekas 1996 Section 4.2 (stale-multiplier
        //            reuse rationale);
        //            Nocedal and Wright 2e Section 18.3 + Algorithm 18.3
        //            (working-set identification);
        //            eq. 18.15 (least-squares lambda);
        //            Definition 12.1 (KKT conditions).
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
        //            Nocedal and Wright 2e Procedure 18.2 (damping
        //            guard);
        //            eq. 18.22-18.24 (Powell damping, accurate path).
        //
        // Accurate mode silently bypasses the push when s^T y <= 0;
        // Powell damping inside dense_ldl_bfgs::push handles the
        // h1_raw < 0.2 * h2 case as long as h2 > 0, but the explicit
        // s^T y guard above keeps the policy-step semantics legible.
        // Only fast mode increments bfgs_skip_count -- the diagnostic
        // surface is fast-mode-only by convention, matching the in-tree
        // analog at nw_sqp_policy::step().
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
        // un-reformulated problem and step_result return.
        // constraint_violation reads on the un-reformulated c_eq /
        // c_ineq for cross-policy reporting parity (the line-search
        // SQP family reports primal feasibility on the same x-only
        // quantity); gradient_norm reports the box-projected joint
        // stationarity measure ||P(z - grad_L, l, u) - z||_inf (the
        // raw joint norm is floored at mu by the +mu slack block at
        // inequality-active solutions and is not a stationarity
        // measure there; at interior points the two coincide).
        //
        // Reference: Nocedal and Wright 2e Definition 12.1 (KKT primal
        //            feasibility);
        //            Section 12.3 + eq. 12.34 (Lagrangian stationarity);
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

        // BFGS reset / Armijo-NaN counters stay at zero: TRSQP has no
        // BFGS-reset retry and no Armijo NaN gate (the ratio test does
        // not require trial-iterate finiteness recovery; a NaN-producing
        // problem callback is undefined behavior for this policy).
        return step_result<double>{
            .objective_value      = s.objective_value,
            .gradient_norm        = grad_L_norm_new,
            .step_size            = p_out.norm(),
            .objective_change     = s.objective_value - f_old,
            // Acceptance branch: rho >= eta_1 implies the augmented
            // L2 merit (f + ||c||_2) decreased per the helper's ratio
            // test, so `improved` is true by construction here.
            .improved             = true,
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

    // Hot start -- preserves BFGS Hessian.
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
        s.penalty      = options.initial_penalty;
        s.iteration = 0;

        // Zero the active-set multiplier buffers: they were estimated
        // at the pre-reset iterate's linearization, and the first
        // post-reset joint Lagrangian gradient reads them directly --
        // without the zeroing it would consume multipliers from an
        // unrelated point (under fast mode's re-estimation stride the
        // stale values could persist for several iterations).
        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
    }

    // Cold restart -- clears BFGS Hessian.
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
using tr_sqp_policy_fast = tr_sqp_policy<N, sqp_mode::fast>;

template <int N = dynamic_dimension>
using tr_sqp_policy_accurate = tr_sqp_policy<N, sqp_mode::accurate>;

}

#endif
