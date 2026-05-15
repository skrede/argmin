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
        // basic_solver convergence-framework parity.
        std::uint16_t stall_window{50};

        // Active-set multiplier re-estimation stride. A value of 0 is
        // treated as 1 (re-estimate every step) by the read-site clamp.
        std::size_t multiplier_reest_every_k{
            default_multiplier_reest_every_k};
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
        // s-block: -mu_ineq
        //          (slack-leg of the joint Lagrangian; vanishes at the
        //           KKT point by complementarity per Lalee, Nocedal,
        //           Plantenga 1998 Section 3.1, so the joint and x-only
        //           gradient norms coincide at convergence).
        {
            Eigen::Vector<double, N> grad_L_x = s.g;
            if(s.n_eq > 0)
                grad_L_x.noalias() -= s.J_eq.transpose()
                                    * s.bufs.kkt_lambda_eq_buf;
            if(n_ineq > 0)
                grad_L_x.noalias() -= s.J_ineq.transpose()
                                    * s.bufs.kkt_mu_ineq_buf;
            s.joint_grad_L_new_buf.head(n) = grad_L_x;
            if(n_ineq > 0)
                s.joint_grad_L_new_buf.tail(n_ineq) = -s.bufs.kkt_mu_ineq_buf;
        }

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

        // Section D -- Forcing tolerance eps_k.
        //
        // Reference: Eisenstat and Walker 1996 (accurate-mode);
        //            Dembo, Eisenstat, Steihaug 1982 (fast-mode).
        // The forcing sequence reads the Lagrangian-gradient norm (not
        // the bare objective-gradient norm) so the inexact-Newton tail
        // tightens with stationarity, not with raw f-progress.
        const double grad_L_norm =
            s.joint_grad_L_new_buf.template lpNorm<Eigen::Infinity>();
        double eps_k;
        if constexpr (Mode == sqp_mode::fast)
            eps_k = std::min(0.1, grad_L_norm);
        else
            eps_k = std::min(0.5, std::sqrt(grad_L_norm));

        // Section E -- Steihaug-CG inner-iteration cap. Resolves the
        // policy-level multiplier against the joint primal width.
        const std::size_t max_cg_iter =
            options.max_cg_iterations_multiplier
            * static_cast<std::size_t>(n_total);

        // Section F -- Displaced joint bounds: (lower - z_k, upper - z_k).
        // The s-block carries the slack lower bound s >= 0 and an
        // infinite upper bound; Steihaug-CG projects on the displaced
        // box at each inner iteration.
        Eigen::VectorXd lower_joint(n_total);
        Eigen::VectorXd upper_joint(n_total);
        lower_joint.head(n) = (s.lower - s.x).template cast<double>();
        upper_joint.head(n) = (s.upper - s.x).template cast<double>();
        if(n_ineq > 0)
        {
            lower_joint.tail(n_ineq) = -s.s_slack;
            upper_joint.tail(n_ineq).setConstant(inf);
        }

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

        // Section K -- Byrd-Omojokun composite step + ratio test +
        // radius update via the helper. The helper is templated on
        // Eigen::Dynamic because the joint primal dimension is a
        // runtime quantity even when the policy template N is a fixed
        // compile-time constant for the x-only dim.
        const auto bo = argmin::detail::byrd_omojokun_composite_step<
            double, Eigen::Dynamic>(
            z_k, s.joint_grad_L_new_buf,
            hessian_op,
            A_joint, c_joint,
            s.trust_radius, eps_k, max_cg_iter,
            lower_joint, upper_joint,
            f_old, c_norm_old,
            trial_eval,
            s.AAt_workspace, s.ldlt_feasibility, s.w_workspace,
            v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

        // Section L -- Radius update consumes the helper's verdict.
        s.trust_radius = bo.new_delta;

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
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            ++s.iteration;
            return r;
        }

        // Section N -- Single-step rejection branch (ratio < eta_1 but
        // radius above the floor). Iterate unchanged; the helper has
        // already shrunk s.trust_radius. The null-step status is
        // unset; basic_solver's convergence framework reads
        // is_null_step + the stall window to decide termination.
        //
        // Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1
        //            (rejection contracts the radius, leaves the
        //             iterate unchanged).
        if(!bo.accepted)
        {
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), /*bfgs_reset_count=*/0);
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            ++s.iteration;
            return r;
        }

        // Section O -- Step acceptance.
        //
        // Save the joint old iterate and the joint old Lagrangian
        // gradient for the BFGS curvature pair, then apply the step
        // and re-evaluate problem callbacks at the new x.
        //
        // The joint old iterate is (x_old; s_slack_old); the joint
        // gradient old is the snapshot in grad_L_new_buf BEFORE the
        // re-evaluation overwrites the head/tail blocks below.
        s.joint_x_old_buf.head(n) = s.x;
        if(n_ineq > 0)
            s.joint_x_old_buf.tail(n_ineq) = s.s_slack;
        s.joint_grad_L_old_buf = s.joint_grad_L_new_buf;

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

        // Section Q -- New joint Lagrangian gradient at z_{k+1} and the
        // BFGS curvature pair (s_k, y_k) on the joint (x, s) space.
        {
            Eigen::Vector<double, N> grad_L_x_new = s.g;
            if(s.n_eq > 0)
                grad_L_x_new.noalias() -= s.J_eq.transpose()
                                        * s.bufs.kkt_lambda_eq_buf;
            if(n_ineq > 0)
                grad_L_x_new.noalias() -= s.J_ineq.transpose()
                                        * s.bufs.kkt_mu_ineq_buf;
            s.joint_grad_L_new_buf.head(n) = grad_L_x_new;
            if(n_ineq > 0)
                s.joint_grad_L_new_buf.tail(n_ineq) = -s.bufs.kkt_mu_ineq_buf;
        }

        s.joint_sk_buf.head(n) = s.x;
        if(n_ineq > 0)
            s.joint_sk_buf.tail(n_ineq) = s.s_slack;
        s.joint_sk_buf.noalias() -= s.joint_x_old_buf;
        s.joint_yk_buf = s.joint_grad_L_new_buf - s.joint_grad_L_old_buf;

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

        // Section S -- KKT residual on the x-only un-reformulated
        // problem and step_result return. constraint_violation reads on
        // the un-reformulated c_eq / c_ineq for cross-policy reporting
        // parity (the line-search SQP family reports primal feasibility
        // on the same x-only quantity); gradient_norm reads on the
        // joint Lagrangian gradient infinity-norm (the slack-leg
        // vanishes at the KKT point by complementarity, so joint and
        // x-only norms coincide at convergence).
        //
        // Reference: Nocedal and Wright 2e Definition 12.1 (KKT primal
        //            feasibility);
        //            Section 12.3 + eq. 12.34 (Lagrangian stationarity).
        const double kkt = detail::kkt_residual<double,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        const double grad_L_norm_new =
            s.joint_grad_L_new_buf.template lpNorm<Eigen::Infinity>();

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
        s.iteration = 0;
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
