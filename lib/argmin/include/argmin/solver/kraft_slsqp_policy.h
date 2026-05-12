#ifndef HPP_GUARD_ARGMIN_SOLVER_KRAFT_SLSQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_KRAFT_SLSQP_POLICY_H

// Kraft 1988 SLSQP policy for basic_solver.
//
// Implements Sequential Least Squares Programming (SLSQP) faithful to
// Kraft (1988) DFVLR-FB 88-28: dense damped BFGS Hessian of the
// Lagrangian, Kraft's LSQ/LSEI cascade for the QP subproblem with
// native box-bound handling, L1 merit function for the line search,
// and second-order correction on infeasibility.
//
// Each step: (1) pull the BFGS Hessian B directly (no rebuild),
// (2) solve the QP via the kraft_lsq_qp_solver cascade, (3) update
// the L1 merit penalty sigma from the QP multipliers, (4) backtrack
// on the merit function, (5) damped BFGS update of B using the
// Lagrangian gradient difference.
//
// Differences from the earlier L-BFGS variant:
//   - detail::compact_lbfgs is replaced with detail::dense_ldl_bfgs
//     (packed L D L^T BFGS approximation with Powell-damped rank-1
//     updates per Fletcher & Powell 1974; Shanno 1978 initial-Hessian
//     rescaling theta = y^T y / s^T y is applied only on the first
//     push after construction or reset, after which the LDL absorbs
//     curvature evolution incrementally at O(n^2) per push). The
//     packed factor is consumed directly by the QP via factor_to_E_and_f
//     (skipping LLT(B) at the QP call site).
//   - detail::active_set_qp_solver is replaced with
//     detail::kraft_lsq_qp_solver, eliminating the Phase-1 feasibility
//     projection and the box-bound stalling that required feasibility
//     restoration earlier.
//
// Supports: unconstrained, equality, inequality, box, and mixed
// constraints. When the problem is only differentiable, constraints
// default to empty and the solver reduces to BFGS with a QP step
// computation.
//
// Reference: Kraft, D. (1988) "A Software Package for Sequential
//            Quadratic Programming", DFVLR-FB 88-28, Deutsche
//            Forschungs- und Versuchsanstalt fuer Luft- und Raumfahrt
//            (dense BFGS, LSQ/LSEI cascade, L1 merit, SOC).
//            N&W Chapter 18 (SQP methods), Procedure 18.2 (damped
//            BFGS), Section 18.3 (L1 merit function).
//            K&W Section 10.4 (penalty methods, augmented Lagrangian).

#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/merit_function.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/line_search/armijo.h"
#include "argmin/options/qp_options.h"
#include "argmin/line_search/options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace argmin
{

// argmin variant: closed-set Mode NTTP threaded through the kraft_slsqp
//                 policy class template. `rebind<M>` preserves Mode on
//                 N rebind. Per-mode tolerance + SOC threshold defaults
//                 are exposed as static-constexpr members; the SOC-gate
//                 site uses the per-mode default at the threshold check
//                 (constexpr-folded). `accurate` is the default to preserve
//                 baseline behavior for any consumer that has not opted in.
//
// Reference: KNITRO mode-system precedent (commercial NLP fast/accurate
//            modes); Kraft 1988 §2.2.4 (SOC threshold semantics);
//            N&W 2e Definition 12.1 (Lagrangian-gradient KKT).
template <int N = dynamic_dimension, sqp_mode Mode = sqp_mode::accurate>
struct kraft_slsqp_policy
{
    using scalar_type = double;
    static constexpr sqp_mode mode_ = Mode;

    template <int M>
    using rebind = kraft_slsqp_policy<M, Mode>;

    static constexpr double default_soc_violation_threshold =
        (Mode == sqp_mode::fast) ? 1e-2 : 1e-3;
    static constexpr double default_gradient_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-8;
    static constexpr double default_step_tolerance_rel =
        (Mode == sqp_mode::fast) ? 1e-6 : 1e-12;
    static constexpr double default_feasibility_tolerance =
        (Mode == sqp_mode::fast) ? 1e-4 : 1e-6;
    // Kraft 1988 §2.2.6 (initial penalty default; mode-invariant 1.0).
    static constexpr double default_initial_penalty = 1.0;

    // Per-mode Armijo backtracking budget. NLopt slsqp.c's slsqpb_ outer
    // loop (slsqp.c:1803-2189) interleaves an inner line search whose
    // budget tracks the outer iter limit by convention; the standalone
    // Armijo budget here is 10 for fast (wall-time minimization with
    // early null-step fallback) and 40 for accurate (NLopt-parity
    // sustained backtracking before declaring exhaustion).
    //
    // Reference: NLopt slsqp.c:1803-2189 (slsqpb_ outer line-search budget).
    static constexpr std::uint16_t default_line_search_max_iterations =
        (Mode == sqp_mode::fast) ? std::uint16_t{10} : std::uint16_t{40};

    // Armijo sufficient-decrease parameter. The 1e-4 default is the
    // Wolfe-condition convention used by NLopt slsqp.c and is held
    // constant across both modes (no literature precedent for a
    // per-mode value of c1).
    //
    // Reference: N&W 2e §3.1 (Wolfe-condition convention);
    //            NLopt slsqp.c default c1 = 1e-4.
    static constexpr double default_armijo_c1 = 1e-4;

    // Armijo backtracking shrink factor. NLopt slsqp.c uses 0.5 as the
    // default. Fast mode uses 0.3 (faster geometric back-off so each
    // Armijo iteration probes a more aggressively shorter step inside
    // the smaller fast-mode budget); accurate mode keeps NLopt's 0.5.
    //
    // Reference: NLopt slsqp.c default rho = 0.5;
    //            N&W 2e §3.5 (backtracking shrink factor range 0.1..0.9).
    static constexpr double default_armijo_rho =
        (Mode == sqp_mode::fast) ? 0.3 : 0.5;

    // BFGS Hessian-reset retry cap on line-search exhaustion. Fast mode
    // disables the retry (0 — fall straight through to null-step or
    // QP recovery, prioritizing wall-time); accurate mode matches NLopt
    // slsqp.c's ireset semantics with up to 5 retries.
    //
    // Reference: NLopt slsqp.c:1890-1895 (ireset retry pattern).
    static constexpr std::size_t default_bfgs_reset_max =
        (Mode == sqp_mode::fast) ? std::size_t{0} : std::size_t{5};

    // QP inner solver iteration cap. Fast mode caps at 50 for a quicker
    // bail-out on degenerate / cycling working sets; accurate mode matches
    // NLopt slsqp.c's LSI/LSEI cascade convention of 200.
    //
    // NOTE: Kraft lineage uses kraft_lsq_qp_recovery_solver which
    // currently does not thread these values into its solve API; brace
    // initialization lives for API uniformity with N&W lineage and to
    // make per-mode propagation observable to callers.
    //
    // Reference: NLopt slsqp.c:700-1100 (LSI/LSEI cap convention).
    static constexpr std::uint16_t default_qp_max_iterations =
        (Mode == sqp_mode::fast) ? std::uint16_t{50} : std::uint16_t{200};

    // QP convergence tolerance (acc^2 in Kraft's QP cast). Fast mode
    // relaxes to 1e-8 (sub-iter savings on a problem whose outer-loop
    // KKT residual will dominate); accurate mode matches NLopt's
    // double-precision baseline at 1e-12.
    //
    // NOTE: Same dead-wire as default_qp_max_iterations above; field
    // is observable but currently has no behavioral effect on the kraft
    // QP solve path.
    //
    // Reference: Kraft 1988 §3.5 (QP convergence acc^2 threshold).
    static constexpr double default_qp_tolerance =
        (Mode == sqp_mode::fast) ? 1e-8 : 1e-12;

    // Active-set Lagrange-multiplier re-estimation stride. Sibling
    // line-search SQP policies (nw_sqp, filter_slsqp, filter_nw_sqp)
    // recompute multipliers via compute_kkt_multipliers_active_set on
    // every step k for which (s.iteration % k == 0); on the skip path
    // the prior step's s.bufs.kkt_lambda_eq_buf / s.bufs.kkt_mu_ineq_buf
    // are reused (state-resident, zero-alloc).
    //
    // NOTE: Kraft lineage uses QP-native multipliers (qp_res.lambda
    // copied directly into s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf
    // on every step) rather than active-set least-squares, so this
    // field is wired for API uniformity across the four line-search
    // SQP policies but has no behavioral effect on Kraft's hot path.
    //
    // Reference: Bertsekas 1996 §4.2 (stale-multiplier reuse rationale);
    //            N&W 2e §18.3 + Algorithm 18.3 (working-set
    //            identification, eq. 18.15).
    //
    // Defaults picked empirically from an internal sweep on
    // HS026 / HS028 / HS043 / HS071 across k ∈ {1, 2, 3, 5, 8, 10}.
    // Accurate-mode locked at 1 (KKT-leg precision); fast-mode picked
    // as the largest k that satisfies a zero-status-regression + iter
    // inflation ≤ 10% gate on the three N&W-lineage sibling policies
    // (kraft is excluded from the gate — see no-op note above).
    static constexpr std::size_t default_multiplier_reest_every_k =
        (Mode == sqp_mode::fast) ? std::size_t{5} : std::size_t{1};

    struct options_type
    {
        // Direct-field-default form: the per-mode default_* static-constexpr
        // members above brace-initialize each field. Solver code reads the
        // field directly at every use site (no value_or / has_value
        // indirection). Callers who want a non-default value assign the field
        // directly.
        //
        // Reference: Kraft 1988 §2.2.6 (initial penalty default 1.0);
        //            Kraft 1988 §2.2.4 / N&W §18.3 (SOC threshold; per-mode
        //            1e-2 fast / 1e-3 accurate).
        double initial_penalty{default_initial_penalty};
        // Minimum constraint violation at x_k below which the second-order
        // correction (Maratos retry) is skipped on Armijo failure. Below this
        // threshold the linearization is consistent enough that the SoC step
        // adds work without materially changing the search direction.
        double soc_violation_threshold{default_soc_violation_threshold};

        // Embedded line-search params. Designated-initializer order follows
        // line_search_options field-declaration order (c1, c2, rho,
        // max_alpha, max_iterations); skipped designators (c2, max_alpha)
        // keep their line_search_options-side defaults (0.9 and 1.0).
        line_search_options line_search{
            .c1 = default_armijo_c1,
            .rho = default_armijo_rho,
            .max_iterations = default_line_search_max_iterations,
        };

        // QP subproblem params. Dead-wired on the kraft lineage at
        // present (see default_qp_max_iterations / default_qp_tolerance
        // comments above) but brace-initialized here for API uniformity
        // with the N&W lineage and to make per-mode propagation visible
        // to callers reading options.qp.
        qp_options qp{
            .max_iterations = default_qp_max_iterations,
            .tolerance = default_qp_tolerance,
        };
        std::uint16_t stall_window{50};

        // BFGS-reset retry cap on line-search exhaustion. On Armijo
        // failure (after SOC retry), the policy resets the BFGS Hessian
        // to identity and re-solves the QP, repeating up to
        // bfgs_reset_max times before returning a null-step. Per-mode
        // brace-init from default_bfgs_reset_max (0 fast / 5 accurate).
        //
        // Reference: NLopt slsqp.c slsqpb_ outer loop (line-search exhaustion
        //            fallback);
        //            NLopt slsqp.c:1890-1895 (ireset retry pattern);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: per-policy options field; cascade-free
        //                 (no new solver_status enum entry).
        std::size_t bfgs_reset_max{default_bfgs_reset_max};

        // Active-set multiplier re-estimation stride (cf. the per-policy
        // default_multiplier_reest_every_k constexpr above). Wired for
        // API uniformity with the N&W lineage; behavioral no-op on
        // Kraft's QP-native KKT-leg.
        std::size_t multiplier_reest_every_k{default_multiplier_reest_every_k};
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
        double objective_value{};
        double sigma{1.0};
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::kraft_lsq_qp_recovery_solver<double, N> qp_solver;

        // Cross-policy state-resident buffer struct. Consolidates the
        // per-step / per-line-search trial buffers, the BFGS curvature-
        // pair buffers, the constraint-axis workspaces (c_all, c_trial,
        // b_eq / b_ineq, b_eq_soc / b_ineq_soc, lam / lam_eq / lam_ineq,
        // kkt_lambda_eq / kkt_mu_ineq), the constraint Jacobian pair
        // (J_all and the J_all_old cached copy that eliminates the second
        // constraint_jacobian(x_old, ...) call on the BFGS curvature-pair
        // path), and the pre-factored Hessian buffers (E_buf / f_buf)
        // consumed by kraft_lsq_qp_recovery_solver::solve_with_factored_hessian.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers
        //               (in-tree precedent: generalizes the kraft *_buf
        //                state-resident layout).
        // Reference: N&W 2e Section 18.
        //
        // argmin variant: cross-policy state-resident buffer struct.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        // Step direction combined with the second-order correction (Maratos
        // retry). Not part of sqp_state_buffers because only kraft and
        // filter_slsqp lineages use the SOC retry; nw_sqp / filter_nw_sqp
        // do not.
        Eigen::Vector<double, N> p_combined_buf;

        // Cumulative count of phi(alpha) calls made by the Armijo
        // backtracker across every step() invocation since init/reset.
        // Counts both the main merit-function line search and the
        // second-order-correction line search if SOC fires. Incremented
        // inside phi_ls and phi_soc lambdas. Exposed so benchmarks can
        // divide by iteration count to get "average backtracks per
        // step" without needing policy-internal instrumentation.
        // Rationale: cartan's UR3e perf profile showed ~2.6x more
        // absolute FK work per pose in argmin_slsqp than in
        // nlopt_slsqp; the merit-function line search is the suspected
        // source and this counter lets us measure the hypothesis
        // directly.
        std::uint64_t line_search_calls{0};

        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int n{0};
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
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n = n;
        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // SOC step buffer (kraft-lineage; not part of sqp_state_buffers).
        s.p_combined_buf.resize(n);

        // Bounds
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

        // Constraints
        if constexpr(constrained<Problem>)
        {
            s.n_eq = problem.num_equality();
            s.n_ineq = problem.num_inequality();
        }
        else
        {
            s.n_eq = 0;
            s.n_ineq = 0;
        }

        // Cross-policy state-resident buffer struct: sizes all per-step
        // workspaces (n-sized iterate / direction / curvature-pair buffers,
        // m-sized constraint workspaces, and the pre-factored Hessian E / f
        // buffers consumed by kraft_lsq_qp_recovery_solver) in one call.
        s.bufs.resize(n, s.n_eq, s.n_ineq);

        if constexpr(constrained<Problem>)
        {
            const int m = s.n_eq + s.n_ineq;
            if(m > 0)
            {
                problem.constraints(x0, s.bufs.c_all);
                s.c_eq = s.bufs.c_all.head(s.n_eq);
                s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

                problem.constraint_jacobian(x0, s.bufs.J_all);
                s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

                s.lambda = Eigen::VectorXd::Zero(m);
            }
        }
        else
        {
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
            s.J_eq.resize(0, n);
            s.J_ineq.resize(0, n);
            s.lambda.resize(0);
        }

        // Packed LDL^T BFGS Hessian (Fletcher-Powell 1974). Initialized
        // at L = I, D = I (B_0 = I) and updated by Powell-damped rank-1
        // LDL updates per push (NLopt slsqp.c slsqpb_); replaces the
        // adaptive-theta compact L-BFGS path that rebuilt B from scratch
        // every push at O(M * n^2) cost.
        s.hessian = detail::dense_ldl_bfgs<double, N>(n);
        s.sigma = options.initial_penalty;
        s.iteration = 0;

        // Pre-allocate kraft_lsq_qp_solver workspace. Size finite-bound
        // counts to n so the allocator covers any runtime pattern of
        // finite/infinite box bounds without re-allocating.
        s.qp_solver.resize(n, s.n_eq, s.n_ineq, n, n);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Invariant at top of step: s.g, s.c_eq/c_ineq, s.J_eq/J_ineq are
        // already fresh at s.x. Maintained by init() / reset() / the prior
        // step()'s post-update block at the bottom of this function, which
        // writes gradient and constraint data for the accepted new iterate
        // before returning. basic_solver::step_n does not mutate state.x
        // between policy.step() calls, so no top-of-step re-evaluation is
        // needed.
        const int n = s.n;

        // BFGS-reset retry loop on line-search exhaustion (NLopt
        // slsqp.c:1890-1895 ireset parity). Wraps factor -> QP solve ->
        // cold-start sigma -> merit
        // -> Armijo -> SOC -> augmented-path null-step. On line-search
        // exhaustion (Armijo + SOC fail on either direct or augmented
        // path), reset the BFGS Hessian to identity and re-solve the QP,
        // bounded by options.bfgs_reset_max. The augmented-path reset
        // (Kraft 1988 §3.4 inconsistent-linearisation recovery) becomes
        // a `continue` rather than a terminal `return` while the cap
        // permits further retries; on cap exhaustion both paths return a
        // null-step with diagnostics.bfgs_reset_count set.
        //
        // Adopted from: NLopt slsqp.c:1890-1895 (ireset retry pattern,
        //               max=5).
        // Reference: NLopt slsqp.c slsqpb_ outer loop (line-search exhaustion
        //            fallback);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: cap is options.bfgs_reset_max (default 5);
        //                 status on exhaustion is is_null_step + the
        //                 diagnostics.bfgs_reset_count counter; rationale:
        //                 cascade-free design (no new solver_status
        //                 enumerant required).
        std::size_t reset_count = 0;
        const std::size_t reset_max = options.bfgs_reset_max;
        // Armijo NaN/Inf recovery counter aggregated across the main line
        // search and any second-order-correction retry within this step()
        // call. Kraft calls armijo() directly (line_search/armijo.h) so the
        // gate lives in the helper; this local mirrors the diagnostic into
        // the step_result return without exposing the helper's scope at
        // the return site (ls_soc is declared inside the SOC if-block and
        // is not visible at the bottom of step()).
        std::size_t nan_eval_count = 0;

        // Survives the retry loop into the post-loop accept-step block.
        detail::qp_result<double, N> qp_res;
        double alpha = 0.0;
        Eigen::Vector<double, N>& p = s.bufs.p_buf;

        for(;;)
        {
            // Pre-factor the BFGS Hessian into E = sqrt(D) * L^T and f =
            // -D^{-1/2} * L^{-1} * g directly off the packed L, D factors
            // maintained by detail::dense_ldl_bfgs. The recovery solver's
            // direct path consumes (E, f) at no further factorization cost,
            // skipping the O(n^3 / 3) Eigen::LLT(B) the previous code ran
            // inside kraft_lsq_qp_solver::solve on every step.
            // Reference: NLopt slsqp.c lsq_ (lines 1234-1340) for the
            // E = sqrt(D) * L^T reconstruction; Kraft 1988 DFVLR-FB 88-28
            // Section 2.2.3 (BFGS update), Section 3.2 (LSQ cast).
            s.hessian.factor_to_E_and_f(s.bufs.E_buf, s.g, s.bufs.f_buf);

            s.bufs.b_eq_workspace = -s.c_eq;
            s.bufs.b_ineq_workspace = -s.c_ineq;

            // Box bounds on the step p: p_lo <= p <= p_hi.
            // The Kraft LSQ cascade handles infinite bounds natively by
            // skipping the augmented +I / -I rows, so we do not need to
            // clip to a large finite surrogate here.
            s.bufs.p_lo_buf.noalias() = s.lower - s.x;
            s.bufs.p_hi_buf.noalias() = s.upper - s.x;

            qp_res = s.qp_solver.solve_with_factored_hessian(
                s.bufs.E_buf, s.bufs.f_buf, s.g,
                s.J_eq, s.bufs.b_eq_workspace,
                s.J_ineq, s.bufs.b_ineq_workspace,
                s.bufs.p_lo_buf, s.bufs.p_hi_buf);

            s.bufs.p_buf = qp_res.x;

            // Cold-start mu calibration. After the first QP solve, sigma must
            // dominate the multiplier scale to make the SQP direction a descent
            // direction for the L1 merit (N&W eq. 18.36). Default sigma = 1.0
            // underestimates the required penalty on HS071-class problems with
            // large multipliers, causing iter-0 line-search rejection.
            //
            // Adopted from: NLopt slsqp.c iter-0 implicit cold-start from QP lambda.
            // Reference: N&W 2e eq. 18.36; Kraft 1988 §2.2.6.
            //            iter-0 cold-start: qp-lambda-driven sigma seed
            //            bounds the first-step rejection rate by ensuring
            //            sigma dominates the multiplier scale at the
            //            start of the run.
            //
            // argmin variant: gated on s.iteration == 0; the existing post-QP
            //                 sigma update at lines below is max-monotone and
            //                 therefore idempotent against this cold-start.
            if(s.iteration == 0 && qp_res.lambda.size() > 0)
            {
                s.sigma = detail::calibrate_initial_penalty(s.sigma, qp_res.lambda);
            }

            double p_norm = p.norm();
            if(p_norm < 1e-15)
            {
                // Zero QP direction: the LSQ/LSEI cascade returned p = 0,
                // which happens either at a true KKT point (correct solver
                // behavior at a stationary point) or under active-set
                // degeneracy. is_null_step = true exempts
                // step_tolerance_criterion from firing stalled on iter 2;
                // kkt_residual via the Full E-measure lets
                // objective_tolerance_criterion declare convergence when
                // the iterate is a true KKT point; constraint_violation
                // reports L-infinity primal feasibility for dimensional
                // consistency with kkt_residual. Mirrors nw_sqp_policy's
                // null-step return so the two SQP policies behave
                // consistently at near-optimum initial points.
                //
                // Reference: N&W 2e Section 18.3 (SQP null-step semantics);
                //            Definition 12.1 + eq. 12.34 (full KKT
                //            first-order optimality E-measure).
                // Adopted from: argmin/detail/sqp_common.h extract_qp_multipliers.
                s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
                s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
                if constexpr(constrained<P>)
                {
                    if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                    {
                        argmin::detail::extract_qp_multipliers<double>(
                            qp_res.lambda, s.n_eq, s.n_ineq,
                            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
                    }
                }

                // Adopted from: argmin/detail/sqp_common.h null_step_result.
                auto r = argmin::detail::null_step_result<double, N,
                                                          Eigen::Dynamic,
                                                          Eigen::Dynamic>(
                    s.objective_value, s.g, s.J_eq, s.J_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                    s.c_eq, s.c_ineq, s.x.norm(), reset_count);
                // kraft reports gradient_norm via the previous-step
                // s.lambda for cross-policy consistency with the accepted-
                // step path; override the helper's internal computation
                // from the per-leg buffers.
                r.gradient_norm = lagrangian_gradient_norm(s);
                return r;
            }

            // Update penalty parameter sigma from the QP multipliers.
            // Ensures sigma >= |lambda|_inf + delta so the QP step is a
            // descent direction for the L1 merit under the linearized
            // KKT conditions. Monotone (never decreases).
            //
            // Reference: N&W eq. 18.36 (sufficient penalty for descent).
            const double constraint_viol_0 = detail::constraint_violation(
                s.c_eq, s.c_ineq);

            if(qp_res.lambda.size() > 0)
            {
                const double lambda_max = qp_res.lambda.cwiseAbs().maxCoeff();
                if(lambda_max + 0.5 > s.sigma)
                    s.sigma = lambda_max + 1.0;
            }

            // L1 merit function for the line search (Kraft 1988, N&W 18.3).
            auto merit = [&](const Eigen::Vector<double, N>& xk) -> double {
                double fval = s.problem->value(xk);
                double viol = 0.0;

                if constexpr(constrained<P>)
                {
                    if(s.n_eq + s.n_ineq > 0)
                    {
                        s.problem->constraints(xk, s.bufs.c_trial_buf);
                        auto ceq = s.bufs.c_trial_buf.head(s.n_eq);
                        auto cineq = s.bufs.c_trial_buf.tail(s.n_ineq);

                        if(ceq.size() > 0)
                            viol += ceq.cwiseAbs().sum();
                        if(cineq.size() > 0)
                            viol += (-cineq).cwiseMax(0.0).sum();
                    }
                }

                return fval + s.sigma * viol;
            };

            double merit_0 = merit(s.x);

            // h4-weighted directional derivative of the L1 merit (Kraft
            // 1988 §3.4). When the QP wrapper produced the step via
            // augmentation (relaxation_factor > 0), the augmented step
            // targets a (1 - s_aug)-scaled relaxation of the original
            // linearized constraints; weighting the violation term by
            // h4 = 1 - s_aug gives the slope of the merit along p that
            // reflects the residual violation the step actually attacks.
            // For the direct path (relaxation_factor = 0), h4 = 1 and
            // dphi_merit reduces to the standard unweighted slope.
            //
            // The h4 slope feeds the Armijo line search below; rejection
            // of the augmented step is deferred to a post-line-search
            // guard so an alpha-shrunk version of p has a chance to
            // satisfy the merit decrease test even when the unit-step
            // slope is non-negative. Only when the line search and the
            // second-order correction both fail to find a decreasing
            // step do we reset BFGS and retry (or null-step on cap
            // exhaustion).
            //
            // Reference: Kraft, D. (1988). DFVLR-FB 88-28, §3 (line
            //            search) and §3.4 (Inconsistent Linearization).
            //            Nocedal & Wright (2006). Numerical Optimization,
            //            2e, §18.3 (L1 merit function for SQP line search).
            //
            // Adopted from: argmin/detail/merit_function.h l1_merit_dphi_h4.
            const double h4 = 1.0 - qp_res.relaxation_factor;
            const double grad_f_dot_p = s.g.dot(p);
            double dphi_merit = argmin::detail::l1_merit_dphi_h4<double>(
                grad_f_dot_p, s.c_eq, s.c_ineq, s.sigma, h4);

            // If the L1 merit directional derivative is non-negative, the
            // penalty parameter sigma is too small to make p a descent
            // direction. Bump sigma so dphi becomes strictly negative,
            // capped at 1e10 to bound runaway growth on near-feasible
            // iterates.
            //
            // Reference: Powell (1978) "A fast algorithm for nonlinearly
            //            constrained optimization calculations", §6;
            //            N&W 2e Section 18.5 eq. 18.36.
            //
            // Adopted from: argmin/detail/merit_function.h bump_sigma_for_descent.
            const double sigma_bumped = argmin::detail::bump_sigma_for_descent<double>(
                s.sigma, grad_f_dot_p, constraint_viol_0, h4, 1e10);
            if(sigma_bumped > s.sigma)
            {
                s.sigma = sigma_bumped;
                dphi_merit = argmin::detail::l1_merit_dphi_h4<double>(
                    grad_f_dot_p, s.c_eq, s.c_ineq, s.sigma, h4);
            }

            // Backtracking Armijo line search on the L1 merit.
            auto phi_ls = [&](double alpha_) {
                ++s.line_search_calls;
                // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
                s.bufs.x_trial_buf.noalias() = s.x + alpha_ * p;
                s.bufs.x_trial_buf = detail::project(s.bufs.x_trial_buf, s.lower, s.upper);
                return merit(s.bufs.x_trial_buf);
            };

            auto ls = armijo(phi_ls, merit_0, dphi_merit, options.line_search);
            alpha = ls.alpha;
            bool ls_success = ls.success;
            nan_eval_count += ls.diagnostics.nan_eval_count;

            // Second-order correction (Kraft 1988 Section 2.2.4,
            // N&W Section 18.3 "Maratos effect").
            //
            // If the Armijo line search rejects the full step because
            // the linearization underestimates the constraint curvature
            // (Maratos effect), re-solve the QP with an updated RHS
            // that subtracts the nonlinear constraint residual at the
            // trial point x + p. Using the ORIGINAL Jacobian J_k (not
            // J_{k+1}) preserves the QP solver factorizations.
            //
            // RHS update:
            //   b_eq_soc  = -c_eq(x + p)  + J_eq(x)  * p
            //   b_ineq_soc = -c_ineq(x + p) + J_ineq(x) * p
            //
            // The correction step dp is added to p and the line search
            // is retried with the combined direction.
            if(!ls.success && constraint_viol_0 > options.soc_violation_threshold)
            {
                if constexpr(constrained<P>)
                {
                    if(s.n_eq + s.n_ineq > 0)
                    {
                        // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
                        s.bufs.x_trial_buf.noalias() = s.x + p;
                        s.bufs.x_trial_buf = detail::project(s.bufs.x_trial_buf, s.lower, s.upper);

                        s.problem->constraints(s.bufs.x_trial_buf, s.bufs.c_trial_buf);
                        auto c_eq_trial = s.bufs.c_trial_buf.head(s.n_eq);
                        auto c_ineq_trial = s.bufs.c_trial_buf.tail(s.n_ineq);

                        if(s.n_eq > 0)
                            s.bufs.b_eq_soc_buf.noalias() = -c_eq_trial + s.J_eq * p;
                        if(s.n_ineq > 0)
                            s.bufs.b_ineq_soc_buf.noalias() = -c_ineq_trial + s.J_ineq * p;

                        // Reuse the (E, f) factored on this step's main QP
                        // solve: the Hessian and gradient have not changed
                        // between the main solve and this SOC retry, only
                        // the constraint RHS (b_eq_soc_buf, b_ineq_soc_buf).
                        auto soc_res = s.qp_solver.solve_with_factored_hessian(
                            s.bufs.E_buf, s.bufs.f_buf, s.g,
                            s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                            s.bufs.p_lo_buf, s.bufs.p_hi_buf);

                        if(soc_res.status == detail::qp_status::optimal)
                        {
                            s.p_combined_buf.noalias() = p + soc_res.x;
                            auto phi_soc = [&](double a) {
                                ++s.line_search_calls;
                                // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
                                s.bufs.x_trial_buf.noalias() = s.x + a * s.p_combined_buf;
                                s.bufs.x_trial_buf = detail::project(s.bufs.x_trial_buf, s.lower, s.upper);
                                return merit(s.bufs.x_trial_buf);
                            };
                            auto ls_soc = armijo(phi_soc, merit_0, dphi_merit,
                                                 options.line_search);
                            nan_eval_count += ls_soc.diagnostics.nan_eval_count;
                            if(ls_soc.success && ls_soc.alpha > alpha)
                            {
                                p = s.p_combined_buf;
                                alpha = ls_soc.alpha;
                                ls_success = true;
                            }
                        }
                    }
                }
            }

            if(ls_success)
                break;

            // Line search and SOC retry both failed to find a decreasing
            // step. Reset BFGS to identity (Shanno-rescale on next push;
            // dense_ldl_bfgs.h:104 zeroes updates_since_reset_) and retry
            // the QP with B = I. On exhaustion of the cap, return a
            // null-step with diagnostics. This unifies what was
            // previously two separate branches: a terminal augmented-
            // path reset (Kraft 1988 §3.4) and a direct-path force-
            // continue fallback (small-alpha forced step) that distorted
            // the next BFGS curvature pair.
            //
            // Reference: NLopt slsqp.c:1890-1895 (ireset retry parity);
            //            Kraft 1988 DFVLR-FB 88-28 §3 (line-search recovery);
            //            dense_ldl_bfgs.h:86-105 (reset semantics).
            if(reset_count >= reset_max)
            {
                // Adopted from: argmin/detail/sqp_common.h extract_qp_multipliers.
                s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
                s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
                if constexpr(constrained<P>)
                {
                    if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                    {
                        argmin::detail::extract_qp_multipliers<double>(
                            qp_res.lambda, s.n_eq, s.n_ineq,
                            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
                    }
                }

                // Adopted from: argmin/detail/sqp_common.h null_step_result.
                auto r = argmin::detail::null_step_result<double, N,
                                                          Eigen::Dynamic,
                                                          Eigen::Dynamic>(
                    s.objective_value, s.g, s.J_eq, s.J_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                    s.c_eq, s.c_ineq, s.x.norm(), reset_count);
                // kraft reports gradient_norm via the previous-step
                // s.lambda for cross-policy consistency with the accepted-
                // step path; override the helper's internal computation.
                r.gradient_norm = lagrangian_gradient_norm(s);
                // Surface the Armijo NaN-recovery count accumulated across
                // the main LS and any SOC retry within this step() call.
                r.diagnostics.nan_eval_count = nan_eval_count;
                return r;
            }

            s.hessian.reset();
            ++reset_count;
        }

        // Update iterate
        s.bufs.x_old_buf = s.x;
        Eigen::Vector<double, N>& x_old = s.bufs.x_old_buf;
        double old_f = s.objective_value;
        s.x = s.x + alpha * p;

        // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
        s.x = detail::project(s.x, s.lower, s.upper);

        s.objective_value = s.problem->value(s.x);

        s.bufs.g_old_buf = s.g;
        Eigen::Vector<double, N>& g_old = s.bufs.g_old_buf;
        s.problem->gradient(s.x, s.g);

        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                // Cache J(x_k) into J_all_old before the next-iter
                // Jacobian overwrites s.bufs.J_all. The BFGS curvature-pair
                // update below needs J(x_old) = J(x_k) to compute
                // grad_L_old = g_k - J(x_k)^T lambda; with this cache
                // there is no second constraint_jacobian(x_old, ...) call.
                // First-step initialization is correct: init() populates
                // s.bufs.J_all with J(x_0) and the first step's x_old
                // is x_0, so the cache holds J(x_0) on entry to the BFGS
                // block.
                s.bufs.J_all_old = s.bufs.J_all;

                s.problem->constraints(s.x, s.bufs.c_all);
                s.c_eq = s.bufs.c_all.head(s.n_eq);
                s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

                s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
            }
        }

        // BFGS update driven by the Lagrangian gradient difference.
        //
        // y_k = grad_L(x_{k+1}, lam) - grad_L(x_k, lam) with
        // grad_L(x, lam) = g(x) - A(x)^T lam. Each gradient is
        // evaluated at its own iterate's Jacobian so that y_k picks
        // up the second-order constraint curvature term
        // (A_{k+1} - A_k)^T lam alongside the objective curvature
        // g_{k+1} - g_k. This is Kraft's SLSQP variant rather than
        // the "fixed Jacobian" modification in N&W Section 18.4.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (SLSQP
        //            Hessian of the Lagrangian);
        //            N&W eq. 18.13 (Lagrangian Hessian).
        Eigen::Vector<double, N>& grad_L_old = s.bufs.grad_L_old_buf;
        Eigen::Vector<double, N>& grad_L_new = s.bufs.grad_L_new_buf;

        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0 && qp_res.lambda.size() > 0)
            {
                const int m_total = s.n_eq + s.n_ineq;
                const int lam_take = std::min(m_total,
                                              static_cast<int>(qp_res.lambda.size()));
                if(s.bufs.lam_buf.size() < lam_take) s.bufs.lam_buf.resize(lam_take);
                s.bufs.lam_buf.head(lam_take) = qp_res.lambda.head(lam_take);

                if(lam_take == m_total)
                    s.lambda = s.bufs.lam_buf.head(m_total);

                if(lam_take == m_total)
                {
                    // Adopted from: argmin/detail/sqp_common.h compute_bfgs_pair_fused
                    //               (in-tree precedent).
                    argmin::detail::compute_bfgs_pair_fused<double, N>(
                        g_old, s.g, s.bufs.J_all_old, s.bufs.J_all,
                        s.bufs.lam_buf.head(m_total), m_total,
                        grad_L_old, grad_L_new,
                        s.bufs.sk_buf, s.bufs.yk_buf,
                        s.x, x_old);
                }
                else
                {
                    // Eq-only fallback (kraft_slsqp_policy::compute_bfgs_pair_fused
                    // m_total branch above): the helper's m_total
                    // branch only handles the full-multiplier case; the
                    // partial-multiplier path stays inline because it touches
                    // s.J_eq (policy-private state) rather than J_all_old.topRows.
                    grad_L_old = g_old;
                    grad_L_new = s.g;
                    if(s.n_eq > 0 && lam_take >= s.n_eq)
                    {
                        s.bufs.lam_eq_buf.head(s.n_eq) = s.bufs.lam_buf.head(s.n_eq);
                        grad_L_old.noalias() -= s.bufs.J_all_old.topRows(s.n_eq).transpose()
                                               * s.bufs.lam_eq_buf.head(s.n_eq);
                        grad_L_new.noalias() -= s.J_eq.transpose()
                                               * s.bufs.lam_eq_buf.head(s.n_eq);
                    }
                    s.bufs.sk_buf.noalias() = s.x - x_old;
                    s.bufs.yk_buf.noalias() = grad_L_new - grad_L_old;
                }
            }
            else
            {
                grad_L_old = g_old;
                grad_L_new = s.g;
                s.bufs.sk_buf.noalias() = s.x - x_old;
                s.bufs.yk_buf.noalias() = grad_L_new - grad_L_old;
            }
        }
        else
        {
            grad_L_old = g_old;
            grad_L_new = s.g;
            s.bufs.sk_buf.noalias() = s.x - x_old;
            s.bufs.yk_buf.noalias() = grad_L_new - grad_L_old;
        }

        Eigen::Vector<double, N>& sk = s.bufs.sk_buf;
        Eigen::Vector<double, N>& yk = s.bufs.yk_buf;

        // Skip BFGS updates on near-zero or non-positive curvature
        // pairs. In SLSQP the Lagrangian gradient difference y_k can
        // easily have s^T y < 0 when the constraint Hessian contribution
        // (A_{k+1} - A_k)^T lam dominates the objective curvature; the
        // dense_ldl_bfgs::push call below applies Powell damping per
        // N&W eq. 18.22-18.24 internally, but the explicit guard here
        // keeps the policy-step semantics legible and avoids the
        // damping logic for the trivial null-step / non-curvature case.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard.
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
        {
            // dense_ldl_bfgs::push applies the Shanno (1978) initial
            // scaling theta = y^T y / s^T y on the first call after
            // construction or reset (N&W eq. 6.20), then evolves the
            // packed L D L^T factor by Powell-damped rank-1 LDL updates
            // for every subsequent pair (Fletcher-Powell 1974). The
            // damping safeguard ensures B remains SPD even when the
            // Lagrangian curvature shifts toward a degenerate
            // constrained optimum (HS026 family).
            s.hessian.push(sk, yk);
        }

        ++s.iteration;

        // KKT residual: full first-order optimality error E(x, lambda,
        // mu) as the L-infinity maximum over five legs (stationarity,
        // primal equality feasibility, primal inequality feasibility,
        // dual feasibility, complementarity). Multipliers come from the
        // QP solution; equality multipliers occupy the first n_eq
        // entries of qp_res.lambda and inequality multipliers follow.
        // When m == 0 the helper collapses to ||grad_f||_inf.
        //
        // Kraft's policy uses the QP-native multipliers here rather
        // than an active-set least-squares estimate at the post-step
        // iterate. The QP returns dual-feasible multipliers for its
        // own KKT conditions (Goldfarb-Idnani), and at convergence
        // qp_res.lambda agrees with any post-step LS estimate; off
        // convergence the QP-native source avoids a per-iter ColPivQR on the
        // active-set Jacobian. Sibling SQP policies (nw_sqp, filter_slsqp,
        // filter_nw_sqp) keep the active-set LS path via
        // detail::compute_kkt_multipliers_active_set in lagrangian.h.
        //
        // Reference: N&W 2e Definition 12.1 (KKT conditions:
        //            stationarity, primal feasibility, dual feasibility,
        //            complementarity); eq. 12.34 (Lagrangian
        //            stationarity leg); Goldfarb-Idnani 1983
        //            (active-set QP duality, kraft_lsq_qp lineage).
        //
        // NOTE: options.multiplier_reest_every_k is intentionally NOT
        // consulted here. The QP-native copy below is the multiplier
        // source for kraft on every step and runs unconditionally; the
        // field exists on options_type for API uniformity across the
        // four line-search SQP policies but is a documented no-op on
        // the Kraft hot path.
        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
        if constexpr(constrained<P>)
        {
            if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
            {
                if(s.n_eq > 0)
                    s.bufs.kkt_lambda_eq_buf = qp_res.lambda.head(s.n_eq);
                if(s.n_ineq > 0)
                    s.bufs.kkt_mu_ineq_buf = qp_res.lambda.segment(s.n_eq, s.n_ineq);
            }
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency with
        // step_result.kkt_residual. L1-merit internal paths keep L1 per
        // N&W eq. 15.24 (merit semantics distinct from reporting semantics).
        //
        // Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = lagrangian_gradient_norm(s),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .diagnostics = {
                .bfgs_reset_count = reset_count,
                .bfgs_skip_count = 0,
                .nan_eval_count = nan_eval_count,
            },
        };
    }

    // Hot start -- preserves BFGS curvature information.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                s.problem->constraints(x0, s.bufs.c_all);
                s.c_eq = s.bufs.c_all.head(s.n_eq);
                s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

                s.problem->constraint_jacobian(x0, s.bufs.J_all);
                s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
            }
        }
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS curvature information (B := I).
    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        s.sigma = options.initial_penalty;
    }

private:
    // Lagrangian gradient norm for KKT-consistent stationarity reporting.
    //
    // grad_L = grad_f - [J_eq; J_ineq]^T * lambda. At a constrained optimum
    // grad_L vanishes (N&W eq. 12.34) while raw ||grad_f|| in general does
    // not. Convergence criteria gate on Lagrangian-gradient norm.
    //
    // Adopted from: argmin/solver/nw_sqp_policy.h::step null-step branch (in-tree precedent).
    // Reference: N&W 2e Section 12.3 / eq. 12.34 (Lagrangian stationarity);
    //            eq. 18.2-18.3 (Lagrangian gradient definition).
    //
    // argmin variant: per-policy static helper parameterised on state_type<P>;
    //                 a shared free-function variant is deferred to a later
    //                 helper-extraction pass. Rationale: state_type members
    //                 differ across SQP policies; a shared helper would need
    //                 a concept-or-trait abstraction not yet in place.
    template <typename P>
    static double lagrangian_gradient_norm(const state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;
        if(m == 0)
            return s.g.norm();

        Eigen::Matrix<double, Eigen::Dynamic, N> A(m, n);
        if(s.n_eq > 0) A.topRows(s.n_eq) = s.J_eq;
        if(s.n_ineq > 0) A.bottomRows(s.n_ineq) = s.J_ineq;
        return detail::lagrangian_gradient(s.g, A, s.lambda).norm();
    }
};

template <int N = dynamic_dimension>
using kraft_slsqp_policy_fast = kraft_slsqp_policy<N, sqp_mode::fast>;

template <int N = dynamic_dimension>
using kraft_slsqp_policy_accurate = kraft_slsqp_policy<N, sqp_mode::accurate>;

}

#endif
