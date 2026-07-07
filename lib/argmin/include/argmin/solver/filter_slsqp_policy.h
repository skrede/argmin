#ifndef HPP_GUARD_ARGMIN_SOLVER_FILTER_SLSQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_FILTER_SLSQP_POLICY_H

// Filter SQP policy for step_budget_solver.
//
// Mirrors kraft_slsqp_policy in QP solver (kraft_lsq_qp), Hessian
// (dense_ldl_bfgs, packed L D L^T), and constraint handling. Differs
// in globalization:
// Fletcher-Leyffer 2002 filter acceptance with Wachter-Biegler 2006
// switching condition replaces the L1 merit function, eliminating the
// penalty parameter sigma.
//
// Filter acceptance partitions iterations into f-type (near-feasible,
// Armijo on f) and h-type (infeasible, filter dominance test). A
// second-order correction fires on severe Maratos cases where the
// filter rejects the full step despite high constraint violation.
// Feasibility restoration via hybrid L1-then-QP recovers when the
// filter blocks all step sizes.
//
// Reference: Fletcher & Leyffer 2002, "Nonlinear programming without
//            a penalty function", Math. Program. 91:239-269.
//            Wachter & Biegler 2006, "On the implementation of an
//            interior-point filter line-search algorithm for large-scale
//            nonlinear programming", Math. Program. 106:25-57.
//            N&W Section 15.5 (filter methods).
//            Kraft 1988 DFVLR-FB 88-28 (QP solver, Hessian update).

#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/filter_acceptance.h"
#include "argmin/detail/filter_restoration.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/merit_function.h"
#include "argmin/detail/bound_projection.h"

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
#include <optional>
#include <algorithm>

namespace argmin
{

// argmin variant: single-mode filter-acceptance SLSQP policy. The earlier
//                 per-mode dispatch (fast / accurate as a closed-set NTTP)
//                 is removed here: empirical bench measurements showed
//                 `fast` lost wall-time AND iteration count against
//                 `accurate` on every measured cell across the entire
//                 line-search SQP family. The strictly-worse mode has been
//                 removed; the surviving defaults below are the former
//                 `accurate` values.
//
// Reference: Wachter & Biegler 2006 Section 2.3 (filter envelope
//            semantics, eq. 6); Fletcher & Leyffer 2002 Section 2.2
//            (filter dominance ordering).
template <int N = dynamic_dimension>
struct filter_slsqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = filter_slsqp_policy<M>;

    static constexpr double default_filter_gamma_f = 1e-5;
    static constexpr double default_filter_gamma_h = 1e-5;
    static constexpr double default_gradient_tolerance = 1e-8;
    static constexpr double default_step_tolerance_rel = 1e-12;
    static constexpr double default_feasibility_tolerance = 1e-6;

    // Armijo backtracking budget (NLopt-parity sustained backtracking
    // before declaring exhaustion).
    //
    // Reference: NLopt slsqp.c:1803-2189 (slsqpb_ outer line-search budget).
    static constexpr std::uint16_t default_line_search_max_iterations = 40;

    // Armijo sufficient-decrease parameter (Wolfe-condition convention).
    //
    // Reference: N&W 2e §3.1 (Wolfe-condition convention);
    //            NLopt slsqp.c default c1 = 1e-4.
    static constexpr double default_armijo_c1 = 1e-4;

    // Armijo backtracking shrink factor (NLopt slsqp.c default).
    //
    // Reference: NLopt slsqp.c default rho = 0.5;
    //            N&W 2e §3.5 (backtracking shrink factor range 0.1..0.9).
    static constexpr double default_armijo_rho = 0.5;

    // BFGS Hessian-reset retry cap on filter / line-search exhaustion.
    // Matches NLopt slsqp.c's ireset semantics with up to 5 retries.
    //
    // Reference: NLopt slsqp.c:1890-1895 (ireset retry pattern).
    static constexpr std::size_t default_bfgs_reset_max = 5;

    // QP inner solver iteration cap (NLopt slsqp.c LSI/LSEI cascade
    // convention).
    //
    // NOTE: filter_slsqp shares kraft_lsq_qp_recovery_solver with
    // kraft_slsqp. The recovery solver does not currently thread
    // options.qp values into its solve API; brace initialization lives
    // for API uniformity with the N&W lineage and to make the value
    // observable on options.qp.
    //
    // Reference: NLopt slsqp.c:700-1100 (LSI/LSEI cap convention).
    static constexpr std::uint16_t default_qp_max_iterations = 200;

    // QP convergence tolerance (NLopt's double-precision baseline).
    //
    // NOTE: Same dead-wire as default_qp_max_iterations above.
    //
    // Reference: Kraft 1988 §3.5 (QP convergence acc^2 threshold).
    static constexpr double default_qp_tolerance = 1e-12;

    // Active-set Lagrange-multiplier re-estimation stride. The post-step
    // KKT-leg invokes compute_kkt_multipliers_active_set only when
    // (s.iteration % multiplier_reest_every_k == 0); on the skip path
    // the prior step's s.bufs.kkt_lambda_eq_buf / s.bufs.kkt_mu_ineq_buf
    // are reused (state-resident, zero-alloc). k=1 recomputes every
    // step; k>1 trades a stationarity-leg lag of ~k steps for k-1x
    // savings on the per-step ColPivHouseholderQR.
    //
    // Reference: Bertsekas 1996 §4.2 (stale-multiplier reuse rationale);
    //            N&W 2e §18.3 + Algorithm 18.3 (working-set
    //            identification, eq. 18.15).
    static constexpr std::size_t default_multiplier_reest_every_k = 1;

    struct options_type
    {
        // Embedded line-search params. Designated-initializer order follows
        // line_search_options field-declaration order (c1, c2, rho,
        // max_alpha, max_iterations); skipped designators (c2, max_alpha)
        // keep their line_search_options-side defaults (0.9 and 1.0).
        line_search_options line_search{
            .c1 = default_armijo_c1,
            .rho = default_armijo_rho,
            .max_iterations = default_line_search_max_iterations,
        };

        // QP subproblem params. Dead-wired on filter_slsqp at present
        // (kraft_lsq_qp_recovery_solver does not consume options.qp);
        // brace-initialized here for API uniformity with the N&W lineage.
        qp_options qp{
            .max_iterations = default_qp_max_iterations,
            .tolerance = default_qp_tolerance,
        };
        detail::restoration_strategy restoration{detail::restoration_strategy::hybrid};
        std::uint16_t max_restoration_steps{10};
        std::uint16_t stall_window{50};

        // Filter envelope margins (Wachter & Biegler 2006 Section 2.3,
        // eq. 6). Direct-field-default form: per-mode default_filter_gamma_*
        // static-constexpr above brace-initializes the field. Callers who
        // want a non-default value assign the field directly.
        //
        // Reference: Wachter & Biegler 2006 Section 2.3;
        //            Fletcher & Leyffer 2002 Section 5.
        double gamma_f{default_filter_gamma_f};
        double gamma_h{default_filter_gamma_h};

        // BFGS-reset retry cap on line-search/filter exhaustion. After
        // restoration fails to recover an acceptable trial point, the policy
        // resets the BFGS Hessian to identity and re-solves the QP, repeating
        // up to bfgs_reset_max times before returning a null-step.
        //
        // Reference: NLopt slsqp.c slsqpb_ outer loop (line-search exhaustion
        //            fallback);
        //            NLopt slsqp.c:1890-1895 (ireset retry pattern);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: filter policies layer this AFTER the existing
        //                 restoration path (restoration first then BFGS
        //                 reset then null-step). Cascade-free: no new
        //                 solver_status enum entry.
        std::size_t bfgs_reset_max{default_bfgs_reset_max};

        // Active-set multiplier re-estimation stride (cf. the per-policy
        // default_multiplier_reest_every_k constexpr above). At
        // (s.iteration % k == 0) the post-step KKT-leg invokes the
        // active-set LS helper; otherwise the prior step's
        // s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf are reused.
        // A value of 0 is treated as 1 (re-estimate every step) by the read-site clamp.
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
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::kraft_lsq_qp_recovery_solver<double, N> qp_solver;
        detail::filter_set<double> filter;

        // Cross-policy state-resident buffer struct. Consolidates the
        // per-step / per-line-search trial buffers, the BFGS curvature-
        // pair buffers, the constraint-axis workspaces (c_all, c_trial,
        // b_eq / b_ineq, b_eq_soc / b_ineq_soc, lam / lam_eq / lam_ineq,
        // kkt_lambda_eq / kkt_mu_ineq), the constraint Jacobian pair
        // (J_all and the J_all_old cached copy that eliminates the
        // second constraint_jacobian(x_old, ...) call on the BFGS
        // curvature-pair path), and the
        // pre-factored Hessian buffers (E_buf / f_buf) consumed by
        // kraft_lsq_qp_recovery_solver::solve_with_factored_hessian.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers
        //               (in-tree precedent: generalizes the kraft *_buf
        //                state-resident layout).
        // Reference: N&W 2e Section 18.
        //
        // argmin variant: cross-policy state-resident buffer struct.
        argmin::detail::sqp_state_buffers<double, N> bufs;

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
                             const solver_options<Convergence>&)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n = n;
        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

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

        double h_0 = 0.0;
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
                h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
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

        s.hessian = detail::dense_ldl_bfgs<double, N>(n);
        s.iteration = 0;

        // Initialize filter with h_max per Wachter-Biegler 2006 eq. (8)
        // and thread the configured envelope margins onto the filter
        // (Wachter & Biegler 2006 Section 2.3, eq. 6). Defaults
        // 1e-5 / 1e-5 preserve the previously-shipped behavior when the
        // options are unset.
        s.filter.initialize(1e4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f, options.gamma_h);

        s.qp_solver.resize(n, s.n_eq, s.n_ineq, n, n);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.n;

        // Cold-start mu calibration: documented no-op for filter_slsqp.
        //
        // The cold-start fix (calibrate_initial_penalty after iter-0 QP solve)
        // applies to SQP policies whose main acceptance test is L1 merit:
        // kraft_slsqp, nw_sqp, filter_nw_sqp. filter_slsqp's main loop uses
        // filter dominance (Wachter & Biegler 2006 Section 2.3) and does
        // not consume sigma on the accepted path. The only sigma site in
        // this policy is sigma_restore in the restoration helper, which
        // already implements the equivalent of the cold-start formula
        // (sigma_restore = ||lambda||_inf + 1e-4, matching N&W eq. 18.36
        // up to the choice of safety constant).
        //
        // Reference: N&W 2e eq. 18.36 (sigma sufficient for L1-merit descent);
        //            Wachter & Biegler 2006 Section 2.3 (filter acceptance);
        //            cold-start applies only to merit-based policies (the
        //            qp-lambda-driven sigma seed bounds first-step rejection
        //            on L1-merit acceptance).
        //
        // argmin variant: filter_slsqp main loop is filter-based, so the
        //                 main-path cold-start is a no-op; sigma calibration
        //                 lives in the restoration helper. Future helper
        //                 extraction may unify these into a single shared
        //                 cold-start site; preserved in current form for
        //                 bit-identical regression compatibility.

        // BFGS-reset retry layered after the restoration path.
        //
        // Filter policies prefer feasibility restoration (which explicitly
        // minimises ||c||) over a BFGS reset on rejection, because the
        // restoration step is conceptually cheaper than discarding curvature.
        // The retry ordering is therefore: restoration first then BFGS-reset
        // retry then null-step. This preserves the existing restoration
        // semantics while adding the BFGS-reset escape hatch for the case
        // where the restoration helper also fails to recover an acceptable
        // trial point.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h::step retry-loop
        //               pattern (in-tree precedent).
        //               NLopt slsqp.c:1890-1895 (ireset retry pattern, max=5).
        // Reference: NLopt slsqp.c slsqpb_ outer loop (line-search exhaustion
        //            fallback);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: cap is options.bfgs_reset_max (default 5);
        //                 status on exhaustion is is_null_step + the
        //                 diagnostics.bfgs_reset_count counter; rationale
        //                 is cascade-free (no new solver_status enum entry).
        std::size_t reset_count = 0;
        const std::size_t reset_max = options.bfgs_reset_max;
        // Armijo NaN/Inf recovery counter aggregated across the main inline
        // filter line search and the SOC retry inline loop. filter_slsqp
        // hand-rolls the backtracking loops (does not call
        // line_search/armijo.h) because the inline filter-acceptance + Armijo
        // f-descent pattern avoids per-backtrack VectorXd materialisation;
        // this counter mirrors armijo()'s diagnostic on the hand-rolled
        // paths. Incremented inline when a trial-iterate evaluation returns
        // non-finite f_trial or constraint values; the backtracker shrinks
        // alpha and continues. Both modes enable the gate.
        //
        // NaN/Inf gate: see argmin/line_search/armijo.h header comment.
        std::size_t nan_eval_count = 0;
        // Second-order-correction retry counter. Increments once per SOC
        // trigger (Maratos-effect rejection) regardless of whether the
        // correction step is ultimately accepted; surfaced on every
        // step_result return path for telemetry.
        std::size_t soc_retry_count = 0;

        // Loop-carried variables that survive the retry block into the
        // post-loop accept-step block. Restoration-success returns inline
        // from inside the loop; the only path that exits via `break` is
        // the LS/filter accept path (accepted = true).
        detail::qp_result<double, N> qp_res;
        Eigen::Vector<double, N>& p = s.bufs.p_buf;
        double alpha = 1.0;
        // f_k and h_k are computed from s.objective_value / s.c_eq / s.c_ineq
        // which do not change between BFGS-reset retries (s.x is fixed
        // during the retry); fire the h-type filter add at most once per
        // outer step() invocation.
        bool filter_added = false;

        for(;;)
        {
        // Factor the BFGS LDL Hessian into (E, f) directly off the
        // packed L, D factors -- skipping the O(n^3 / 3) LLT(B) the QP
        // solver would otherwise run on every solve. See
        // detail::dense_ldl_bfgs::factor_to_E_and_f.
        s.hessian.factor_to_E_and_f(s.bufs.E_buf, s.g, s.bufs.f_buf);

        // QP subproblem: identical to kraft_slsqp.
        s.bufs.b_eq_workspace = -s.c_eq;
        s.bufs.b_ineq_workspace = -s.c_ineq;

        s.bufs.p_lo_buf.noalias() = s.lower - s.x;
        s.bufs.p_hi_buf.noalias() = s.upper - s.x;

        qp_res = s.qp_solver.solve_with_factored_hessian(
            s.bufs.E_buf, s.bufs.f_buf, s.g,
            s.J_eq, s.bufs.b_eq_workspace,
            s.J_ineq, s.bufs.b_ineq_workspace,
            s.bufs.p_lo_buf, s.bufs.p_hi_buf);

        s.bufs.p_buf = qp_res.x;
        double p_norm = p.norm();
        bool zero_step_restoration_failed = false;
        bool accepted = false;
        if(p_norm < 1e-15)
        {
            // Zero step with high constraint violation: attempt restoration
            // before giving up, since the QP may be infeasible at box bounds.
            //
            // Dual convention: step_result.constraint_violation reports the
            // L-infinity primal feasibility (dimensionally consistent with
            // kkt_residual per N&W 2e Definition 12.1); filter_set entries
            // retain L1 h_k per Fletcher-Leyffer 2002 Section 2 dominance
            // ordering. The `h_zero > 1e-8` restoration trigger keeps the
            // L1 form for consistency with filter semantics at that gate.
            double h_zero_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
            bool zero_step_restoration_attempted = false;
            if constexpr(constrained<P>)
            {
                if(h_zero_l1 > 1e-8 && s.n_eq + s.n_ineq > 0)
                {
                    zero_step_restoration_attempted = true;
                    auto rest = run_restoration_(s);
                    if(rest.success)
                    {
                        // Capture x before the restoration assignment so
                        // step_size below reports the actual primal step
                        // norm (||rest.x - x_old||) rather than the
                        // residual constraint violation.
                        const double rest_step_norm = (rest.x - s.x).norm();
                        s.x = rest.x;
                        s.objective_value = s.problem->value(s.x);
                        s.problem->gradient(s.x, s.g);

                        s.problem->constraints(s.x, s.bufs.c_all);
                        s.c_eq = s.bufs.c_all.head(s.n_eq);
                        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
                        s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

                        const double h_rest_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                        s.filter.add(s.objective_value, h_rest_l1);

                        ++s.iteration;
                        return step_result<double>{
                            .objective_value = s.objective_value,
                            .gradient_norm = lagrangian_gradient_norm(s),
                            .step_size = rest_step_norm,
                            .objective_change = s.objective_value - s.objective_value,
                            .improved = false,
                            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                            .x_norm = s.x.norm(),
                            .diagnostics = {
                                .bfgs_reset_count = reset_count,
                                .bfgs_skip_count = 0,
                                .nan_eval_count = nan_eval_count,
                                .soc_retry_count = soc_retry_count,
                            },
                        };
                    }
                }
            }

            // Restoration was attempted (zero-step + high constraint
            // violation) and failed: per the restoration-first ordering,
            // fall through to the BFGS-reset retry block at the bottom of
            // the loop. The genuine-feasibility null-step path below
            // (when restoration was NOT attempted, e.g., h_zero_l1 <= 1e-8
            // or unconstrained) does not benefit from a curvature reset:
            // a true zero direction at a feasible iterate is the QP's
            // correct answer at the optimum.
            if(zero_step_restoration_attempted)
            {
                zero_step_restoration_failed = true;
            }
            else
            {

            // Null step: QP zero direction with high constraint
            // violation; filter blocks all trial steps. Kraft 1988
            // Section 2.2.3.
            //
            // is_null_step=true exempts this iterate from
            // step_tolerance stall detection so step_budget_solver does not
            // double-report the stall via the tolerance criterion on
            // top of the policy-reported solver_status::stalled below.
            //
            // Null-step branch: x_{k+1} = x_k so the active-set LS
            // re-estimate is measured at the current iterate with no
            // staleness. Populating kkt_residual here lets the outer
            // convergence check terminate on a true KKT point even when
            // the QP returns zero at the optimum (otherwise the composite
            // gate falls back to gradient_norm = ||grad_f||_inf which is
            // not zero at constrained optima and blocks termination).
            //
            // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
            argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                              Eigen::Dynamic,
                                                              Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq, s.c_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);

            // Adopted from: argmin/detail/sqp_common.h null_step_result.
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), reset_count,
                h_zero_l1 > 1e-8
                    ? std::optional{solver_status::stalled}
                    : std::nullopt);
            // filter_slsqp reports gradient_norm via the s.lambda
            // multiplier vector for cross-policy consistency with the
            // accepted-step path; override the helper's internal
            // computation from the per-leg buffers.
            r.gradient_norm = lagrangian_gradient_norm(s);
            // Mirror the cap-exhausted exit below: surface the
            // accumulated NaN-recovery count so the in-loop null-step
            // return is symmetric (filter_slsqp does not maintain a
            // bfgs_skip_count local — the Powell-damped path is
            // unconditional on the Kraft lineage).
            r.diagnostics.nan_eval_count = nan_eval_count;
            r.diagnostics.soc_retry_count = soc_retry_count;
            ++s.iteration;
            return r;
            }  // end !zero_step_restoration_attempted else-branch
        }

        // Skip the LS / SOC / restoration block when the zero-step branch
        // already attempted restoration and failed: that path falls
        // straight through to the BFGS-reset retry at the bottom of this
        // loop. The non-zero-step path (regular QP direction) runs the
        // LS / SOC / restoration sequence here.
        if(!zero_step_restoration_failed)
        {

        // Current constraint violation and objective.
        double h_k = detail::constraint_violation(s.c_eq, s.c_ineq);
        double f_k = s.objective_value;
        double grad_f_dot_p = s.g.dot(p);

        // Switching condition: f-type when near-feasible with sufficient
        // predicted f-decrease.
        // Reference: Wachter & Biegler 2006 eq. (14).
        bool f_type = detail::is_f_type_iteration(h_k, grad_f_dot_p, s.filter.h_max());

        // Add current point to filter on h-type iterations.
        // Reference: Wachter & Biegler 2006 Algorithm A.
        if(!f_type && !filter_added)
        {
            s.filter.add(f_k, h_k);
            filter_added = true;
        }

        // Backtracking with filter acceptance or Armijo f-descent.
        alpha = 1.0;
        double f_trial = f_k;
        double h_trial = h_k;
        // Constraint violation at the full unit step theta(x_k + p), captured
        // on the alpha = 1 backtracking pass. Drives the IPOPT Section 2.4
        // second-order-correction trigger below.
        double h_full_step = h_k;

        const double c1 = options.line_search.c1;
        const double rho = options.line_search.rho;
        const auto max_ls = options.line_search.max_iterations;

        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
        {
            // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
            s.bufs.x_trial_buf.noalias() = s.x + alpha * p;
            s.bufs.x_trial_buf = detail::project(s.bufs.x_trial_buf, s.lower, s.upper);

            f_trial = s.problem->value(s.bufs.x_trial_buf);
            ++s.line_search_calls;

            bool c_finite = true;
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    s.problem->constraints(s.bufs.x_trial_buf, s.bufs.c_trial_buf);
                    c_finite = s.bufs.c_trial_buf.allFinite();
                    // Inline L1 constraint violation via head/tail VectorBlock
                    // expressions to avoid a per-backtrack VectorXd
                    // materialization. Equivalent to
                    // detail::constraint_violation(c_eq, c_ineq).
                    h_trial = 0.0;
                    if(s.n_eq > 0)
                        h_trial += s.bufs.c_trial_buf.head(s.n_eq).cwiseAbs().sum();
                    if(s.n_ineq > 0)
                        h_trial += (-s.bufs.c_trial_buf.tail(s.n_ineq)).cwiseMax(0.0).sum();
                }
                else
                {
                    h_trial = 0.0;
                }
            }
            else
            {
                h_trial = 0.0;
            }

            // NaN/Inf gate: see argmin/line_search/armijo.h header comment.
            //               Hand-rolled inline-filter / inline-merit path
            //               mirrors that semantics — non-finite f_trial or
            //               constraints trigger a backtrack rather than
            //               crossing the filter dominance / Armijo
            //               comparison. Both modes enable the gate.
            if(!std::isfinite(f_trial) || !c_finite)
            {
                ++nan_eval_count;
                alpha *= rho;
                continue;
            }

            // Record the full-unit-step violation once, on the first
            // (alpha = 1) finite trial, for the SOC trigger below.
            if(ls == 0)
                h_full_step = h_trial;

            if(f_type)
            {
                // F-type acceptance (Wachter & Biegler 2006 Algorithm A):
                // filter acceptability and the h_max ceiling are required
                // in ADDITION to the Armijo f-descent test, not replaced by
                // it. A trial that improves f marginally while inflating the
                // constraint violation past the filter envelope must be
                // rejected; the Armijo test alone does not see the violation.
                if(s.filter.is_acceptable(f_trial, h_trial)
                   && f_trial <= f_k + c1 * alpha * grad_f_dot_p
                   && h_trial <= s.filter.h_max())
                {
                    accepted = true;
                    break;
                }
            }
            else
            {
                // Filter acceptance check.
                if(s.filter.is_acceptable(f_trial, h_trial))
                {
                    accepted = true;
                    break;
                }
            }

            alpha *= rho;
        }

        // Second-order correction for the Maratos effect. Fires when the
        // full unit step was rejected AND it did not reduce the constraint
        // violation (theta(x_k + p) >= theta(x_k)), the signature of the
        // linearization underestimating constraint curvature near a feasible
        // iterate. This is the IPOPT Section 2.4 trigger, independent of the
        // current iterate's own violation level.
        //
        // The full unit step counts as rejected whenever it was not
        // accepted at alpha = 1 (either the line search failed outright or
        // it accepted a backtracked short step, alpha < 1) -- the Maratos
        // effect manifests precisely as a backtracked acceptance replacing
        // the quadratic full step.
        //
        // Reference: Wachter & Biegler 2006 Section 2.4;
        //            N&W Section 18.3 (Maratos effect).
        if((!accepted || alpha < 1.0) && h_full_step >= h_k)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    ++soc_retry_count;
                    // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
                    Eigen::Vector<double, N> x_full = s.x + p;
                    x_full = detail::project(x_full, s.lower, s.upper);

                    s.problem->constraints(x_full, s.bufs.c_trial_buf);
                    auto c_eq_full = s.bufs.c_trial_buf.head(s.n_eq);
                    auto c_ineq_full = s.bufs.c_trial_buf.tail(s.n_ineq);

                    if(s.n_eq > 0)
                        s.bufs.b_eq_soc_buf.noalias() = -c_eq_full + s.J_eq * p;
                    if(s.n_ineq > 0)
                        s.bufs.b_ineq_soc_buf.noalias() = -c_ineq_full + s.J_ineq * p;

                    // Reuse the (E, f) factored on this step's main QP
                    // solve: Hessian and gradient unchanged, only the
                    // constraint RHS differs (SOC retry).
                    auto soc_res = s.qp_solver.solve_with_factored_hessian(
                        s.bufs.E_buf, s.bufs.f_buf, s.g,
                        s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                        s.bufs.p_lo_buf, s.bufs.p_hi_buf);

                    if(soc_res.status == detail::qp_status::optimal)
                    {
                        // The SOC RHS -c(x + p) + J * p re-anchors the
                        // linearized constraints at the full unit step, so
                        // the re-solved QP direction is ALREADY the full
                        // corrected step from the current iterate: the trial
                        // is x + p_soc, NOT x + p + p_soc. Composing p again
                        // double-counts the step and lands at
                        // x + 2p + O(||p||^2), re-lifting the violation the
                        // correction was meant to remove (N&W 2e Section 18.3
                        // replacement-form SOC).
                        Eigen::Vector<double, N> p_soc = soc_res.x;
                        double alpha_soc = 1.0;

                        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
                        {
                            // Adopted from: argmin/detail/bound_projection.h::project (in-tree precedent).
                            s.bufs.x_trial_buf.noalias() = s.x + alpha_soc * p_soc;
                            s.bufs.x_trial_buf = detail::project(s.bufs.x_trial_buf, s.lower, s.upper);

                            double f_soc = s.problem->value(s.bufs.x_trial_buf);
                            ++s.line_search_calls;

                            double h_soc = 0.0;
                            bool c_soc_finite = true;
                            if(s.n_eq + s.n_ineq > 0)
                            {
                                s.problem->constraints(s.bufs.x_trial_buf, s.bufs.c_trial_buf);
                                c_soc_finite = s.bufs.c_trial_buf.allFinite();
                                // Inline L1 constraint violation via head/tail
                                // VectorBlock expressions to avoid a per-backtrack
                                // VectorXd materialization (see main LS).
                                if(s.n_eq > 0)
                                    h_soc += s.bufs.c_trial_buf.head(s.n_eq).cwiseAbs().sum();
                                if(s.n_ineq > 0)
                                    h_soc += (-s.bufs.c_trial_buf.tail(s.n_ineq)).cwiseMax(0.0).sum();
                            }

                            // NaN/Inf recovery on the SOC retry path
                            // (parallel to the main-LS inline gate above).
                            // Shrink alpha_soc and continue on non-finite
                            // trial-iterate evaluation.
                            if(!std::isfinite(f_soc) || !c_soc_finite)
                            {
                                ++nan_eval_count;
                                alpha_soc *= rho;
                                continue;
                            }

                            bool soc_accepted = false;
                            if(f_type)
                            {
                                // Same both-conditions f-type rule as the
                                // main line search: filter acceptability and
                                // the h_max ceiling gate the SOC trial in
                                // addition to Armijo f-descent.
                                if(s.filter.is_acceptable(f_soc, h_soc)
                                   && f_soc <= f_k + c1 * alpha_soc * grad_f_dot_p
                                   && h_soc <= s.filter.h_max())
                                    soc_accepted = true;
                            }
                            else
                            {
                                if(s.filter.is_acceptable(f_soc, h_soc))
                                    soc_accepted = true;
                            }

                            if(soc_accepted)
                            {
                                p = p_soc;
                                alpha = alpha_soc;
                                f_trial = f_soc;
                                h_trial = h_soc;
                                accepted = true;
                                break;
                            }

                            alpha_soc *= rho;
                        }
                    }
                }
            }
        }

        // Feasibility restoration: recover from filter-blocked iterates.
        // Restoration runs FIRST (before any BFGS-reset retry) per the
        // restoration-first ordering documented at the top of this loop:
        // restoration explicitly minimises ||c|| and is conceptually
        // cheaper than discarding curvature. The BFGS-reset retry layered
        // at the bottom of the loop fires only if restoration also fails
        // to recover an acceptable trial point.
        //
        // Reference: Wachter & Biegler 2006 Section 3.
        if(!accepted)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    auto restoration_result = run_restoration_(s);
                    if(restoration_result.success)
                    {
                        // Capture x before the restoration assignment so
                        // step_size below reports the actual primal step
                        // norm rather than zero.
                        const double rest_step_norm =
                            (restoration_result.x - s.x).norm();
                        s.x = restoration_result.x;
                        s.objective_value = s.problem->value(s.x);
                        s.problem->gradient(s.x, s.g);

                        s.problem->constraints(s.x, s.bufs.c_all);
                        s.c_eq = s.bufs.c_all.head(s.n_eq);
                        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
                        s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

                        const double h_rest_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                        s.filter.add(s.objective_value, h_rest_l1);

                        ++s.iteration;
                        return step_result<double>{
                            .objective_value = s.objective_value,
                            .gradient_norm = lagrangian_gradient_norm(s),
                            .step_size = rest_step_norm,
                            .objective_change = s.objective_value - f_k,
                            .improved = s.objective_value < f_k,
                            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                            .x_norm = s.x.norm(),
                            .diagnostics = {
                                .bfgs_reset_count = reset_count,
                                .bfgs_skip_count = 0,
                                .nan_eval_count = nan_eval_count,
                                .soc_retry_count = soc_retry_count,
                            },
                        };
                    }

                    // Restoration also failed: fall through to the
                    // BFGS-reset retry block at the bottom of the loop.
                }
            }
        }

        }  // end if(!zero_step_restoration_failed) wrapper

        if(accepted)
            break;

        // Line search / filter / SOC / restoration all failed to find an
        // acceptable trial point. Reset BFGS to identity (Shanno-rescale
        // on next push; dense_ldl_bfgs.h:86-105 zeroes
        // updates_since_reset_) and retry the QP with B = I. On
        // exhaustion of the cap, return a null-step with
        // diagnostics.bfgs_reset_count populated. Reaches this block on
        // zero-step restoration failure, main-path restoration failure,
        // or unconstrained / m == 0 fall-through where no restoration
        // was applicable.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset retry parity);
        //            N&W 2e Section 3.3 (recovery from non-descent);
        //            dense_ldl_bfgs.h:86-105 (reset semantics).
        if(reset_count >= reset_max)
        {
            ++s.iteration;
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .policy_status = solver_status::stalled,
                .diagnostics = {
                    .bfgs_reset_count = reset_count,
                    .bfgs_skip_count = 0,
                    .nan_eval_count = nan_eval_count,
                    .soc_retry_count = soc_retry_count,
                },
            };
        }
        s.hessian.reset();
        ++reset_count;
        }  // end BFGS-reset retry loop

        // Accept step.
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
                //
                // Adopted from: argmin/solver/kraft_slsqp_policy.h (in-tree precedent).
                s.bufs.J_all_old = s.bufs.J_all;

                s.problem->constraints(s.x, s.bufs.c_all);
                s.c_eq = s.bufs.c_all.head(s.n_eq);
                s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

                s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
            }
        }

        // Multiplier update. Use the QP subproblem multipliers when the
        // solver returned a full lambda vector; the QP solver returns the
        // KKT multipliers of the SQP subproblem in the canonical
        // [lambda_eq; mu_ineq] layout, which agree with the LS estimate
        // (N&W eq. 18.15) at the optimum. Fall back to the LS estimate
        // only when the QP multipliers are unavailable (rare failure
        // path; allocation there is acceptable and out of the hot path).
        if constexpr(constrained<P>)
        {
            const int m_lambda = s.n_eq + s.n_ineq;
            if(m_lambda > 0)
            {
                if(qp_res.lambda.size() >= m_lambda)
                {
                    s.lambda.head(m_lambda) = qp_res.lambda.head(m_lambda);
                }
                else if(qp_res.lambda.size() == 0)
                {
                    Eigen::Matrix<double, Eigen::Dynamic, N> A_all(m_lambda, n);
                    if(s.n_eq > 0)   A_all.topRows(s.n_eq)    = s.J_eq;
                    if(s.n_ineq > 0) A_all.bottomRows(s.n_ineq) = s.J_ineq;
                    s.lambda = detail::estimate_multipliers(s.g, A_all);
                }
            }
        }

        // BFGS update using Lagrangian gradient difference.
        // Reference: Kraft 1988 Section 2.2.3; N&W eq. 18.13.
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
                    // Eq-only fallback: the helper's m_total branch only
                    // handles the full-multiplier case; the partial-
                    // multiplier path stays inline because it touches
                    // s.J_eq (policy-private state) rather than
                    // J_all_old.topRows.
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

        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
            s.hessian.push(sk, yk);

        ++s.iteration;

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency
        // with step_result.kkt_residual per N&W 2e Definition 12.1.
        // Filter set internals (filter.add / is_acceptable elsewhere in
        // this policy) retain L1 per Fletcher-Leyffer 2002 Section 2.
        double h_new = 0.0;
        if constexpr(constrained<P>)
            h_new = detail::primal_feasibility_inf(s.c_eq, s.c_ineq);

        // KKT residual: full first-order optimality error E(x, lambda,
        // mu) using least-squares multiplier estimates (N&W eq. 18.15).
        // filter_slsqp uses L-BFGS with no explicit multiplier update
        // from the QP, so s.lambda is the only multiplier source.
        // Active-set multiplier re-estimation at x_{k+1} for the kkt leg:
        // detects binding inequalities (|c_ineq[i]| < 1e-8), restricts
        // the LS to the equality + active rows, and clips mu_ineq to
        // >= 0. Plain LS + cwiseMax breaks on optima with parallel
        // inequality gradients (HS024: row 2 = -row 3 of J_ineq at x*)
        // because the min-norm split between parallel rows is not
        // KKT-valid after sign projection.
        //
        // Reference: N&W 2e Definition 12.1 (KKT conditions:
        //            stationarity, primal feasibility, dual feasibility,
        //            complementarity); eq. 12.34 (Lagrangian
        //            stationarity leg); eq. 18.15 (multiplier LS);
        //            Section 18.3 + Algorithm 18.3 (working-set
        //            identification).
        //
        // Strided re-estimation: when options.multiplier_reest_every_k
        // > 1 the helper is invoked only on steps where
        // (s.iteration % k == 0); on the skip path the prior step's
        // s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf are reused
        // (state-resident, zero-alloc; Bertsekas 1996 §4.2). The
        // helper only writes binding entries, so the buffers are
        // explicitly zeroed inside the gate-fire branch to clear any
        // previously-binding multipliers whose row may have left the
        // active set since the prior fire. At k=1 the gate always
        // fires; behavior is bit-identical to unconditional
        // re-estimation.
        // Clamp the user-supplied stride to >= 1 at the read site.
        // A value of 0 maps to 1 (re-estimate every step) rather than
        // triggering integer-divide-by-zero (SIGFPE on x86-64). The
        // default-constructed value is non-zero (1 accurate / 5 fast),
        // so this clamp only engages when a caller explicitly assigns 0.
        const std::size_t reest_stride = std::max<std::size_t>(
            options.multiplier_reest_every_k, std::size_t{1});
        if(s.iteration % reest_stride == 0)
        {
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
            // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
            argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                              Eigen::Dynamic,
                                                              Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq, s.c_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
        }
        // else: reuse s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf from
        //       the prior step's gate-fire path.
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = lagrangian_gradient_norm(s),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = h_new,
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .diagnostics = {
                .bfgs_reset_count = reset_count,
                .bfgs_skip_count = 0,
                .nan_eval_count = nan_eval_count,
                .soc_retry_count = soc_retry_count,
            },
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        double h_0 = 0.0;
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

                h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
            }
        }
        // Clear then re-seed the filter at the NEW x0. A warm reset must not
        // inherit the previous run's h_max ceiling: a high-violation restart
        // that kept a too-tight ceiling would reject its own first step. Seed
        // exactly as init() / reset_clear() -- h_max = 1e4 * max(1, h_0) with
        // the configured envelope margins.
        //
        // Reference: Wachter and Biegler 2006 eq. (8), Section 2.3.
        s.filter.clear();
        s.filter.initialize(1e4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f, options.gamma_h);
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        reset(s, x0);
        s.hessian.reset();
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

    // Adapter wrapping a argmin problem type for the restoration
    // functions, which expect eval_constraints(x, c_eq, c_ineq) and
    // eval_constraint_jacobians(x, J_eq, J_ineq) methods.
    template <typename P>
    struct restoration_adapter
    {
        const P& problem;
        int n_eq;
        int n_ineq;

        void eval_constraints(const Eigen::Vector<double, N>& x,
                              Eigen::VectorXd& c_eq,
                              Eigen::VectorXd& c_ineq) const
        {
            Eigen::VectorXd c_all(n_eq + n_ineq);
            problem.constraints(x, c_all);
            c_eq = c_all.head(n_eq);
            c_ineq = c_all.tail(n_ineq);
        }

        void eval_constraint_jacobians(const Eigen::Vector<double, N>& x,
                                       Eigen::MatrixXd& J_eq,
                                       Eigen::MatrixXd& J_ineq) const
        {
            Eigen::MatrixXd J_all(n_eq + n_ineq, x.size());
            problem.constraint_jacobian(x, J_all);
            J_eq = J_all.topRows(n_eq);
            J_ineq = J_all.bottomRows(n_ineq);
        }
    };

    // Run the configured restoration strategy.
    //
    // Reference: Wachter & Biegler 2006 Section 3.
    template <typename P>
    detail::restoration_result<double, N> run_restoration_(state_type<P>& s)
    {
        restoration_adapter<P> adapter{*s.problem, s.n_eq, s.n_ineq};

        // sigma for L1 restoration: max(|lambda|) + 1e-4
        // Reference: N&W eq. 18.36 (same formula as kraft_slsqp update_penalty).
        double sigma_restore = 1e4;
        if(s.lambda.size() > 0)
            sigma_restore = s.lambda.cwiseAbs().maxCoeff() + 1e-4;

        auto strategy = options.restoration;
        auto max_steps = options.max_restoration_steps;

        if(strategy == detail::restoration_strategy::l1_penalty ||
           strategy == detail::restoration_strategy::hybrid)
        {
            auto result = detail::restore_l1<double, N>(
                adapter, s.x, s.g, s.lower, s.upper,
                sigma_restore, max_steps);
            if(result.success)
                return result;
        }

        if(strategy == detail::restoration_strategy::feasibility_qp ||
           strategy == detail::restoration_strategy::hybrid)
        {
            return detail::restore_feasibility_qp<double, N>(
                adapter, s.qp_solver, s.x, s.lower, s.upper, max_steps);
        }

        return {s.x, detail::constraint_violation(s.c_eq, s.c_ineq), false, 0};
    }
};

// Single-mode shape: filter_slsqp_policy_accurate is retained as a
// source-call-site readable alias for the bare template, matching
// tr_sqp_policy_accurate on the surviving dual-mode trust-region policy.
template <int N = dynamic_dimension>
using filter_slsqp_policy_accurate = filter_slsqp_policy<N>;

}

#endif
