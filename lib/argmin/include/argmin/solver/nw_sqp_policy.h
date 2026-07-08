#ifndef HPP_GUARD_ARGMIN_SOLVER_NW_SQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_NW_SQP_POLICY_H

// N&W Chapter 18 Sequential Quadratic Programming (SQP) policy.
//
// Implements the line-search SQP method (N&W Algorithm 18.3 / Section 18.6):
//   1. Solve QP subproblem for step p and multipliers lambda (eq. 18.12)
//   2. Update penalty sigma for L1 merit descent (eq. 18.36)
//   3. Backtracking line search on L1 merit function (Section 18.5)
//   4. Damped BFGS update of Hessian of Lagrangian (Procedure 18.2)
//
// The policy satisfies the step_budget_solver contract with plain member functions
// (init/step, no explicit-object parameter).
// Requires: differentiable<Problem> && constrained<Problem>.
// Box bounds: detected via if constexpr(bound_constrained<Problem>).
//
// Reference: N&W Chapter 18, Sections 18.1-18.6.
//            N&W Algorithm 18.1 (local SQP), extended with merit function
//            globalization per Section 18.5-18.6.

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/dense_ldl_bfgs.h"
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
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace argmin
{

// argmin variant: single-mode line-search SQP policy (N&W lineage). The
//                 earlier per-mode dispatch (fast / accurate as a closed-set
//                 NTTP) is removed here: empirical bench measurements showed
//                 `fast` lost wall-time AND iteration count against
//                 `accurate` on every measured cell across the entire
//                 line-search SQP family. The strictly-worse mode has been
//                 removed; the defaults below are the former `accurate`
//                 values.
//
// Reference: N&W 2e Algorithm 18.3 (line-search SQP framework).
template <int N = dynamic_dimension>
struct nw_sqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = nw_sqp_policy<M>;

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

    // BFGS Hessian-reset retry cap on line-search exhaustion. Matches
    // NLopt slsqp.c's ireset semantics with up to 5 retries.
    //
    // Reference: NLopt slsqp.c:1890-1895 (ireset retry pattern).
    static constexpr std::size_t default_bfgs_reset_max = 5;

    // QP inner solver iteration cap (NLopt slsqp.c LSI/LSEI cascade
    // convention).
    //
    // Reference: NLopt slsqp.c:700-1100 (LSI/LSEI cap convention).
    static constexpr std::uint16_t default_qp_max_iterations = 200;

    // QP convergence tolerance (NLopt's double-precision baseline).
    //
    // Reference: Kraft 1988 §3.5 (QP convergence acc^2 threshold).
    static constexpr double default_qp_tolerance = 1e-12;

    // L1-merit penalty parameter ceiling. The bump_sigma_for_descent
    // helper caps sigma at this value to prevent unbounded penalty growth
    // on problems with degenerate constraint Jacobians (NLopt-parity
    // headroom).
    //
    // Reference: N&W 2e §18.3 + Procedure 18.2 (L1-merit penalty cap).
    static constexpr double default_sigma_max = 1e10;

    // Active-set Lagrange-multiplier re-estimation stride. The post-step
    // KKT-leg invokes compute_kkt_multipliers_active_set only when
    // (s.iteration % multiplier_reest_every_k == 0); on the skip path
    // the prior step's s.bufs.kkt_lambda_eq_buf / s.bufs.kkt_mu_ineq_buf
    // are reused (state-resident, zero-alloc). k=1 recomputes every
    // step (the previous unconditional re-estimation behavior); k>1
    // trades a stationarity-leg lag of ~k steps for k-1x savings on the
    // per-step ColPivHouseholderQR run by the helper.
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

        // QP subproblem params. Brace-initialized from the policy's
        // static-constexpr defaults so options.qp.{max_iterations,
        // tolerance} reflect the shipped policy contract.
        qp_options qp{
            .max_iterations = default_qp_max_iterations,
            .tolerance = default_qp_tolerance,
        };
        std::uint16_t stall_window{50};

        // BFGS-reset retry cap on line-search exhaustion. On Armijo
        // failure (after SOC retry), the policy resets the BFGS Hessian
        // to identity and re-solves the QP, repeating up to
        // bfgs_reset_max times before returning a null-step.
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
        // J_eq / J_ineq match the QP solver's expected
        // Matrix<Scalar, Meq, N> shape so the policy can pass them
        // directly into qp_solver.solve without an A_eq/A_ineq local
        // copy (the prior pattern existed because s.J_eq was MatrixXd
        // — fully dynamic cols — and would not deduce to Matrix<,, N>
        // on the templated solve call when N is a compile-time bound).
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;

        // Cross-policy state-resident buffer struct. Consolidates the
        // per-step / per-line-search trial buffers, the BFGS curvature-pair
        // buffers, the constraint-axis workspaces (c_all, c_trial,
        // b_eq / b_ineq, lam / lam_eq / lam_ineq, kkt_lambda_eq /
        // kkt_mu_ineq), and the constraint Jacobian pair (J_all and the
        // J_all_old cached copy that supports the BFGS curvature-pair on
        // the post-step Jacobian re-evaluation path). nw_sqp does not use
        // the SOC-retry b_eq_soc_buf / b_ineq_soc_buf or the kraft-only
        // pre-factored Hessian E_buf / f_buf members; those remain present
        // (zero-sized after resize on the n_eq / n_ineq dynamic axes) for
        // cross-policy struct consistency.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers
        //               (in-tree precedent: generalizes the per-policy
        //                *_buf state-resident layout).
        // Reference: N&W 2e Section 18.
        //
        // argmin variant: cross-policy state-resident buffer struct.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        Eigen::MatrixXd AAt_workspace;      // Feasibility LDLT workspace
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;  // Reusable factorization
        Eigen::VectorXd w_workspace;        // LDLT solve in-place buffer
        double objective_value{};
        double sigma{1.0};
        // Canonical BFGS operator. References: Shanno 1978 (N&W eq. 6.20); N&W Section 7.2 (L-BFGS); Kraft 1988 DFVLR-FB 88-28 Section 2.2.3.
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::active_set_qp_solver<double, N> qp_solver;
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
                      "nw_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "nw_sqp_policy requires constrained<Problem>");

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

        // Cross-policy state-resident buffer struct: sizes all per-step
        // workspaces (n-sized iterate / direction / curvature-pair buffers,
        // m-sized constraint workspaces, and the constraint Jacobian pair)
        // in one call. nw_sqp does not consume the kraft-only E_buf / f_buf
        // pre-factored Hessian members; those remain default-resized.
        s.bufs.resize(n, s.n_eq, s.n_ineq);
        s.bufs.J_all_old.setZero(m, n);

        if(s.n_eq > 0)
        {
            s.AAt_workspace.resize(s.n_eq, s.n_eq);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(s.n_eq);
            s.w_workspace.resize(s.n_eq);
        }

        // Evaluate constraints: single vector, split into eq/ineq
        if(m > 0)
            problem.constraints(x0, s.bufs.c_all);
        s.c_eq = s.bufs.c_all.head(s.n_eq);
        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

        // Evaluate Jacobian: single matrix, split into eq/ineq
        if(m > 0)
            problem.constraint_jacobian(x0, s.bufs.J_all);
        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

        // Box bounds
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

        // BFGS Hessian of Lagrangian, initialized to I
        s.hessian = detail::dense_ldl_bfgs<double, N>(n);
        // Pre-allocate QP solver workspace: equality + inequality + 2n box bounds
        int max_constraints = s.n_eq + s.n_ineq + 2 * n;
        s.qp_solver = detail::active_set_qp_solver<double, N>(n, max_constraints);
        s.sigma = 1.0;
        s.iteration = 0;

        // Initial multiplier estimate via least-squares (N&W eq. 18.15)
        if(m > 0)
        {
            Eigen::Matrix<double, Eigen::Dynamic, N> A_all(m, n);
            if(s.n_eq > 0) A_all.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_all.bottomRows(s.n_ineq) = s.J_ineq;
            s.lambda = detail::estimate_multipliers(s.g, A_all);
        }
        else
        {
            s.lambda = Eigen::VectorXd::Zero(0);
        }

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;

        // BFGS-reset retry loop on line-search exhaustion (NLopt
        // slsqp.c:1890-1895 ireset parity). Wraps QP solve -> cold-start
        // sigma -> zero-step null-step -> sigma update -> Armijo line
        // search -> Maratos SOC retry. On line-search exhaustion (Armijo
        // and SOC both fail to find a decreasing step), reset the BFGS
        // Hessian to identity and re-solve the QP, bounded by
        // options.bfgs_reset_max. On cap exhaustion, return a null-step
        // with diagnostics.bfgs_reset_count populated.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h::step retry-loop
        //               pattern (in-tree precedent).
        //               NLopt slsqp.c:1890-1895 (ireset retry pattern,
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
        // Fast-mode BFGS-update skip counter. Increments on the policy-level
        // hessian.push() guard whenever the curvature pair has s^T y <= 0
        // and the policy mode dispatches to the skip branch (the Powell-
        // damped helper path is bypassed). Always zero in accurate mode.
        std::size_t bfgs_skip_count = 0;
        // Armijo NaN/Inf recovery counter aggregated across the main inline
        // line search and the SOC retry inline loop. nw_sqp hand-rolls the
        // Armijo backtracking loop (does not call line_search/armijo.h)
        // because the inline-merit pattern avoids a per-backtrack VectorXd
        // materialisation; this counter mirrors armijo()'s diagnostic on
        // the hand-rolled paths. Incremented inline when a trial-iterate
        // evaluation returns non-finite f_trial or constraint values; the
        // backtracker shrinks alpha and continues. Both modes enable the
        // gate.
        //
        // NaN/Inf gate: see argmin/line_search/armijo.h header comment.
        std::size_t nan_eval_count = 0;
        // Second-order correction retry counter. Increments once per SOC
        // retry attempt at the full-step rejection site (regardless of
        // whether the retry succeeds); surfaced through the contracted
        // step_result diagnostic of the same name (trust-region SQP
        // precedent).
        //
        // Reference: N&W 2e Section 18.3 (Maratos effect, second-order
        //            correction).
        std::size_t soc_retry_count = 0;

        // Loop-carried variables that survive the retry block into the
        // post-loop accept-step block. The cold-start sigma calibration
        // is gated on s.iteration == 0 AND the per-step cold_start_done
        // flag; subsequent retries within the same step() call do NOT
        // advance s.iteration, so without the flag the cold-start would
        // re-fire on each retry (max-monotone and therefore idempotent
        // but wasteful and obscures intent). detail::update_penalty
        // below is similarly monotone-up.
        detail::qp_result<double, N> qp;
        Eigen::Vector<double, N>& p = s.bufs.p_buf;
        Eigen::VectorXd& lambda_new = s.bufs.lam_buf;
        lambda_new.setZero(m);
        double phi0 = 0.0;
        double dphi0 = 0.0;
        double alpha = 1.0;
        Eigen::Vector<double, N>& x_trial = s.bufs.x_trial_buf;
        x_trial = s.x;
        double f_trial = s.objective_value;
        bool ls_success = false;
        // iter-0 cold start: fires at most once per outer step()
        // invocation; gates the calibrate_initial_penalty call against
        // re-firing on every BFGS-reset retry within the same step().
        bool cold_start_done = false;

        for(;;)
        {

        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        // Pass s.J_eq / s.J_ineq directly into the QP solver — the
        // state matrices already match the Matrix<Scalar, Meq, N> shape
        // the templated solve() expects, so no local A_eq/A_ineq copy
        // is needed.
        s.bufs.b_eq_workspace = -s.c_eq;
        s.bufs.b_ineq_workspace = -s.c_ineq;

        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Adopted from: argmin/detail/sqp_common.h equality_feasibility_warmstart
            //               (in-tree precedent — helper hoists this very pattern from nw_sqp).
            // Reference: N&W 2e Section 18.3 (equality-feasibility step).
            //
            // argmin variant: caller-owned AAt + LDLT workspaces; nw_sqp is the
            //                 canonical extraction site for this helper.
            argmin::detail::equality_feasibility_warmstart<double, N, Eigen::Dynamic>(
                s.J_eq, s.bufs.b_eq_workspace,
                s.AAt_workspace, s.ldlt_feasibility, s.w_workspace, p0);
        }

        // Embedded QP options pass through directly: qp_options is plain-type
        // and brace-initialized with the accurate-mode fallback literals at
        // the struct level (200, 1e-12). Per-policy per-mode wiring happens
        // at options_type instantiation time, no read-site indirection.
        const argmin::qp_options& qp_opts = options.qp;

        bool has_finite_bounds = has_finite_box(s.lower, s.upper);

        // Inequality-feasible warm start for the main QP. At an
        // inequality-infeasible iterate the violated rows have
        // b_ineq = -c_ineq > 0, so the zero / equality-projected p0
        // violates them and active_set_qp_solver (whose phase-1
        // restores equality feasibility only) freezes at the warm
        // start: blocking_step_length clamps alpha to 0 against the
        // violated rows and the policy stalls at an infeasible
        // stationary-looking point. At feasible iterates b_ineq <= 0,
        // the pre-check inside the helper passes, and the projection
        // is a no-op — the healthy path is untouched.
        //
        // Adopted from: argmin/detail/sqp_common.h soc_seed_projection.
        if(has_finite_bounds)
        {
            s.bufs.p_lo_buf.noalias() = s.lower - s.x;
            s.bufs.p_hi_buf.noalias() = s.upper - s.x;
            p0 = p0.cwiseMax(s.bufs.p_lo_buf).cwiseMin(s.bufs.p_hi_buf);
            detail::soc_seed_projection<double, N>(
                s.J_ineq, s.bufs.b_ineq_workspace,
                s.bufs.p_lo_buf, s.bufs.p_hi_buf,
                /*use_bounds=*/true, s.bufs, p0);
            s.qp_solver.solve_into(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace, s.J_ineq, s.bufs.b_ineq_workspace,
                                   s.bufs.p_lo_buf, s.bufs.p_hi_buf, p0, qp_opts, qp);
        }
        else
        {
            detail::soc_seed_projection<double, N>(
                s.J_ineq, s.bufs.b_ineq_workspace,
                s.bufs.p_lo_buf, s.bufs.p_hi_buf,
                /*use_bounds=*/false, s.bufs, p0);
            s.qp_solver.solve_into(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace, s.J_ineq, s.bufs.b_ineq_workspace,
                                   p0, qp_opts, qp);
        }

        p = qp.x;

        // Cold-start mu calibration. After the first QP solve, sigma must
        // dominate the multiplier scale to make the SQP direction a descent
        // direction for the L1 merit (N&W eq. 18.36). Default sigma = 1.0
        // underestimates the required penalty on HS071-class problems with
        // large multipliers, causing iter-0 line-search rejection.
        //
        // Adopted from: NLopt slsqp.c iter-0 implicit cold-start from QP lambda.
        // Reference: N&W 2e eq. 18.36; Kraft 1988 §2.2.6.
        //            iter-0 cold-start: qp-lambda-driven sigma seed bounds
        //            the first-step rejection rate by ensuring sigma
        //            dominates the multiplier scale at the start of the
        //            run.
        //
        // argmin variant: gated on s.iteration == 0 AND the per-step
        //                 cold_start_done flag declared above the retry
        //                 loop. iter-0 cold start fires at most once per
        //                 outer step() invocation; the existing
        //                 detail::update_penalty call below is
        //                 max-monotone and therefore idempotent.
        if(!cold_start_done && s.iteration == 0 && qp.lambda.size() > 0)
        {
            s.sigma = detail::calibrate_initial_penalty(s.sigma, qp.lambda);
            cold_start_done = true;
        }

        // Zero step check.
        //
        // Null step: QP returned a zero direction (degeneracy or
        // active-set cycling). N&W 2e S18.4.
        //
        // is_null_step=true exempts this iterate from step_tolerance
        // stall detection so step_budget_solver gives the policy another
        // iteration to break the degeneracy rather than flagging
        // solver_status::stalled on iter 2.
        //
        // The QP still returns multipliers at a zero-direction
        // solution, so the KKT residual on this iterate is a
        // faithful stationarity measure even though no step was
        // taken. Reporting it lets objective_tolerance_criterion
        // terminate when the iterate is a true KKT point (zero
        // direction is the correct QP answer at an optimum) rather
        // than running out max_iterations on a converged problem.
        //
        // Null-step branch: x_{k+1} = x_k so QP multipliers are
        // measured at the current iterate; no staleness. Multiplier
        // re-estimation not applied here.
        if(p.norm() < 1e-15)
        {
            // Adopted from: argmin/detail/sqp_common.h extract_qp_multipliers.
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if(m > 0 && qp.lambda.size() > 0)
            {
                argmin::detail::extract_qp_multipliers<double>(
                    qp.lambda, s.n_eq, s.n_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
            }

            // Adopted from: argmin/detail/sqp_common.h null_step_result.
            // Reference: N&W 2e Definition 12.1, eq. 12.34 (full KKT
            //            first-order optimality E-measure).
            //
            // argmin variant: nw_sqp's null-step path overrides the helper's
            //                 gradient_norm (which uses lpNorm<Infinity>) with
            //                 the L2 Lagrangian-gradient norm computed against
            //                 the CURRENT QP multipliers. Pre-refactor in-line
            //                 code used `grad_L.norm()` (L2); the override
            //                 preserves bit-identical reporting on the
            //                 cross-policy step_result consistency suite.
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), reset_count);
            // L2 override: helper reports lpNorm<Infinity>, nw_sqp's prior
            // in-line code reported L2.
            double grad_L_null_norm = s.g.norm();
            if(m > 0)
            {
                Eigen::Vector<double, N> grad_L = s.g;
                if(s.n_eq > 0)
                    grad_L.noalias() -= s.J_eq.transpose() * s.bufs.kkt_lambda_eq_buf;
                if(s.n_ineq > 0)
                    grad_L.noalias() -= s.J_ineq.transpose() * s.bufs.kkt_mu_ineq_buf;
                grad_L_null_norm = grad_L.norm();
            }
            r.gradient_norm = grad_L_null_norm;
            // Mirror the cap-exhausted exit below: surface the
            // accumulated NaN-recovery and BFGS-skip counts so the
            // in-loop null-step return is symmetric with the post-loop
            // null-step return (the field is a contracted public
            // diagnostic and the consistency suite asserts on it).
            r.diagnostics.nan_eval_count = nan_eval_count;
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            r.diagnostics.soc_retry_count = soc_retry_count;
            return r;
        }

        // --- 2. Extract QP multipliers ---
        lambda_new.setZero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // --- 3. Update penalty parameter (N&W eq. 18.36) ---
        s.sigma = detail::update_penalty(s.sigma, lambda_new);

        // --- 4. Line search on L1 merit ---
        phi0 = detail::l1_merit(s.objective_value,
                                s.c_eq, s.c_ineq, s.sigma);
        // Adopted from: argmin/detail/merit_function.h l1_merit_dphi_h4.
        // nw_sqp's lineage uses h4 = 1 (no augmented relaxation), so the
        // helper is bit-identical to the prior l1_merit_directional_derivative
        // call.
        const double grad_f_dot_p = s.g.dot(p);
        dphi0 = argmin::detail::l1_merit_dphi_h4<double>(
            grad_f_dot_p, s.c_eq, s.c_ineq, s.sigma, /*h4=*/1.0);

        // If dphi0 >= 0, the penalty is insufficient for descent.
        // Increase sigma until the SQP direction is a descent direction
        // for the L1 merit function (N&W eq. 18.36).
        //
        // sigma_saturated records whether the bump hit the sigma_max cap and
        // left the merit slope non-negative; on saturation the merit line
        // search below is bypassed entirely (bounded-work short-circuit into
        // the BFGS-reset retry / null-step recovery ladder).
        bool sigma_saturated = false;
        if(dphi0 >= 0.0)
        {
            double cv = detail::constraint_violation(s.c_eq, s.c_ineq);
            if(cv > 1e-15)
            {
                // Adopted from: argmin/detail/merit_function.h bump_sigma_for_descent.
                //
                // argmin variant: helper guards on h4 > 0 in addition to violation > 0;
                //                 nw_sqp's prior in-line bump used grad_f_dot_p / cv + 1
                //                 directly; helper's `abs(grad_f_dot_p) / (violation * h4) + 1`
                //                 with h4 = 1 reduces to abs(grad_f_dot_p) / cv + 1 — bit-
                //                 identical for grad_f_dot_p >= 0 (the only branch reached
                //                 here, since dphi0 >= 0 implies grad_f_dot_p >= sigma * cv > 0).
                const auto sigma_bump = argmin::detail::bump_sigma_for_descent<double>(
                    s.sigma, grad_f_dot_p, cv, /*h4=*/1.0,
                    /*sigma_max=*/default_sigma_max);
                s.sigma = sigma_bump.sigma;
                sigma_saturated = sigma_bump.saturated;
                phi0 = detail::l1_merit(s.objective_value,
                                        s.c_eq, s.c_ineq, s.sigma);
                dphi0 = grad_f_dot_p - s.sigma * cv;
            }
            else
            {
                dphi0 = grad_f_dot_p;
                if(dphi0 >= 0.0)
                    dphi0 = -1e-8;
            }
        }

        // Backtracking Armijo on L1 merit using embedded line search options
        alpha = 1.0;
        const double ls_c1 = options.line_search.c1;
        const double ls_rho = options.line_search.rho;
        const std::uint16_t max_ls = options.line_search.max_iterations;

        // Initialize x_trial from s.x so that the post-loop read at
        // s.x = x_trial; is well-defined even in the pathological
        // max_ls == 0 configuration. GCC 15 flags an uninitialized SSE
        // packet load through this path if x_trial is left default-
        // constructed after only being assigned inside the loop body.
        x_trial = s.x;
        f_trial = s.objective_value;
        ls_success = false;

        // On sigma saturation the loop guard is false from the first
        // iteration: the doomed merit search is skipped, ls_success stays
        // false, and control falls straight through to the recovery ladder.
        for(std::uint16_t ls = 0; ls < max_ls && !sigma_saturated; ++ls)
        {
            x_trial = s.x + alpha * p;

            if(has_finite_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            f_trial = s.problem->value(x_trial);
            if(m > 0)
                s.problem->constraints(x_trial, s.bufs.c_all);

            // NaN/Inf gate: see argmin/line_search/armijo.h header comment.
            //               Hand-rolled inline-merit path mirrors that
            //               semantics — non-finite f_trial or constraints
            //               trigger backtrack rather than crossing the L1
            //               merit comparison. Both modes enable the gate.
            if(!std::isfinite(f_trial)
               || (m > 0 && !s.bufs.c_all.allFinite()))
            {
                ++nan_eval_count;
                alpha *= ls_rho;
                continue;
            }

            // Inline L1 merit via head/tail VectorBlock expressions to avoid
            // a per-backtrack VectorXd materialization. Equivalent to
            // detail::l1_merit(f, c_eq, c_ineq, sigma).
            double viol_trial = 0.0;
            if(s.n_eq > 0)
                viol_trial += s.bufs.c_all.head(s.n_eq).cwiseAbs().sum();
            if(s.n_ineq > 0)
                viol_trial += (-s.bufs.c_all.tail(s.n_ineq)).cwiseMax(0.0).sum();
            double phi_trial = f_trial + s.sigma * viol_trial;

            if(phi_trial <= phi0 + ls_c1 * alpha * dphi0)
            {
                // Persist the accepted trial constraints for the post-loop
                // accept block (replaces the prior eager s.c_eq_trial /
                // s.c_ineq_trial assignment that ran on every backtrack).
                s.bufs.c_trial_buf = s.bufs.c_all;
                ls_success = true;
                break;
            }

            alpha *= ls_rho;
        }

        // Maratos second-order correction retry on rejection of the full
        // step.
        //
        // Near a tight active set the linearised constraints understate
        // quadratic curvature, so the L1 merit at the full unit step is
        // lifted by the second-order constraint residual even when p is a
        // valid first-order descent direction (the Maratos effect). The
        // trigger is the filter line-search SOC convention: attempt the
        // correction when the full step was rejected (line-search failure
        // or a backtracked acceptance) AND failed to reduce the constraint
        // violation, theta(x + p) >= theta(x_k). Re-solve the QP with the
        // constraint RHS re-anchored at the trial point:
        //   b_eq_soc  = -c_eq(x + p)  + J_eq(x)  * p
        //   b_ineq_soc = -c_ineq(x + p) + J_ineq(x) * p
        // and accept the corrected direction p_soc if the Armijo test
        // passes on the L1 merit at a step length exceeding the plain
        // backtracked one.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h::step SOC retry
        //               block (in-tree precedent).
        //
        // argmin variant: nw_sqp uses active_set_qp_solver with the dense
        //                 Hessian B = s.hessian.hessian(); kraft_slsqp uses
        //                 kraft_lsq_qp_recovery_solver with a factored
        //                 (E, f) form. The SOC math is identical; the QP
        //                 API differs.
        //
        // Reference: Wachter & Biegler 2006 Section 2.4 (SOC trigger on
        //            rejection of the first trial step with
        //            theta(x + p) >= theta(x_k));
        //            Kraft 1988 DFVLR-FB 88-28 §2.2.4 (Maratos SOC);
        //            N&W 2e §18.3 (Maratos effect, second-order correction).
        const double cv_now = detail::constraint_violation(s.c_eq, s.c_ineq);

        // theta(x + p): L1 constraint violation at the full unit step,
        // evaluated only when the full step was rejected. Leaves c(x + p)
        // in s.bufs.c_all for the SOC RHS below. Non-finite constraint
        // values keep h_full NaN so the trigger comparison is false and
        // the retry is skipped.
        double h_full = std::numeric_limits<double>::quiet_NaN();
        if((!ls_success || alpha < 1.0) && m > 0 && !sigma_saturated)
        {
            Eigen::Vector<double, N> x_full = s.x + p;
            if(has_finite_bounds)
                x_full = detail::project(x_full, s.lower, s.upper);

            s.problem->constraints(x_full, s.bufs.c_all);
            if(s.bufs.c_all.allFinite())
            {
                h_full = 0.0;
                if(s.n_eq > 0)
                    h_full += s.bufs.c_all.head(s.n_eq).cwiseAbs().sum();
                if(s.n_ineq > 0)
                    h_full += (-s.bufs.c_all.tail(s.n_ineq))
                                  .cwiseMax(0.0).sum();
            }
        }

        if(h_full >= cv_now && m > 0)
        {
            ++soc_retry_count;

            auto c_eq_full = s.bufs.c_all.head(s.n_eq);
            auto c_ineq_full = s.bufs.c_all.tail(s.n_ineq);

            if(s.n_eq > 0)
                s.bufs.b_eq_soc_buf.noalias() = -c_eq_full + s.J_eq * p;
            if(s.n_ineq > 0)
                s.bufs.b_ineq_soc_buf.noalias() = -c_ineq_full + s.J_ineq * p;

            // Inequality-feasible SOC warm start. p is a stationary
            // point of the unchanged QP objective, and the QP solver's
            // phase-1 restores equality feasibility only, so seeding
            // the re-solve at an inequality-infeasible p makes the
            // working-set loop terminate at p without consulting the
            // corrected RHS. Project p onto the corrected inequality
            // polyhedron (min-norm LDP shift; bound rows stacked in the
            // bounded branch) before the solve; equality-only problems
            // (n_ineq == 0) are untouched inside the helper.
            //
            // Adopted from: argmin/detail/sqp_common.h
            //               soc_seed_projection.
            detail::qp_result<double, N> qp_soc;
            if(has_finite_bounds)
            {
                // p_lo_buf / p_hi_buf already populated from the main QP
                // build above; reuse without re-computing s.lower - s.x.
                Eigen::Vector<double, N> p0_soc = p;
                p0_soc = p0_soc.cwiseMax(s.bufs.p_lo_buf).cwiseMin(s.bufs.p_hi_buf);
                detail::soc_seed_projection<double, N>(
                    s.J_ineq, s.bufs.b_ineq_soc_buf,
                    s.bufs.p_lo_buf, s.bufs.p_hi_buf,
                    /*use_bounds=*/true, s.bufs, p0_soc);
                s.qp_solver.solve_into(
                    s.hessian.hessian(), s.g,
                    s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                    s.bufs.p_lo_buf, s.bufs.p_hi_buf, p0_soc, qp_opts, qp_soc);
            }
            else
            {
                Eigen::Vector<double, N> p0_soc = p;
                detail::soc_seed_projection<double, N>(
                    s.J_ineq, s.bufs.b_ineq_soc_buf,
                    s.bufs.p_lo_buf, s.bufs.p_hi_buf,
                    /*use_bounds=*/false, s.bufs, p0_soc);
                s.qp_solver.solve_into(
                    s.hessian.hessian(), s.g,
                    s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                    p0_soc, qp_opts, qp_soc);
            }

            if(qp_soc.status == detail::qp_status::optimal)
            {
                // The re-solved QP already yields the full corrected
                // direction from the current iterate: its constraint RHS
                // -c(x + p) + J * p models the linearization error of the
                // whole step, so the corrected trial is x + p_soc, NOT
                // x + p + p_soc — with the replacement-form RHS, adding p
                // again double-counts the step and lands at
                // x + 2p + O(||p||^2), which re-lifts the violation the
                // correction was meant to remove.
                const Eigen::Vector<double, N>& p_soc = qp_soc.x;
                double alpha_soc = 1.0;
                Eigen::Vector<double, N> x_trial_soc = s.x;
                double f_trial_soc{s.objective_value};

                for(std::uint16_t ls = 0; ls < max_ls; ++ls)
                {
                    // A corrected step supersedes an already-accepted
                    // backtracked step only at a strictly larger step
                    // length (kraft_slsqp SOC precedence rule); once
                    // alpha_soc falls to the accepted alpha the retry
                    // cannot improve on the main line search.
                    if(ls_success && alpha_soc <= alpha)
                        break;

                    x_trial_soc = s.x + alpha_soc * p_soc;

                    if(has_finite_bounds)
                        x_trial_soc = detail::project(x_trial_soc,
                                                      s.lower, s.upper);

                    f_trial_soc = s.problem->value(x_trial_soc);
                    if(m > 0)
                        s.problem->constraints(x_trial_soc, s.bufs.c_all);

                    // NaN/Inf recovery on the SOC retry path (parallel to
                    // the main-LS inline gate above). Shrink alpha_soc and
                    // continue on non-finite trial-iterate evaluation.
                    if(!std::isfinite(f_trial_soc)
                       || (m > 0 && !s.bufs.c_all.allFinite()))
                    {
                        ++nan_eval_count;
                        alpha_soc *= ls_rho;
                        continue;
                    }

                    // Inline L1 merit via head/tail VectorBlock expressions to
                    // avoid per-backtrack VectorXd materialization (see main LS).
                    double viol_soc = 0.0;
                    if(s.n_eq > 0)
                        viol_soc += s.bufs.c_all.head(s.n_eq).cwiseAbs().sum();
                    if(s.n_ineq > 0)
                        viol_soc += (-s.bufs.c_all.tail(s.n_ineq)).cwiseMax(0.0).sum();
                    double phi_trial_soc = f_trial_soc + s.sigma * viol_soc;

                    if(phi_trial_soc <= phi0 + ls_c1 * alpha_soc * dphi0)
                    {
                        p = p_soc;
                        alpha = alpha_soc;
                        x_trial = x_trial_soc;
                        f_trial = f_trial_soc;
                        // Persist accepted trial constraints (parallel to the
                        // main line-search acceptance branch).
                        s.bufs.c_trial_buf = s.bufs.c_all;
                        ls_success = true;
                        break;
                    }

                    alpha_soc *= ls_rho;
                }
            }
        }

        if(ls_success)
            break;

        // Line search and SOC retry both failed to find a decreasing
        // step. Reset BFGS to identity (Shanno-rescale on next push;
        // dense_ldl_bfgs.h:86-105 zeroes updates_since_reset_) and
        // retry the QP with B = I. On exhaustion of the cap, return a
        // null-step with diagnostics.bfgs_reset_count populated.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset retry parity);
        //            N&W 2e Section 3.3 (recovery from non-descent);
        //            dense_ldl_bfgs.h:86-105 (reset semantics).
        if(reset_count >= reset_max)
        {
            // Adopted from: argmin/detail/sqp_common.h extract_qp_multipliers.
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if(m > 0 && qp.lambda.size() >= m)
            {
                argmin::detail::extract_qp_multipliers<double>(
                    qp.lambda, s.n_eq, s.n_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
            }

            // Adopted from: argmin/detail/sqp_common.h null_step_result.
            //
            // argmin variant: nw_sqp overrides the helper's gradient_norm
            //                 (lpNorm<Infinity>) with the L2 Lagrangian-
            //                 gradient norm to preserve cross-policy
            //                 step_result consistency.
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), reset_count);
            double grad_L_cap_norm = s.g.norm();
            if(m > 0)
            {
                Eigen::Vector<double, N> grad_L = s.g;
                if(s.n_eq > 0)
                    grad_L.noalias() -= s.J_eq.transpose() * s.bufs.kkt_lambda_eq_buf;
                if(s.n_ineq > 0)
                    grad_L.noalias() -= s.J_ineq.transpose() * s.bufs.kkt_mu_ineq_buf;
                grad_L_cap_norm = grad_L.norm();
            }
            r.gradient_norm = grad_L_cap_norm;
            // Surface the Armijo NaN-recovery count accumulated across
            // the main hand-rolled LS and the SOC retry within this
            // step() call (null_step_result leaves nan_eval_count at
            // its in-class default 0).
            r.diagnostics.bfgs_skip_count = bfgs_skip_count;
            r.diagnostics.nan_eval_count = nan_eval_count;
            r.diagnostics.soc_retry_count = soc_retry_count;
            return r;
        }

        s.hessian.reset();
        ++reset_count;
        }  // end BFGS-reset retry loop

        // --- 5. Compute BFGS update vectors ---
        // Reuse objective and constraint values from line search (already
        // evaluated at x_trial). Only gradient and Jacobian need fresh eval.
        s.bufs.x_old_buf = s.x;
        s.bufs.g_old_buf = s.g;
        Eigen::Vector<double, N>& x_old = s.bufs.x_old_buf;
        Eigen::Vector<double, N>& g_old = s.bufs.g_old_buf;
        double f_old = s.objective_value;

        s.x = x_trial;
        s.objective_value = f_trial;
        s.problem->gradient(s.x, s.g);
        s.c_eq = s.bufs.c_trial_buf.head(s.n_eq);
        s.c_ineq = s.bufs.c_trial_buf.tail(s.n_ineq);

        // Save old Jacobian before re-evaluation for BFGS update (N&W eq. 18.13)
        if(m > 0)
            s.bufs.J_all_old.noalias() = s.bufs.J_all;

        if(m > 0)
            s.problem->constraint_jacobian(s.x, s.bufs.J_all);
        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

        Eigen::Vector<double, N>& grad_L_new = s.bufs.grad_L_new_buf;
        Eigen::Vector<double, N>& grad_L_old = s.bufs.grad_L_old_buf;
        Eigen::Vector<double, N>& sk = s.bufs.sk_buf;
        Eigen::Vector<double, N>& yk = s.bufs.yk_buf;

        // Lagrangian gradient at new and old points using new multipliers.
        // grad_L_old must use the Jacobian at x_old (not x_new) so that
        // y_k = grad_L_new - grad_L_old captures constraint curvature.
        // Reference: N&W eq. 18.13.
        if(m > 0)
        {
            // Adopted from: argmin/detail/sqp_common.h compute_bfgs_pair_fused
            //               (in-tree precedent).
            //
            // argmin variant: nw_sqp's lambda_new is the full m-sized QP-multiplier
            //                 vector (lam_buf alias); the helper's m_total branch
            //                 runs a single fused J_all^T * lam GEMV per gradient
            //                 against topRows(m) of J_all_old / J_all.
            argmin::detail::compute_bfgs_pair_fused<double, N>(
                g_old, s.g, s.bufs.J_all_old, s.bufs.J_all,
                lambda_new.head(m), m,
                grad_L_old, grad_L_new,
                sk, yk,
                s.x, x_old);
        }
        else
        {
            grad_L_new = s.g;
            grad_L_old = g_old;
            sk.noalias() = s.x - x_old;
            yk.noalias() = grad_L_new - grad_L_old;
        }

        // --- 6. BFGS curvature push (dense_ldl_bfgs, Powell-damped) ---
        // Skip BFGS updates on near-zero or non-positive curvature
        // pairs. In SQP the Lagrangian gradient difference y_k can
        // easily have s^T y < 0 when the constraint Hessian contribution
        // (A_{k+1} - A_k)^T lam dominates the objective curvature; the
        // dense_ldl_bfgs::push call below applies Powell damping per
        // N&W eq. 18.22-18.24 internally, but the explicit guard here
        // keeps the policy-step semantics legible.
        //
        // On strictly-positive curvature push the BFGS curvature pair
        // (Powell damping applied inside dense_ldl_bfgs::push per N&W eq.
        // 18.22-18.24); on non-positive curvature drop the pair (the
        // earlier fast-mode skip-and-count path was removed after
        // empirical evidence showed it lost wall-time and iteration count
        // against the Powell-damped path on every measured cell).
        //
        // The bfgs_skip_count diagnostic field on step_result is retained
        // for cross-policy schema parity but is now never incremented
        // here; the local counter feeds the existing post-step
        // diagnostics propagation untouched.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard;
        //            N&W eq. 18.22-18.24 (Powell damping).
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15)
        {
            if(sTy > 0.0)
                s.hessian.push(sk, yk);
        }

        // --- 7. Update multipliers and iteration ---
        s.lambda = lambda_new;
        ++s.iteration;

        double grad_L_norm = grad_L_new.norm();

        double phi_new = detail::l1_merit(s.objective_value,
                                            s.c_eq, s.c_ineq, s.sigma);

        // Active-set multiplier re-estimation at x_{k+1}. The QP-derived
        // lambda_new satisfies the linearised KKT at x_k, not at x_{k+1}
        // where grad_f and the constraint Jacobian have moved; reusing
        // lambda_new at the new iterate produces a stationarity leg that
        // oscillates with the remaining problem curvature. Active-set LS
        // detects binding inequalities (|c_ineq[i]| < 1e-8) and solves the
        // reduced LS problem on just the equality + active rows, with
        // mu_ineq clamped to >= 0 for dual feasibility. Plain LS +
        // cwiseMax fails on optima with parallel inequality gradients
        // (HS024: row 2 = -row 3 of J_ineq at x*) because the min-norm
        // split between the parallel rows is not KKT-valid after sign
        // projection. lambda_reest is local to this kkt evaluation;
        // BFGS curvature-pair construction above at :419-423 continues
        // to use lambda_new (QP multipliers) so inter-iteration BFGS
        // semantics are unchanged.
        //
        // Reference: N&W 2e Section 18.3 and Algorithm 18.3 (working-set
        //            identification);
        //            eq. 18.15 (least-squares lambda);
        //            Definition 12.1 (KKT conditions: stationarity +
        //            dual feasibility + complementarity);
        //            eq. 12.34 (Lagrangian stationarity leg).
        //
        // Strided re-estimation: when options.multiplier_reest_every_k
        // > 1 the helper is invoked only on steps where
        // (s.iteration % k == 0); on the skip path the prior step's
        // s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf are reused
        // (state-resident, zero-alloc; Bertsekas 1996 §4.2). The
        // .setZero() pair MUST live inside the gate-fire branch so
        // the skip path reuses prior values rather than zeroing them.
        // At k=1 the gate always fires; behavior is bit-identical to
        // unconditional re-estimation.
        if(m > 0)
        {
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
                    s.g, s.J_eq, s.J_ineq, s.c_ineq, s.bufs.kkt_ws,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
            }
            // else: reuse s.bufs.kkt_lambda_eq_buf / kkt_mu_ineq_buf
            //       from the prior step's gate-fire path (or from
            //       init() which leaves them zero — equivalent to "no
            //       active set yet identified" at iter 0).
        }
        else
        {
            // Unconstrained branch: KKT residual collapses to ||grad_f||;
            // zero the buffers so the kkt_residual helper sees a clean
            // multiplier vector regardless of prior state.
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency
        // with step_result.kkt_residual. The L1-merit local `cv` above
        // keeps L1 per N&W eq. 15.24 (merit function semantics).
        //
        // Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_L_norm,
            .step_size = sk.norm(),
            .objective_change = s.objective_value - f_old,
            .improved = phi_new < phi0,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .diagnostics = {
                .bfgs_reset_count = reset_count,
                .bfgs_skip_count = bfgs_skip_count,
                .nan_eval_count = nan_eval_count,
                .soc_retry_count = soc_retry_count,
            },
        };
    }

    // Hot start -- preserves BFGS Hessian.
    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        const int n = static_cast<int>(x0.size());
        const int m = s.n_eq + s.n_ineq;
        // Evaluate at s.x (a concrete Eigen::Vector) rather than the Ref x0 so
        // a dynamic-N warm reset does not materialize a heap temporary when the
        // problem method takes a plain vector.
        s.x = x0;
        s.objective_value = s.problem->value(s.x);
        s.problem->gradient(s.x, s.g);
        if(m > 0)
            s.problem->constraints(s.x, s.bufs.c_all);
        s.c_eq = s.bufs.c_all.head(s.n_eq);
        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
        if(m > 0)
            s.problem->constraint_jacobian(s.x, s.bufs.J_all);
        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS Hessian.
    template <typename P>
    void reset_clear(state_type<P>& s,
                     Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        reset(s, x0);
        s.hessian.reset();
        s.sigma = 1.0;
        s.lambda.setZero();
    }

private:
    static bool has_finite_box(const Eigen::Vector<double, N>& lower,
                               const Eigen::Vector<double, N>& upper)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int i = 0; i < lower.size(); ++i)
        {
            if(lower[i] > -inf || upper[i] < inf)
                return true;
        }
        return false;
    }
};

// Single-mode shape: nw_sqp_policy_accurate is retained as a
// source-call-site readable alias for the bare template, matching
// tr_sqp_policy_accurate on the surviving dual-mode trust-region policy.
template <int N = dynamic_dimension>
using nw_sqp_policy_accurate = nw_sqp_policy<N>;

}

#endif
