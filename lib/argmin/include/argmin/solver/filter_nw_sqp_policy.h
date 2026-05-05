#ifndef HPP_GUARD_ARGMIN_SOLVER_FILTER_NW_SQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_FILTER_NW_SQP_POLICY_H

// Dense BFGS SQP with Fletcher-Leyffer 2002 filter acceptance.
//
// Mirrors nw_sqp_policy in QP solver (active_set_qp_solver) and Hessian
// (dense_ldl_bfgs, Powell-damped packed L D L^T BFGS per Fletcher-Powell
// 1974; Shanno 1978 initial-Hessian rescale on the first push only;
// Kraft 1988 DFVLR-FB 88-28 Section 2.2.3) but replaces user-tuned L1
// merit globalization with a filter-based acceptance test. The filter maintains a set of non-dominated (f, h) pairs.
// Trial points must be filter-acceptable AND satisfy an Armijo
// condition on an automatically-derived L1 merit. The penalty sigma
// is computed from QP multipliers (no user tuning).
//
// The switching condition distinguishes f-type iterations (near-feasible,
// Armijo on objective suffices) from h-type iterations (infeasible, aim
// to reduce constraint violation via filter acceptance).
//
// Reference: Fletcher & Leyffer 2002, "Nonlinear programming without a
//            penalty function", Math. Program. 91:239-269;
//            Wachter & Biegler 2006, "On the implementation of an
//            interior-point filter line-search algorithm", Section 2.3;
//            N&W Section 15.5 (filter methods);
//            N&W Chapter 18 (dense BFGS SQP).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/merit_function.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/filter_acceptance.h"
#include "argmin/detail/filter_restoration.h"
#include "argmin/options/qp_options.h"
#include "argmin/line_search/options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace argmin
{

template <int N = dynamic_dimension>
struct filter_nw_sqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = filter_nw_sqp_policy<M>;

    struct options_type
    {
        line_search_options line_search{};
        detail::restoration_strategy restoration{detail::restoration_strategy::hybrid};
        std::uint16_t max_restoration_steps{10};
        double soc_violation_threshold{1e-8};
        std::uint16_t stall_window{50};

        // Filter envelope margins (Wachter & Biegler 2006 Section 2.3,
        // eq. 6). Default 1e-5 / 1e-5 preserve v0.2.1 behaviour; tuning
        // is per-policy and selected empirically by the v0.3.0 envelope
        // sweep.
        //
        // Reference: Wachter & Biegler 2006 Section 2.3;
        //            Fletcher & Leyffer 2002 Section 5.
        std::optional<double> gamma_f{};
        std::optional<double> gamma_h{};

        // BFGS-reset retry cap on line-search/filter exhaustion. After
        // restoration fails to recover an acceptable trial point, the policy
        // resets the BFGS Hessian to identity and re-solves the QP, repeating
        // up to bfgs_reset_max times before returning a null-step. Default 5
        // matches NLopt slsqp.c:1890-1895 ireset semantics.
        //
        // Reference: PITFALLS section L (line-search exhaustion fallback);
        //            NLopt slsqp.c:1890-1895 (ireset retry pattern);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: filter policies layer this AFTER the existing
        //                 restoration path (restoration first then BFGS
        //                 reset then null-step). Cascade-free: no new
        //                 solver_status enum entry.
        std::size_t bfgs_reset_max{5};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        // J_eq / J_ineq match the QP solver's expected
        // Matrix<Scalar, Meq, N> shape so the policy can pass them
        // directly into qp_solver.solve without a local A_eq/A_ineq
        // copy. Compile-time-N column count restores parity with
        // nw_sqp_policy.
        //
        // Adopted from: argmin/solver/nw_sqp_policy.h:107-108 (in-tree
        //               precedent — typed J_eq / J_ineq for templated
        //               qp_solver.solve dispatch).
        // Reference: N&W 2e Section 18.3.
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;

        // Cross-policy state-resident buffer struct. Consolidates the
        // per-step / per-line-search trial buffers, the BFGS curvature-pair
        // buffers, the constraint-axis workspaces, and the constraint
        // Jacobian pair. filter_nw_sqp uses the SOC-retry b_eq_soc_buf /
        // b_ineq_soc_buf for the Maratos correction; the kraft-only
        // pre-factored Hessian E_buf / f_buf members remain present (zero-
        // sized after resize) for cross-policy struct consistency.
        //
        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers
        //               (in-tree precedent — generalizes the per-policy
        //                *_buf state-resident layout).
        // Reference: N&W 2e Section 18.
        //
        // argmin variant: cross-policy state-resident buffer struct.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        // Adopted from: argmin/solver/nw_sqp_policy.h:88-96, :222-232
        //               (in-tree precedent — stateful active_set_qp_solver
        //                with reusable LDLT warm-start workspace).
        // Reference: N&W 2e Section 16.5 (active-set QP).
        //
        // argmin variant: pre-allocated QP solver + LDLT factorization
        //                 workspace for the equality-feasibility warm-start;
        //                 closes the per-step ColPivHouseholderQR allocation
        //                 site that filter_nw_sqp's prior solve_qp free-
        //                 function path carried.
        detail::active_set_qp_solver<double, N> qp_solver;
        Eigen::MatrixXd AAt_workspace;
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;

        double objective_value{};
        // Shanno-rescaled compact L-BFGS operator (N&W eq. 6.20 /
        // Section 7.2; Kraft 1988 DFVLR-FB 88-28 Section 2.2.3).
        // Adaptive theta rebuilds B from the stored history with a
        // fresh theta = y^T y / s^T y on every push, tracking local
        // Lagrangian curvature rather than freezing at iteration 0.
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::filter_set<double> filter;

        // Automatically-derived penalty for L1 merit acceptance.
        // Unlike nw_sqp_policy, sigma is NOT exposed to users. It is
        // computed from QP multipliers each step (N&W eq. 18.36).
        double sigma{1.0};

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
        static_assert(differentiable<Problem>,
                      "filter_nw_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "filter_nw_sqp_policy requires constrained<Problem>");

        state_type<Problem> s;
        s.problem = &problem;
        s.n = problem.dimension();
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int m = s.n_eq + s.n_ineq;

        s.x = x0;
        s.g.setZero(s.n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // Adopted from: argmin/detail/sqp_common.h sqp_state_buffers::resize.
        s.bufs.resize(s.n, s.n_eq, s.n_ineq);
        s.bufs.J_all_old.setZero(m, s.n);

        if(s.n_eq > 0)
        {
            s.AAt_workspace.resize(s.n_eq, s.n_eq);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(s.n_eq);
        }

        if(m > 0)
            problem.constraints(x0, s.bufs.c_all);
        s.c_eq = s.bufs.c_all.head(s.n_eq);
        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);

        if(m > 0)
            problem.constraint_jacobian(x0, s.bufs.J_all);
        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(s.n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(s.n, inf);
        }

        s.hessian = detail::dense_ldl_bfgs<double, N>(s.n);
        // Pre-allocate QP solver workspace: equality + inequality + 2n box bounds.
        const int max_constraints = s.n_eq + s.n_ineq + 2 * s.n;
        s.qp_solver = detail::active_set_qp_solver<double, N>(s.n, max_constraints);
        s.sigma = 1.0;
        s.iteration = 0;

        // Initialize filter with h_max based on initial constraint
        // violation, then thread the configured envelope margins onto
        // the filter (Wachter & Biegler 2006 Section 2.3, eq. 6).
        // Defaults 1e-5 / 1e-5 preserve v0.2.1 behaviour when the
        // options are unset.
        // Reference: Wachter & Biegler 2006, eq. (8).
        double h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.filter.initialize(1e4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f.value_or(1e-5),
                              options.gamma_h.value_or(1e-5));

        // Initial multiplier estimate via least-squares (N&W eq. 18.15).
        if(m > 0)
        {
            Eigen::Matrix<double, Eigen::Dynamic, N> A_all(m, s.n);
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
        const int n = s.n;
        const int m = s.n_eq + s.n_ineq;

        // Invariant: s.objective_value, s.g, s.c_eq/c_ineq, s.J_eq/J_ineq
        // are fresh at s.x. Init() seeds them at x_0; each step()'s
        // bottom block (or the restoration branch) updates them at the
        // accepted iterate before returning. basic_solver does not
        // mutate state.x between policy.step() calls, so the redundant
        // top-of-step re-evaluation that this branch used to perform is
        // pure overhead (one wasted gradient + Jacobian + constraints
        // call per outer iter from iter 1 onward).

        // BFGS-reset retry layered after the restoration path.
        //
        // Same restoration-first ordering as filter_slsqp: feasibility
        // restoration runs first on rejection (it explicitly minimises
        // ||c|| and is conceptually cheaper than discarding curvature).
        // The BFGS-reset retry fires only when restoration also fails to
        // recover an acceptable trial point. Ordering: line search /
        // SOC / restoration first then BFGS-reset retry then null-step.
        // Preserves v0.2.1 restoration semantics while adding the
        // BFGS-reset escape hatch.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h retry-loop pattern;
        //               argmin/solver/filter_slsqp_policy.h restoration-first
        //               ordering (in-tree precedents landed alongside this fix).
        //               NLopt slsqp.c:1890-1895 (ireset retry pattern, max=5).
        // Reference: PITFALLS section L (line-search exhaustion fallback);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: cap is options.bfgs_reset_max (default 5);
        //                 status on exhaustion is is_null_step + the
        //                 diagnostics.bfgs_reset_count counter; rationale
        //                 is cascade-free (no new solver_status enum entry).
        std::size_t reset_count = 0;
        const std::size_t reset_max = options.bfgs_reset_max;

        // Loop-carried variables that survive the retry block into the
        // post-loop accept-step block. The only path that exits the loop
        // via `break` is the LS / SOC accept path (accepted = true).
        // Restoration-success and restoration-exhaustion (after reset cap)
        // return inline from inside the loop.
        const bool has_bounds = has_finite_box(s.lower, s.upper);
        qp_options qp_opts;
        qp_opts.max_iterations = std::uint16_t{200};
        qp_opts.tolerance = 1e-12;

        detail::qp_result<double, N> qp;
        Eigen::Vector<double, N>& p = s.bufs.p_buf;
        Eigen::VectorXd& lambda_new = s.bufs.lam_buf;
        lambda_new.setZero(m);
        Eigen::Vector<double, N>& x_trial = s.bufs.x_trial_buf;
        x_trial = s.x;
        Eigen::VectorXd c_eq_trial, c_ineq_trial;
        double f_trial = s.objective_value;
        double f_k = s.objective_value;
        double h_k = 0.0;
        double phi0 = 0.0;
        double alpha = 1.0;

        for(;;)
        {
        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        // Re-bind QP RHS each retry: s.c_eq / s.c_ineq are unchanged
        // (s.x has not moved) but the Hessian has been reset on retry,
        // so the QP direction differs. Linearisation matrices s.J_eq /
        // s.J_ineq are likewise unchanged across retries; pass them
        // directly into qp_solver.solve (REF-06 typing change drops
        // the prior local A_eq / A_ineq copies).
        s.bufs.b_eq_workspace = -s.c_eq;
        s.bufs.b_ineq_workspace = -s.c_ineq;

        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Adopted from: argmin/detail/sqp_common.h equality_feasibility_warmstart
            //               (in-tree precedent — helper hoists nw_sqp's LDLT pattern).
            // Reference: N&W 2e Section 18.3 (equality-feasibility step).
            //
            // argmin variant: caller-owned AAt + LDLT workspaces; replaces the
            //                 per-step ColPivHouseholderQR allocation.
            argmin::detail::equality_feasibility_warmstart<double, N, Eigen::Dynamic>(
                s.J_eq, s.bufs.b_eq_workspace,
                s.AAt_workspace, s.ldlt_feasibility, p0);
        }

        if(has_bounds)
        {
            s.bufs.p_lo_buf.noalias() = s.lower - s.x;
            s.bufs.p_hi_buf.noalias() = s.upper - s.x;
            p0 = p0.cwiseMax(s.bufs.p_lo_buf).cwiseMin(s.bufs.p_hi_buf);
            // Adopted from: argmin/solver/nw_sqp_policy.h:335-337
            //               (in-tree precedent — stateful active_set_qp_solver::solve).
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace,
                                   s.J_ineq, s.bufs.b_ineq_workspace,
                                   s.bufs.p_lo_buf, s.bufs.p_hi_buf, p0, qp_opts);
        }
        else
        {
            // Adopted from: argmin/solver/nw_sqp_policy.h:341-343
            //               (in-tree precedent — stateful active_set_qp_solver::solve, no-bounds overload).
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace,
                                   s.J_ineq, s.bufs.b_ineq_workspace,
                                   p0, qp_opts);
        }

        p = qp.x;

        // Cold-start mu calibration. After the first QP solve, sigma must
        // dominate the multiplier scale to make the SQP direction a descent
        // direction for the L1 merit (N&W eq. 18.36). Default sigma = 1.0
        // underestimates required penalty on HS071-class problems with
        // large multipliers, causing iter-0 line-search rejection.
        //
        // Adopted from: NLopt slsqp.c iter-0 implicit cold-start from QP lambda.
        // Reference: N&W 2e eq. 18.36; Kraft 1988 §2.2.6;
        //            PITFALLS §B remedy 1.
        //
        // argmin variant: gated on s.iteration == 0; the existing
        //                 detail::update_penalty call below is monotone-up
        //                 and idempotent against this cold-start. Re-running
        //                 on retry (s.iteration is unchanged across retries)
        //                 is harmless: update_penalty is monotone-up and
        //                 idempotent.
        if(s.iteration == 0 && qp.lambda.size() > 0)
        {
            s.sigma = detail::calibrate_initial_penalty(s.sigma, qp.lambda);
        }

        if(p.norm() < 1e-15)
        {
            // Null step: QP returned a zero direction (degeneracy or
            // active-set cycling). N&W 2e S18.4.
            //
            // is_null_step=true exempts this iterate from
            // step_tolerance stall detection so basic_solver gives
            // the policy another iteration to break the degeneracy
            // rather than flagging solver_status::stalled on iter 2.
            //
            // Dual convention: step_result.constraint_violation reports
            // L-infinity primal feasibility (dimensionally consistent with
            // kkt_residual per N&W 2e Definition 12.1); filter_set entries
            // elsewhere in this policy retain L1 h_k per Fletcher-Leyffer
            // 2002 Section 2 filter dominance ordering.
            //
            // Null-step branch: x_{k+1} = x_k so the active-set LS
            // re-estimate is measured at the current iterate with no
            // staleness. Populating kkt_residual here lets the outer
            // convergence check terminate on a true KKT point even when
            // the QP returns zero at the optimum (otherwise the composite
            // gate falls back to gradient_norm = ||grad_f||_inf which is
            // not zero at constrained optima and blocks termination).

            // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if(m > 0)
            {
                argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                                  Eigen::Dynamic,
                                                                  Eigen::Dynamic>(
                    s.g, s.J_eq, s.J_ineq, s.c_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
            }

            // Adopted from: argmin/detail/sqp_common.h null_step_result.
            // Reference: N&W 2e Definition 12.1, eq. 12.34 (full KKT
            //            first-order optimality E-measure).
            //
            // argmin variant: filter_nw_sqp's null-step path overrides the
            //                 helper's gradient_norm (lpNorm<Infinity>) with
            //                 the L2 lagrangian_gradient_norm(s) computed
            //                 against the previous-step lambda. Pre-refactor
            //                 in-line code used lagrangian_gradient_norm(s);
            //                 the override preserves bit-identical reporting
            //                 on the cross-policy step_result consistency
            //                 suite.
            auto r = argmin::detail::null_step_result<double, N,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic>(
                s.objective_value, s.g, s.J_eq, s.J_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq, s.x.norm(), reset_count);
            r.gradient_norm = lagrangian_gradient_norm(s);
            return r;
        }

        // --- 2. Extract QP multipliers and update penalty ---
        lambda_new.setZero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // Auto-derive penalty sigma from QP multipliers (N&W eq. 18.36).
        // This replaces the user-tunable penalty of nw_sqp_policy.
        // Additionally increase sigma if the directional derivative of the
        // L1 merit is non-negative, which indicates the penalty is too low
        // for the current step to be a descent direction on the merit.
        //
        // Inline bump preserves filter_nw_sqp's `+ 1e-4` constant (vs the
        // detail::bump_sigma_for_descent helper's `+ 1` constant). The
        // helper's signature is sized for nw_sqp / kraft cold-bump
        // semantics where the additive ceiling is `+ 1`; filter_nw_sqp's
        // smaller `+ 1e-4` constant is load-bearing for the HS043 SEED-006
        // over-rejection marker (Catch2 Approx margin 4.0). Adopting the
        // helper here would shift HS043 cv beyond the marker; left inline.
        // Reference: N&W 2e eq. 18.36; PITFALLS §B.
        s.sigma = detail::update_penalty(s.sigma, lambda_new);
        double h_cur = detail::constraint_violation(s.c_eq, s.c_ineq);
        double dphi_check = s.g.dot(p) - s.sigma * h_cur;
        if(dphi_check >= 0.0 && h_cur > 1e-12)
            s.sigma = std::max(s.sigma,
                std::abs(s.g.dot(p)) / h_cur + 1e-4);

        // --- 3. Filter + merit acceptance line search ---
        //
        // The filter prevents cycling (Fletcher & Leyffer 2002).
        // The L1 merit Armijo ensures constraint satisfaction via the
        // automatically derived penalty sigma.
        //
        // Reference: Wachter & Biegler 2006, Section 2.3;
        //            N&W Section 18.5 (L1 merit Armijo).
        f_k = s.objective_value;
        h_k = detail::constraint_violation(s.c_eq, s.c_ineq);
        const double grad_f_dot_p = s.g.dot(p);

        // Check switching condition to determine iteration type.
        const bool f_type = detail::is_f_type_iteration(
            h_k, grad_f_dot_p, s.filter.h_max());

        // Add current point to filter for h-type iterations.
        if(!f_type)
            s.filter.add(f_k, h_k);

        // L1 merit at current point.
        // Adopted from: argmin/detail/merit_function.h l1_merit_dphi_h4.
        // filter_nw_sqp's lineage uses h4 = 1, so the helper is bit-identical
        // to the prior l1_merit_directional_derivative call.
        phi0 = detail::l1_merit(f_k, s.c_eq, s.c_ineq, s.sigma);
        const double dphi0 = argmin::detail::l1_merit_dphi_h4<double>(
            grad_f_dot_p, s.c_eq, s.c_ineq, s.sigma, /*h4=*/1.0);

        alpha = 1.0;
        const double c1 = options.line_search.c1;
        const double rho_shrink = options.line_search.rho;
        const int max_ls = options.line_search.max_iterations;

        f_trial = f_k;
        double h_trial = h_k;
        bool accepted = false;

        for(int ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * p;
            if(has_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                s.problem->constraints(x_trial, s.bufs.c_trial_buf);
                c_eq_trial = s.bufs.c_trial_buf.head(s.n_eq);
                c_ineq_trial = s.bufs.c_trial_buf.tail(s.n_ineq);
            }
            else
            {
                c_eq_trial = Eigen::VectorXd{};
                c_ineq_trial = Eigen::VectorXd{};
            }
            h_trial = detail::constraint_violation(c_eq_trial, c_ineq_trial);

            // Filter acceptance check.
            bool filter_ok = s.filter.is_acceptable(f_trial, h_trial);

            // L1 merit Armijo check.
            double phi_trial = detail::l1_merit(
                f_trial, c_eq_trial, c_ineq_trial, s.sigma);
            bool merit_ok = phi_trial <= phi0 + c1 * alpha * dphi0;

            if(f_type)
            {
                // F-type: require both filter acceptance and merit Armijo.
                if(filter_ok && merit_ok)
                {
                    accepted = true;
                    break;
                }
            }
            else
            {
                // H-type: filter acceptance OR merit Armijo suffices.
                if(filter_ok || merit_ok)
                {
                    accepted = true;
                    break;
                }
            }

            alpha *= rho_shrink;
        }

        // Second-order correction (SOC) if rejected and infeasible.
        if(!accepted && h_k > options.soc_violation_threshold)
        {
            // SOC RHS lives in state-resident b_eq_soc_buf / b_ineq_soc_buf
            // (sqp_state_buffers fields) — replaces the prior per-step
            // b_eq_soc / b_ineq_soc local materializations.
            if(s.n_eq > 0)
                s.bufs.b_eq_soc_buf.noalias() = -c_eq_trial;
            if(s.n_ineq > 0)
                s.bufs.b_ineq_soc_buf.noalias() = -c_ineq_trial;

            detail::qp_result<double, N> qp_soc;
            if(has_bounds)
            {
                Eigen::Vector<double, N> p_lower_soc = s.lower - x_trial;
                Eigen::Vector<double, N> p_upper_soc = s.upper - x_trial;
                Eigen::Vector<double, N> p_soc_0 = Eigen::Vector<double, N>::Zero(n);
                // Adopted from: argmin/solver/nw_sqp_policy.h:581-584
                //               (in-tree precedent — stateful active_set_qp_solver
                //                SOC-retry path).
                qp_soc = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                           s.J_eq, s.bufs.b_eq_soc_buf,
                                           s.J_ineq, s.bufs.b_ineq_soc_buf,
                                           p_lower_soc, p_upper_soc, p_soc_0,
                                           qp_opts);
            }
            else
            {
                Eigen::Vector<double, N> p_soc_0 = Eigen::Vector<double, N>::Zero(n);
                // Adopted from: argmin/solver/nw_sqp_policy.h:588-591
                //               (in-tree precedent — stateful active_set_qp_solver,
                //                no-bounds overload).
                qp_soc = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                           s.J_eq, s.bufs.b_eq_soc_buf,
                                           s.J_ineq, s.bufs.b_ineq_soc_buf,
                                           p_soc_0, qp_opts);
            }

            Eigen::Vector<double, N> x_soc = x_trial + qp_soc.x;
            if(has_bounds)
                x_soc = detail::project(x_soc, s.lower, s.upper);

            double f_soc = s.problem->value(x_soc);
            Eigen::VectorXd c_eq_soc, c_ineq_soc;
            if(m > 0)
            {
                s.problem->constraints(x_soc, s.bufs.c_trial_buf);
                c_eq_soc = s.bufs.c_trial_buf.head(s.n_eq);
                c_ineq_soc = s.bufs.c_trial_buf.tail(s.n_ineq);
            }
            double h_soc = detail::constraint_violation(c_eq_soc, c_ineq_soc);

            double phi_soc = detail::l1_merit(
                f_soc, c_eq_soc, c_ineq_soc, s.sigma);
            if(s.filter.is_acceptable(f_soc, h_soc)
               && phi_soc <= phi0 + c1 * dphi0)
            {
                x_trial = x_soc;
                f_trial = f_soc;
                h_trial = h_soc;
                c_eq_trial = c_eq_soc;
                c_ineq_trial = c_ineq_soc;
                accepted = true;
            }
        }

        // If still rejected, attempt feasibility restoration.
        // Restoration runs FIRST (before any BFGS-reset retry) per the
        // restoration-first ordering documented at the top of this
        // step(): restoration explicitly minimises ||c|| and is
        // conceptually cheaper than discarding curvature. The
        // BFGS-reset retry layered after this block fires only when
        // restoration also fails to recover an acceptable trial point.
        //
        // Reference: Wachter & Biegler 2006, Section 3.
        if(!accepted)
        {
            auto rest = run_restoration_(s);
            if(rest.success)
            {
                // Capture x before the restoration assignment so
                // step_size below reports the actual primal step
                // norm rather than zero.
                const double rest_step_norm = (rest.x - s.x).norm();
                s.x = rest.x;
                s.objective_value = s.problem->value(s.x);
                s.problem->gradient(s.x, s.g);
                if(m > 0)
                {
                    s.problem->constraints(s.x, s.bufs.c_all);
                    s.c_eq = s.bufs.c_all.head(s.n_eq);
                    s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
                    s.problem->constraint_jacobian(s.x, s.bufs.J_all);
                    s.J_eq = s.bufs.J_all.topRows(s.n_eq);
                    s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
                }
                const double h_restored_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                s.filter.add(s.objective_value, h_restored_l1);
                ++s.iteration;

                // Populate kkt_residual on the restoration return so the
                // cross-policy step_result consistency suite has a valid
                // stationarity quantity at the converged optimum (cases
                // where the line search exhausts on micro-steps near the
                // KKT point but restoration yields a no-movement success).
                //
                // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
                s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
                s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
                std::optional<double> kkt_rest;
                if(m > 0)
                {
                    argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                                      Eigen::Dynamic,
                                                                      Eigen::Dynamic>(
                        s.g, s.J_eq, s.J_ineq, s.c_ineq,
                        s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
                    kkt_rest = detail::kkt_residual<double,
                                                    Eigen::Dynamic,
                                                    Eigen::Dynamic,
                                                    Eigen::Dynamic>(
                        s.g, s.J_eq, s.J_ineq,
                        s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                        s.c_eq, s.c_ineq);
                }

                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = lagrangian_gradient_norm(s),
                    .step_size = rest_step_norm,
                    .objective_change = s.objective_value - f_k,
                    .improved = h_restored_l1 < h_k,
                    .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                    .kkt_residual = kkt_rest,
                    .diagnostics = { .bfgs_reset_count = reset_count },
                };
            }

            // Restoration also failed: fall through to the BFGS-reset
            // retry block at the bottom of this loop.
        }

        if(accepted)
            break;

        // Line search / SOC / restoration all failed to find an
        // acceptable trial point. Reset BFGS to identity (Shanno-rescale
        // on next push; dense_ldl_bfgs.h zeroes updates_since_reset_)
        // and retry the QP with B = I. On exhaustion of the cap, return
        // a null-step with diagnostics.bfgs_reset_count populated.
        //
        // is_null_step=true prevents step_tolerance_criterion from
        // labelling the zero step as a stall; the iterate is carried
        // forward without movement while the outer loop retains
        // flexibility (restoration may succeed on a future iteration
        // once the filter envelope relaxes).
        //
        // step_result.constraint_violation reports L-infinity primal
        // feasibility (kkt_residual-consistent). The local h_k (L1)
        // is consumed by filter dominance checks per Fletcher-Leyffer
        // 2002 Section 2.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset retry parity);
        //            N&W 2e Section 3.3 (recovery from non-descent);
        //            dense_ldl_bfgs.h reset() semantics.
        if(reset_count >= reset_max)
        {
            // Populate kkt_residual via active-set LS at the current (non-
            // moved) iterate so the cross-policy step_result consistency
            // suite has a valid stationarity quantity on the cap-exhausted
            // null-step return.
            //
            // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
            s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
            std::optional<double> kkt_cap;
            if(m > 0)
            {
                argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                                  Eigen::Dynamic,
                                                                  Eigen::Dynamic>(
                    s.g, s.J_eq, s.J_ineq, s.c_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
                kkt_cap = detail::kkt_residual<double,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic>(
                    s.g, s.J_eq, s.J_ineq,
                    s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                    s.c_eq, s.c_ineq);
            }

            ++s.iteration;
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .kkt_residual = kkt_cap,
                .diagnostics = { .bfgs_reset_count = reset_count },
            };
        }
        s.hessian.reset();
        ++reset_count;
        }  // end BFGS-reset retry loop

        // --- 4. Accept step and compute BFGS update ---
        s.bufs.x_old_buf = s.x;
        s.bufs.g_old_buf = s.g;
        Eigen::Vector<double, N>& x_old = s.bufs.x_old_buf;
        Eigen::Vector<double, N>& g_old = s.bufs.g_old_buf;
        double f_old = s.objective_value;

        // Save Jacobian at x_k before re-evaluation for BFGS y-vector (N&W eq. 18.13).
        if(m > 0)
            s.bufs.J_all_old.noalias() = s.bufs.J_all;

        s.x = x_trial;
        s.objective_value = f_trial;
        s.c_eq = c_eq_trial;
        s.c_ineq = c_ineq_trial;
        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            s.problem->constraint_jacobian(s.x, s.bufs.J_all);
            s.J_eq = s.bufs.J_all.topRows(s.n_eq);
            s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
        }

        Eigen::Vector<double, N>& sk = s.bufs.sk_buf;
        Eigen::Vector<double, N>& yk = s.bufs.yk_buf;
        Eigen::Vector<double, N>& grad_L_new = s.bufs.grad_L_new_buf;
        Eigen::Vector<double, N>& grad_L_old = s.bufs.grad_L_old_buf;

        // Lagrangian gradient at new and old points for BFGS y-vector.
        // N&W eq. 18.13: grad_L_new uses Jacobian at x_{k+1}, grad_L_old uses Jacobian at x_k.
        if(m > 0)
        {
            // Adopted from: argmin/detail/sqp_common.h compute_bfgs_pair_fused
            //               (in-tree precedent — single fused J^T*lambda GEMV per gradient).
            //
            // argmin variant: filter_nw_sqp's lambda_new is the full m-sized QP-multiplier
            //                 vector (lam_buf alias); the helper's m_total branch runs a
            //                 single fused J_all^T * lam GEMV per gradient against
            //                 topRows(m) of J_all_old / J_all.
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

        // --- 5. BFGS curvature push (dense_ldl_bfgs, Powell-damped) ---
        // Skip BFGS updates on near-zero or non-positive curvature
        // pairs. In filter-SQP the Lagrangian gradient difference y_k
        // can easily have s^T y < 0 when the constraint Hessian
        // contribution (A_{k+1} - A_k)^T lam dominates the objective
        // curvature; the dense_ldl_bfgs::push call below applies
        // Powell damping per N&W eq. 18.22-18.24 internally, but the
        // explicit guard here keeps the policy-step semantics legible.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard.
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
        {
            s.hessian.push(sk, yk);
        }

        // --- 6. Update multipliers and iteration ---
        s.lambda = lambda_new;
        ++s.iteration;

        double phi_new = detail::l1_merit(
            s.objective_value, s.c_eq, s.c_ineq, s.sigma);

        // Active-set multiplier re-estimation at x_{k+1}. The QP-derived
        // lambda_new satisfies the linearised KKT at x_k, not at x_{k+1}
        // where grad_f and the constraint Jacobian have moved; reusing
        // lambda_new at the new iterate produces a stationarity leg that
        // oscillates with the remaining problem curvature. Active-set LS
        // detects binding inequalities (|c_ineq[i]| < 1e-8) and solves
        // the reduced LS problem on just the equality + active rows,
        // with mu_ineq clamped to >= 0 for dual feasibility. Plain LS +
        // cwiseMax fails on optima with parallel inequality gradients
        // (HS024: row 2 = -row 3 of J_ineq at x*) because the min-norm
        // split between the parallel rows is not KKT-valid after sign
        // projection. lambda_reest is local to this kkt evaluation;
        // BFGS curvature-pair construction above at :521-524 uses
        // lambda_new (QP multipliers) so inter-iteration BFGS semantics
        // are unchanged.
        //
        // Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set
        //            identification);
        //            eq. 18.15 (least-squares lambda);
        //            Definition 12.1 (KKT dual feasibility);
        //            eq. 12.34 (Lagrangian stationarity leg).
        s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
        if(m > 0)
        {
            // Adopted from: argmin/detail/lagrangian.h compute_kkt_multipliers_active_set.
            argmin::detail::compute_kkt_multipliers_active_set<double, N,
                                                              Eigen::Dynamic,
                                                              Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq, s.c_ineq,
                s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into step_result for
        // dimensional consistency with kkt_residual per N&W 2e Definition
        // 12.1. The loop-local h_trial (L1) above was consumed by
        // filter.is_acceptable at the line-search step per Fletcher-Leyffer
        // 2002 Section 2 dominance ordering.
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_L_new.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - f_old,
            .improved = phi_new < phi0,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .kkt_residual = kkt,
            .diagnostics = { .bfgs_reset_count = reset_count },
        };
    }

    // Hot start: preserves BFGS Hessian, but clears the filter envelope.
    //
    // The filter is a per-run dominance set: entries added during one
    // solve represent (objective, violation) pairs that were rejected
    // along that trajectory. Carrying them across to a new run on the
    // same state object causes the canonical Wachter-Biegler oscillation
    // -- a converged near-optimum entry from the prior run dominates
    // virtually every trial point of the new run, the filter rejects
    // every line-search step, and the outer loop stalls at alpha -> 0.
    //
    // BFGS Hessian curvature, on the other hand, IS transferable across
    // runs that approach the same local model, so reset() preserves it.
    //
    // Reference: Wachter & Biegler 2006, Section 3.3 (filter
    //            re-initialization between independent runs);
    //            N&W 2e Section 15.4 (filter SQP semantics).
    template <typename Problem>
    void reset(state_type<Problem>& s,
               const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        const int m = s.n_eq + s.n_ineq;
        if(m > 0)
        {
            s.problem->constraints(x0, s.bufs.c_all);
            s.c_eq = s.bufs.c_all.head(s.n_eq);
            s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
            s.problem->constraint_jacobian(x0, s.bufs.J_all);
            s.J_eq = s.bufs.J_all.topRows(s.n_eq);
            s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
        }
        s.filter.clear();
        s.iteration = 0;
    }

    // Cold restart: clears BFGS Hessian, filter envelope (with new
    // h_max derived from the new x0), and penalty.
    template <typename Problem>
    void reset_clear(state_type<Problem>& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        const double h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.filter.initialize(1e4 * std::max(1.0, h_0));
        s.filter.set_envelope(options.gamma_f.value_or(1e-5),
                              options.gamma_h.value_or(1e-5));
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

    template <typename Problem>
    static double lagrangian_gradient_norm(const state_type<Problem>& s)
    {
        const int m = s.n_eq + s.n_ineq;
        if(m == 0)
            return s.g.norm();

        Eigen::Matrix<double, Eigen::Dynamic, N> A(m, s.n);
        if(s.n_eq > 0) A.topRows(s.n_eq) = s.J_eq;
        if(s.n_ineq > 0) A.bottomRows(s.n_ineq) = s.J_ineq;
        return detail::lagrangian_gradient(s.g, A, s.lambda).norm();
    }

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

    // Reference: Wachter & Biegler 2006 Section 3.
    template <typename P>
    detail::restoration_result<double, N> run_restoration_(state_type<P>& s)
    {
        restoration_adapter<P> adapter{*s.problem, s.n_eq, s.n_ineq};

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

        // feasibility QP via solve_qp free function
        if(strategy == detail::restoration_strategy::feasibility_qp ||
           strategy == detail::restoration_strategy::hybrid)
        {
            Eigen::Vector<double, N> x = s.x;
            for(std::uint16_t step = 0; step < max_steps; ++step)
            {
                Eigen::VectorXd c_eq, c_ineq;
                adapter.eval_constraints(x, c_eq, c_ineq);
                double h_k = detail::constraint_violation(c_eq, c_ineq);
                if(h_k < 1e-12)
                    return {x, h_k, true, step};

                Eigen::MatrixXd J_eq_r, J_ineq_r;
                adapter.eval_constraint_jacobians(x, J_eq_r, J_ineq_r);

                auto n = x.size();
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
                Eigen::VectorXd g_zero = Eigen::VectorXd::Zero(n);
                Eigen::VectorXd x0_qp = Eigen::VectorXd::Zero(n);

                Eigen::VectorXd neg_c_eq = (-c_eq).eval();
                Eigen::VectorXd neg_c_ineq = (-c_ineq).eval();
                auto qp_res = detail::solve_qp(H, g_zero,
                    J_eq_r, neg_c_eq, J_ineq_r, neg_c_ineq, x0_qp);

                Eigen::Vector<double, N> p = qp_res.x;
                if(p.norm() < 1e-15)
                    return {x, h_k, false, step};

                double alpha = 1.0;
                while(alpha > 1e-10)
                {
                    Eigen::Vector<double, N> x_trial =
                        (x + alpha * p).cwiseMax(s.lower).cwiseMin(s.upper);
                    Eigen::VectorXd c_eq_t, c_ineq_t;
                    adapter.eval_constraints(x_trial, c_eq_t, c_ineq_t);
                    if(detail::constraint_violation(c_eq_t, c_ineq_t) < h_k)
                    {
                        x = x_trial;
                        break;
                    }
                    alpha *= 0.5;
                }
                if(alpha <= 1e-10)
                    return {x, h_k, false, step};
            }
            Eigen::VectorXd c_eq_f, c_ineq_f;
            adapter.eval_constraints(x, c_eq_f, c_ineq_f);
            double h_f = detail::constraint_violation(c_eq_f, c_ineq_f);
            return {x, h_f, h_f < 1e-12, max_steps};
        }

        return {s.x, detail::constraint_violation(s.c_eq, s.c_ineq), false, 0};
    }
};

}

#endif
