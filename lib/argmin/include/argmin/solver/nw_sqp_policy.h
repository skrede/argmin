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
// The policy satisfies the basic_solver contract via deducing this (D-14).
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
struct nw_sqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = nw_sqp_policy<M>;

    struct options_type
    {
        line_search_options line_search{};  // Replaces hardcoded c1=1e-4, rho=0.5, max_iter=30
        qp_options qp{};                   // QP subproblem params
        // Maratos second-order correction retry threshold. SOC fires when
        // Armijo rejects the trial step AND the constraint violation at x_k
        // exceeds this threshold. Below the threshold the linearization is
        // consistent enough that the SOC step adds work without materially
        // changing the search direction. Default 1e-3 applied at the call
        // site (mirrors kraft_slsqp_policy::options_type::soc_min_violation).
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 §2.2.4 (SOC threshold);
        //            N&W 2e §18.3 (Maratos effect).
        std::optional<double> soc_violation_threshold{};
        std::uint16_t stall_window{50};

        // BFGS-reset retry cap on line-search exhaustion. On Armijo
        // failure (after SOC retry), the policy resets the BFGS Hessian
        // to identity and re-solves the QP, repeating up to
        // bfgs_reset_max times before returning a null-step. Default 5
        // matches NLopt slsqp.c:1890-1895 ireset semantics.
        //
        // Reference: PITFALLS section L (line-search exhaustion fallback);
        //            NLopt slsqp.c:1890-1895 (ireset retry pattern);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: per-policy options field; cascade-free
        //                 (no new solver_status enum entry).
        std::size_t bfgs_reset_max{5};
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
        //               (in-tree precedent — landed alongside this refactor;
        //                generalizes the per-policy *_buf state-resident layout).
        // Reference: N&W 2e Section 18.
        //
        // argmin variant: cross-policy state-resident buffer struct.
        argmin::detail::sqp_state_buffers<double, N> bufs;

        Eigen::MatrixXd AAt_workspace;      // Feasibility LDLT workspace
        Eigen::LDLT<Eigen::MatrixXd> ldlt_feasibility;  // Reusable factorization
        double objective_value{};
        double sigma{1.0};
        // Canonical BFGS operator per D-01. References: Shanno 1978 (N&W eq. 6.20); N&W Section 7.2 (L-BFGS); Kraft 1988 DFVLR-FB 88-28 Section 2.2.3.
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

        constexpr int M = state_type<Problem>::M;

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
        constexpr int M = state_type<P>::M;

        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;

        // BFGS-reset retry loop on line-search exhaustion (D-05; NLopt
        // slsqp.c:1890-1895 ireset parity). Wraps QP solve -> cold-start
        // sigma -> zero-step null-step -> sigma update -> Armijo line
        // search -> Maratos SOC retry. On line-search exhaustion (Armijo
        // and SOC both fail to find a decreasing step), reset the BFGS
        // Hessian to identity and re-solve the QP, bounded by
        // options.bfgs_reset_max. On cap exhaustion, return a null-step
        // with diagnostics.bfgs_reset_count populated.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h retry-loop
        //               pattern (in-tree precedent landed alongside
        //               this fix).
        //               NLopt slsqp.c:1890-1895 (ireset retry pattern,
        //               max=5).
        // Reference: PITFALLS section L (line-search exhaustion fallback);
        //            N&W 2e Section 3.3 (recovery from non-descent).
        //
        // argmin variant: cap is options.bfgs_reset_max (default 5);
        //                 status on exhaustion is is_null_step + the
        //                 diagnostics.bfgs_reset_count counter; rationale
        //                 is D-05 cascade-free (no new solver_status
        //                 enum entry).
        std::size_t reset_count = 0;
        const std::size_t reset_max = options.bfgs_reset_max;

        // Loop-carried variables that survive the retry block into the
        // post-loop accept-step block. The cold-start sigma calibration
        // is gated on s.iteration == 0; subsequent retries within the
        // same step() call do NOT advance s.iteration, so the cold-start
        // re-fires on each retry but is monotone-up (idempotent).
        // detail::update_penalty below is similarly monotone-up.
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
                s.AAt_workspace, s.ldlt_feasibility, p0);
        }

        // Use embedded QP options with defaults
        argmin::qp_options qp_opts;
        qp_opts.max_iterations = options.qp.max_iterations.has_value()
            ? options.qp.max_iterations
            : std::optional<std::uint16_t>{200};
        qp_opts.tolerance = options.qp.tolerance.has_value()
            ? options.qp.tolerance
            : std::optional<double>{1e-12};

        bool has_finite_bounds = has_finite_box(s.lower, s.upper);

        if(has_finite_bounds)
        {
            s.bufs.p_lo_buf.noalias() = s.lower - s.x;
            s.bufs.p_hi_buf.noalias() = s.upper - s.x;
            p0 = p0.cwiseMax(s.bufs.p_lo_buf).cwiseMin(s.bufs.p_hi_buf);
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace, s.J_ineq, s.bufs.b_ineq_workspace,
                                   s.bufs.p_lo_buf, s.bufs.p_hi_buf, p0, qp_opts);
        }
        else
        {
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.bufs.b_eq_workspace, s.J_ineq, s.bufs.b_ineq_workspace,
                                   p0, qp_opts);
        }

        p = qp.x;

        // Cold-start mu calibration. After the first QP solve, sigma must
        // dominate the multiplier scale to make the SQP direction a descent
        // direction for the L1 merit (N&W eq. 18.36). Default sigma = 1.0
        // underestimates the required penalty on HS071-class problems with
        // large multipliers, causing iter-0 line-search rejection.
        //
        // Adopted from: NLopt slsqp.c iter-0 implicit cold-start from QP lambda.
        // Reference: N&W 2e eq. 18.36; Kraft 1988 §2.2.6;
        //            PITFALLS §B remedy 1.
        //
        // argmin variant: gated on s.iteration == 0; the existing
        //                 detail::update_penalty call below is monotone-up
        //                 and idempotent against this cold-start.
        if(s.iteration == 0 && qp.lambda.size() > 0)
        {
            s.sigma = detail::calibrate_initial_penalty(s.sigma, qp.lambda);
        }

        // Zero step check.
        //
        // Null step: QP returned a zero direction (degeneracy or
        // active-set cycling). N&W 2e S18.4.
        //
        // is_null_step=true exempts this iterate from step_tolerance
        // stall detection so basic_solver gives the policy another
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
                s.sigma = argmin::detail::bump_sigma_for_descent<double>(
                    s.sigma, grad_f_dot_p, cv, /*h4=*/1.0, /*sigma_max=*/1e10);
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

        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * p;

            if(has_finite_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            f_trial = s.problem->value(x_trial);
            if(m > 0)
                s.problem->constraints(x_trial, s.bufs.c_all);
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

        // Maratos second-order correction retry on Armijo failure.
        //
        // On rejection of the trial step the L1 merit can be lifted by the
        // nonlinear constraint residual at x + p even when p is a valid
        // first-order descent direction (the Maratos effect: linearised
        // constraints understate quadratic curvature near a tight active
        // set). Re-solve the QP with the constraint RHS re-anchored at the
        // trial point:
        //   b_eq_soc  = -c_eq(x + p)  + J_eq(x)  * p
        //   b_ineq_soc = -c_ineq(x + p) + J_ineq(x) * p
        // and accept the combined direction p + p_soc if the Armijo test
        // passes on the L1 merit.
        //
        // Adopted from: argmin/solver/kraft_slsqp_policy.h:541-617
        // (in-tree precedent).
        //
        // argmin variant: nw_sqp uses active_set_qp_solver with the dense
        //                 Hessian B = s.hessian.hessian(); kraft_slsqp uses
        //                 kraft_lsq_qp_recovery_solver with a factored
        //                 (E, f) form. The SOC math is identical; the QP
        //                 API differs.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 §2.2.4 (Maratos SOC);
        //            N&W 2e §18.3 (Maratos effect, second-order correction).
        const double soc_min_violation =
            options.soc_violation_threshold.value_or(1e-3);
        const double cv_now = detail::constraint_violation(s.c_eq, s.c_ineq);

        if(!ls_success && cv_now > soc_min_violation && m > 0)
        {
            Eigen::Vector<double, N> x_full = s.x + p;
            if(has_finite_bounds)
                x_full = detail::project(x_full, s.lower, s.upper);

            s.problem->constraints(x_full, s.bufs.c_all);
            auto c_eq_full = s.bufs.c_all.head(s.n_eq);
            auto c_ineq_full = s.bufs.c_all.tail(s.n_ineq);

            if(s.n_eq > 0)
                s.bufs.b_eq_soc_buf.noalias() = -c_eq_full + s.J_eq * p;
            if(s.n_ineq > 0)
                s.bufs.b_ineq_soc_buf.noalias() = -c_ineq_full + s.J_ineq * p;

            detail::qp_result<double, N> qp_soc;
            if(has_finite_bounds)
            {
                // p_lo_buf / p_hi_buf already populated from the main QP
                // build above; reuse without re-computing s.lower - s.x.
                Eigen::Vector<double, N> p0_soc = p;
                p0_soc = p0_soc.cwiseMax(s.bufs.p_lo_buf).cwiseMin(s.bufs.p_hi_buf);
                qp_soc = s.qp_solver.solve(
                    s.hessian.hessian(), s.g,
                    s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                    s.bufs.p_lo_buf, s.bufs.p_hi_buf, p0_soc, qp_opts);
            }
            else
            {
                qp_soc = s.qp_solver.solve(
                    s.hessian.hessian(), s.g,
                    s.J_eq, s.bufs.b_eq_soc_buf, s.J_ineq, s.bufs.b_ineq_soc_buf,
                    p, qp_opts);
            }

            if(qp_soc.status == detail::qp_status::optimal)
            {
                Eigen::Vector<double, N> p_combined = p + qp_soc.x;
                double alpha_soc = 1.0;
                Eigen::Vector<double, N> x_trial_soc = s.x;
                double f_trial_soc{s.objective_value};

                for(std::uint16_t ls = 0; ls < max_ls; ++ls)
                {
                    x_trial_soc = s.x + alpha_soc * p_combined;

                    if(has_finite_bounds)
                        x_trial_soc = detail::project(x_trial_soc,
                                                      s.lower, s.upper);

                    f_trial_soc = s.problem->value(x_trial_soc);
                    if(m > 0)
                        s.problem->constraints(x_trial_soc, s.bufs.c_all);
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
                        p = p_combined;
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
            //               (in-tree precedent — landed alongside this refactor).
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
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard.
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
        {
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
            .diagnostics = { .bfgs_reset_count = reset_count },
        };
    }

    // Hot start -- preserves BFGS Hessian.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        constexpr int M = state_type<P>::M;

        const int n = static_cast<int>(x0.size());
        const int m = s.n_eq + s.n_ineq;
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        if(m > 0)
            s.problem->constraints(x0, s.bufs.c_all);
        s.c_eq = s.bufs.c_all.head(s.n_eq);
        s.c_ineq = s.bufs.c_all.tail(s.n_ineq);
        if(m > 0)
            s.problem->constraint_jacobian(x0, s.bufs.J_all);
        s.J_eq = s.bufs.J_all.topRows(s.n_eq);
        s.J_ineq = s.bufs.J_all.bottomRows(s.n_ineq);
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS Hessian.
    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<double, N>& x0)
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

    template <typename P>
    static double lagrangian_gradient_norm(const state_type<P>& s)
    {
        constexpr int M = state_type<P>::M;

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

}

#endif
