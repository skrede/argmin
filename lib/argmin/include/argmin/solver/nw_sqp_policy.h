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
#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/merit_function.h"
#include "argmin/detail/active_set_qp.h"
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

        // Trigger threshold for the Maratos second-order correction
        // (SOC) retry. The SOC fires only when the Armijo line search
        // rejects the unit step AND the iter-0 (L1) constraint
        // violation exceeds this threshold; a small violation does not
        // produce enough Maratos curvature to justify the extra QP
        // re-solve. Default 1e-3 matches kraft_slsqp's SOC trigger
        // baseline. The optional shape lets callers express
        // "default" (empty) distinctly from "disabled by zero" (set
        // to 0.0 -> SOC always fires when ls fails).
        //
        // Field name aligns with the filter-family policies
        // (filter_slsqp_policy, filter_nw_sqp_policy) which already
        // expose soc_violation_threshold; nw_sqp follows the filter
        // family naming because it is architecturally closer to
        // filter_nw_sqp than to kraft_slsqp.
        //
        // Reference: Kraft, D. (1988). DFVLR-FB 88-28, §2.2.4
        //            (second-order correction).
        //            N&W 2e Section 18.3 (Maratos effect).
        std::optional<double> soc_violation_threshold{};

        // BFGS-reset-on-LS-failure cap. On Armijo line-search
        // exhaustion (with the Maratos SOC retry already attempted
        // and failed), reset the BFGS Hessian to identity (Shanno-
        // rescaled on next push) and retry the QP + Armijo + SOC
        // together up to bfgs_reset_max times. After exhaustion,
        // return a null step with diagnostics.bfgs_reset_count
        // populated. Default 5 = NLopt slsqp.c:1890-1895 ireset
        // parity.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset loop).
        // argmin variant: retry covers QP + Armijo + SOC together
        // (per-step recovery); SOC alone is the per-Armijo recovery
        // for Maratos curvature. The two retries compose so that on
        // Armijo+SOC failure a fresh BFGS Hessian gets a chance to
        // produce a different QP direction that itself runs through
        // Armijo + SOC.
        // Adopted from: NLopt slsqp.c:1890-1895.
        std::size_t bfgs_reset_max{5};

        std::uint16_t stall_window{50};
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
        // Pre-allocated constraint workspace (sized in init(), reused in step())
        // Eliminates per-step heap allocations for constraint evaluation.
        Eigen::VectorXd c_all;
        Eigen::MatrixXd J_all;
        Eigen::VectorXd c_eq_trial;
        Eigen::VectorXd c_ineq_trial;
        Eigen::MatrixXd J_all_old;  // Saved Jacobian at x_k for BFGS update (N&W eq. 18.13)
        Eigen::VectorXd b_eq_workspace;     // QP RHS workspace
        Eigen::VectorXd b_ineq_workspace;   // QP RHS workspace
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

        // Pre-allocate constraint workspace (sized once, reused in step()).
        // Follows the kraft_slsqp pre-allocation pattern.
        s.c_all.resize(m);
        s.J_all.resize(m, n);
        s.J_all_old.setZero(m, n);
        s.c_eq_trial.resize(s.n_eq);
        s.c_ineq_trial.resize(s.n_ineq);
        s.b_eq_workspace.resize(s.n_eq);
        s.b_ineq_workspace.resize(s.n_ineq);
        if(s.n_eq > 0)
        {
            s.AAt_workspace.resize(s.n_eq, s.n_eq);
            s.ldlt_feasibility = Eigen::LDLT<Eigen::MatrixXd>(s.n_eq);
        }

        // Evaluate constraints: single vector, split into eq/ineq
        if(m > 0)
            problem.constraints(x0, s.c_all);
        s.c_eq = s.c_all.head(s.n_eq);
        s.c_ineq = s.c_all.tail(s.n_ineq);

        // Evaluate Jacobian: single matrix, split into eq/ineq
        if(m > 0)
            problem.constraint_jacobian(x0, s.J_all);
        s.J_eq = s.J_all.topRows(s.n_eq);
        s.J_ineq = s.J_all.bottomRows(s.n_ineq);

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

        // BFGS-reset-on-LS-failure retry loop (NLopt slsqp.c:1890-1895
        // ireset pattern). On Armijo + SOC exhaustion, reset the BFGS
        // Hessian to identity (Shanno-rescaled on next push) and retry
        // the QP + Armijo + SOC together up to bfgs_reset_max times.
        // After exhaustion, return a null step with
        // diagnostics.bfgs_reset_count populated so the caller can
        // detect cap exhaustion via:
        //   is_null_step && diagnostics.bfgs_reset_count >= options.bfgs_reset_max
        //
        // Replaces the previous fall-through-on-LS-failure behaviour
        // where nw_sqp would update BFGS curvature from a step that
        // never satisfied merit decrease.
        //
        // argmin variant: retry covers QP + Armijo + SOC together
        // (per-step recovery); SOC alone is the per-Armijo recovery
        // for Maratos curvature. The two retries compose so that on
        // Armijo+SOC failure a fresh BFGS Hessian gets a chance to
        // produce a different QP direction that itself runs through
        // Armijo + SOC.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset loop).
        // Adopted from: NLopt slsqp.c:1890-1895.

        // Hoisted state observed by the BFGS-update block AFTER the
        // retry loop closes. Declaring these at function scope (not
        // inside the loop body) keeps the post-loop block unchanged
        // while letting the loop iterate on QP + Armijo + SOC.
        std::size_t reset_count = 0;
        bool ls_success = false;
        bool zero_step_detected = false;
        detail::qp_result<double, N> qp;
        Eigen::Vector<double, N> p = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> x_trial = s.x;
        double f_trial{s.objective_value};
        Eigen::VectorXd lambda_null;        // populated only on zero_step_detected path
        Eigen::VectorXd lambda_eq_null;     // populated only on zero_step_detected path
        Eigen::VectorXd mu_ineq_null;       // populated only on zero_step_detected path
        double grad_L_null_norm{0.0};       // populated only on zero_step_detected path
        double kkt_null{0.0};               // populated only on zero_step_detected path

        for(;;)
        {
        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        // Pass s.J_eq / s.J_ineq directly into the QP solver — the
        // state matrices already match the Matrix<Scalar, Meq, N> shape
        // the templated solve() expects, so no local A_eq/A_ineq copy
        // is needed.
        s.b_eq_workspace = -s.c_eq;
        s.b_ineq_workspace = -s.c_ineq;

        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Solve (J_eq J_eq^T) w = b_eq, then p0 = J_eq^T w
            // J*J^T is PSD but may be rank-deficient with numerical noise;
            // LDLT handles indefinite/singular cases that LLT cannot.
            s.AAt_workspace.noalias() = s.J_eq * s.J_eq.transpose();
            s.ldlt_feasibility.compute(s.AAt_workspace);
            auto w = s.ldlt_feasibility.solve(s.b_eq_workspace);
            p0.noalias() = s.J_eq.transpose() * w;
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
            Eigen::Vector<double, N> p_lower = (Eigen::Vector<double, N>(s.lower) -
                                                  Eigen::Vector<double, N>(s.x)).eval();
            Eigen::Vector<double, N> p_upper = (Eigen::Vector<double, N>(s.upper) -
                                                  Eigen::Vector<double, N>(s.x)).eval();
            p0 = p0.cwiseMax(p_lower).cwiseMin(p_upper);
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.b_eq_workspace, s.J_ineq, s.b_ineq_workspace,
                                   p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   s.J_eq, s.b_eq_workspace, s.J_ineq, s.b_ineq_workspace,
                                   p0, qp_opts);
        }

        p = qp.x;

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
        //
        // Inside the BFGS-reset retry loop: a post-reset zero
        // direction is either a true KKT point (the reset confirms
        // it) or active-set degeneracy that no further reset can
        // resolve. Capture the diagnostic state and break out of the
        // loop; the post-loop block returns the null-step result with
        // diagnostics.bfgs_reset_count = reset_count populated.
        if(p.norm() < 1e-15)
        {
            lambda_null = Eigen::VectorXd::Zero(m);
            if(m > 0 && qp.lambda.size() >= m)
                lambda_null = qp.lambda.head(m);
            else if(m > 0 && qp.lambda.size() > 0)
                lambda_null.head(qp.lambda.size()) = qp.lambda;

            lambda_eq_null = s.n_eq > 0
                ? Eigen::VectorXd(lambda_null.head(s.n_eq))
                : Eigen::VectorXd::Zero(0);
            mu_ineq_null = s.n_ineq > 0
                ? Eigen::VectorXd(lambda_null.segment(s.n_eq, s.n_ineq))
                : Eigen::VectorXd::Zero(0);
            // Reference: N&W 2e Definition 12.1, eq. 12.34 (full KKT
            //            first-order optimality E-measure).
            kkt_null = detail::kkt_residual<double,
                                            Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_null, mu_ineq_null,
                s.c_eq, s.c_ineq);

            // Use the CURRENT QP multipliers (lambda_eq_null /
            // mu_ineq_null) to form the Lagrangian-gradient norm.
            // The previous lagrangian_gradient_norm(s) call read
            // s.lambda from the prior successful step, which is stale
            // on a zero-step bailout and made gradient_tolerance_criterion
            // fire silently at non-stationary HS071 iterates.
            grad_L_null_norm = s.g.norm();
            if(m > 0)
            {
                Eigen::Vector<double, N> grad_L = s.g;
                if(s.n_eq > 0)
                    grad_L.noalias() -= s.J_eq.transpose() * lambda_eq_null;
                if(s.n_ineq > 0)
                    grad_L.noalias() -= s.J_ineq.transpose() * mu_ineq_null;
                grad_L_null_norm = grad_L.norm();
            }

            zero_step_detected = true;
            break;
        }

        // --- 2. Extract QP multipliers ---
        Eigen::VectorXd lambda_new = Eigen::VectorXd::Zero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // --- 3. Update penalty parameter (N&W eq. 18.36) ---
        s.sigma = detail::update_penalty(s.sigma, lambda_new);

        // Iter-0 cold-start calibration of sigma — magnitude-aware
        // floor guarding against an under-weighted violation term on
        // objective-dominated initial points. The default cold-start
        // sigma_0 = 1.0 admits an iter-0 step that satisfies merit-
        // decrease against a violation term too weak to dominate the
        // objective, parking the iterate strongly infeasible (HS071
        // from x_0 = (1, 5, 5, 1) is the canonical witness: lambda is
        // O(1) but |f_0| / ||c_0||_1 is O(5)). The K-factor floor
        // computed here is the magnitude-aware companion to the N&W
        // eq. 18.36 lambda-floor; both are applied at iter-0 and the
        // steady-state monotone update_penalty call above retains its
        // delta = 1e-4 margin for iter > 0.
        //
        // The c_0_l1 > feasibility-tolerance gate skips the cold-start
        // when the initial iterate is already (near-)feasible. The
        // K-factor floor's denominator collapses to its eps regularizer
        // in that regime and would push sigma to ~K * |f_0| / eps —
        // catastrophically large on problems like HS026 from
        // x_0 = (-2.6, 2, 2) which start exactly feasible (c_0 = 0).
        // The lambda-floor branch of calibrate_initial_penalty is also
        // unnecessary at exact feasibility because update_penalty's
        // monotone rule above has already enforced the lambda-bound.
        //
        // Reference: N&W 2e Section 18.3 / eq. 18.36 (lambda-floor for
        //            descent on the L1 merit);
        //            Kraft 1988 DFVLR-FB 88-28 Section 2.2.6 (sigma
        //            update rule companion).
        if(s.iteration == 0 && lambda_new.size() > 0)
        {
            const double c_0_l1 =
                detail::constraint_violation(s.c_eq, s.c_ineq);
            if(c_0_l1 > 1e-6)
            {
                s.sigma = detail::calibrate_initial_penalty(
                    s.sigma, lambda_new, s.objective_value, c_0_l1);
            }
        }

        // --- 4. Line search on L1 merit ---
        double phi0 = detail::l1_merit(s.objective_value,
                                       s.c_eq, s.c_ineq, s.sigma);
        double dphi0 = detail::l1_merit_directional_derivative(
            s.g.dot(p), s.c_eq, s.c_ineq, s.sigma);

        // If dphi0 >= 0, the penalty is insufficient for descent.
        // Increase sigma until the SQP direction is a descent direction
        // for the L1 merit function (N&W eq. 18.36).
        if(dphi0 >= 0.0)
        {
            double grad_f_dot_p = s.g.dot(p);
            double cv = detail::constraint_violation(s.c_eq, s.c_ineq);
            if(cv > 1e-15)
            {
                s.sigma = std::max(s.sigma, (grad_f_dot_p / cv) + 1.0);
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
        double alpha = 1.0;
        const double ls_c1 = options.line_search.c1;
        const double ls_rho = options.line_search.rho;
        const std::uint16_t max_ls = options.line_search.max_iterations;

        // L1 constraint violation at the iter-start point. Snapshot
        // here (before x_trial moves) so the SOC trigger gate below
        // and the Maratos block can both reuse it.
        const double constraint_viol_0 =
            detail::constraint_violation(s.c_eq, s.c_ineq);

        // Initialize x_trial from s.x so that the post-loop read at
        // s.x = x_trial; is well-defined even in the pathological
        // max_ls == 0 configuration. GCC 15 flags an uninitialized SSE
        // packet load through this path if x_trial is left default-
        // constructed after only being assigned inside the loop body.
        // x_trial / f_trial / ls_success are hoisted to function
        // scope above (BFGS-reset retry loop variables); reset to the
        // current iterate at every retry so the merit / SOC blocks
        // see a fresh trial baseline.
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
                s.problem->constraints(x_trial, s.c_all);
            s.c_eq_trial = s.c_all.head(s.n_eq);
            s.c_ineq_trial = s.c_all.tail(s.n_ineq);
            double phi_trial = detail::l1_merit(f_trial, s.c_eq_trial,
                                                s.c_ineq_trial, s.sigma);

            if(phi_trial <= phi0 + ls_c1 * alpha * dphi0)
            {
                ls_success = true;
                break;
            }

            alpha *= ls_rho;
        }

        // Second-order correction (Maratos retry).
        //
        // If the Armijo line search rejected the unit step because
        // the linearization underestimates the constraint curvature
        // (the canonical Maratos failure mode on HS071-class problems
        // with a tight nonlinear inequality), re-solve the QP with an
        // updated RHS that subtracts the nonlinear constraint residual
        // at the trial point x + p. Using the ORIGINAL Jacobian J_k
        // (not J_{k+1}) preserves the QP structure so the retry only
        // touches the constraint RHS.
        //
        // RHS update:
        //   b_eq_soc   = -c_eq(x + p)   + J_eq(x)   * p
        //   b_ineq_soc = -c_ineq(x + p) + J_ineq(x) * p
        //
        // The correction step dp is added to p to form a combined
        // direction p + dp; the line search is retried on that
        // direction with the same merit function and slope.
        //
        // The trigger is gated on a minimum L1 violation: SOC is only
        // worthwhile when there is enough constraint curvature to
        // justify the QP re-solve. Default 1e-3 matches the kraft
        // family default. nw_sqp uses stack-local SOC vectors here
        // (kraft uses state-resident buffers, filter family uses a
        // mix); a future refactor to a shared sqp_state_buffers will
        // absorb these into reused storage. nw_sqp's
        // active_set_qp_solver does not expose factor-reuse, so the
        // SOC retry calls solve() with the original Hessian.
        //
        // Reference: Kraft, D. (1988). DFVLR-FB 88-28, §2.2.4
        //            (second-order correction).
        //            N&W 2e Section 18.3 ("Maratos effect" and
        //            second-order corrections in line-search SQP).
        //            Adopted from in-tree kraft_slsqp_policy.h SOC
        //            block; the in-tree mirror keeps the algorithmic
        //            shape and adapts buffer/solver shapes to nw_sqp.
        const double soc_threshold =
            options.soc_violation_threshold.value_or(1e-3);
        if(!ls_success && constraint_viol_0 > soc_threshold)
        {
            if constexpr(constrained<P>)
            {
                if(m > 0)
                {
                    Eigen::Vector<double, N> x_soc_full = s.x + p;
                    if(has_finite_bounds)
                        x_soc_full = detail::project(x_soc_full,
                                                     s.lower, s.upper);

                    Eigen::VectorXd c_soc_full(m);
                    s.problem->constraints(x_soc_full, c_soc_full);
                    auto c_eq_soc = c_soc_full.head(s.n_eq);
                    auto c_ineq_soc = c_soc_full.tail(s.n_ineq);

                    Eigen::VectorXd b_eq_soc;
                    Eigen::VectorXd b_ineq_soc;
                    if(s.n_eq > 0)
                    {
                        b_eq_soc.resize(s.n_eq);
                        b_eq_soc.noalias() = -c_eq_soc + s.J_eq * p;
                    }
                    if(s.n_ineq > 0)
                    {
                        b_ineq_soc.resize(s.n_ineq);
                        b_ineq_soc.noalias() = -c_ineq_soc + s.J_ineq * p;
                    }

                    // Re-solve the QP with the SOC RHS. Hessian,
                    // gradient, and Jacobians are unchanged; only the
                    // constraint RHS shifts.
                    detail::qp_result<double, N> soc_qp;
                    if(has_finite_bounds)
                    {
                        Eigen::Vector<double, N> p_lower =
                            (Eigen::Vector<double, N>(s.lower) -
                             Eigen::Vector<double, N>(s.x)).eval();
                        Eigen::Vector<double, N> p_upper =
                            (Eigen::Vector<double, N>(s.upper) -
                             Eigen::Vector<double, N>(s.x)).eval();
                        Eigen::Vector<double, N> p_soc0 =
                            Eigen::Vector<double, N>::Zero(n);
                        soc_qp = s.qp_solver.solve(
                            s.hessian.hessian(), s.g,
                            s.J_eq, b_eq_soc, s.J_ineq, b_ineq_soc,
                            p_lower, p_upper, p_soc0, qp_opts);
                    }
                    else
                    {
                        Eigen::Vector<double, N> p_soc0 =
                            Eigen::Vector<double, N>::Zero(n);
                        soc_qp = s.qp_solver.solve(
                            s.hessian.hessian(), s.g,
                            s.J_eq, b_eq_soc, s.J_ineq, b_ineq_soc,
                            p_soc0, qp_opts);
                    }

                    if(soc_qp.status == detail::qp_status::optimal)
                    {
                        Eigen::Vector<double, N> p_combined = p + soc_qp.x;

                        // Backtracking Armijo on the combined direction
                        // using the same phi0/dphi0 reference slope.
                        double alpha_soc = 1.0;
                        bool soc_accepted = false;
                        Eigen::Vector<double, N> x_soc_trial = s.x;
                        double f_soc_trial{s.objective_value};
                        Eigen::VectorXd c_eq_soc_trial = s.c_eq_trial;
                        Eigen::VectorXd c_ineq_soc_trial = s.c_ineq_trial;

                        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
                        {
                            x_soc_trial = s.x + alpha_soc * p_combined;
                            if(has_finite_bounds)
                                x_soc_trial = detail::project(
                                    x_soc_trial, s.lower, s.upper);

                            f_soc_trial = s.problem->value(x_soc_trial);
                            if(m > 0)
                                s.problem->constraints(x_soc_trial,
                                                       s.c_all);
                            c_eq_soc_trial = s.c_all.head(s.n_eq);
                            c_ineq_soc_trial = s.c_all.tail(s.n_ineq);
                            double phi_soc = detail::l1_merit(
                                f_soc_trial, c_eq_soc_trial,
                                c_ineq_soc_trial, s.sigma);

                            if(phi_soc <= phi0 + ls_c1 * alpha_soc * dphi0)
                            {
                                soc_accepted = true;
                                break;
                            }
                            alpha_soc *= ls_rho;
                        }

                        // Accept the corrected step only if it took a
                        // longer step than the original Armijo (which
                        // shrunk to alpha = max_ls's last value without
                        // accepting). Promotes the SOC outcome into the
                        // BFGS-update path below.
                        if(soc_accepted && alpha_soc > alpha)
                        {
                            p = p_combined;
                            alpha = alpha_soc;
                            x_trial = x_soc_trial;
                            f_trial = f_soc_trial;
                            s.c_eq_trial = c_eq_soc_trial;
                            s.c_ineq_trial = c_ineq_soc_trial;
                            ls_success = true;
                        }
                    }
                }
            }
        }

        // BFGS-reset retry loop close. If Armijo + SOC accepted a
        // descent direction, exit the loop and fall through to the
        // BFGS-update / step_result return path below. Otherwise,
        // if the cap is not yet hit, reset the BFGS Hessian to
        // identity (Shanno-rescaled on next push) and retry the
        // QP + Armijo + SOC together. After cap exhaustion, exit
        // and let the post-loop block return the cap-exhausted
        // null step with diagnostics.bfgs_reset_count = reset_count.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset loop).
        // Adopted from: NLopt slsqp.c:1890-1895.
        if(ls_success)
            break;
        if(reset_count >= options.bfgs_reset_max)
            break;
        s.hessian.reset();
        ++reset_count;
        }

        // Post-loop: zero-step path. The QP returned p ~= 0 inside
        // the loop (degeneracy / true KKT point), the loop captured
        // the diagnostic state and broke out. Surface
        // reset_count alongside the existing null-step telemetry so
        // callers can distinguish a fresh zero-step (reset_count == 0)
        // from one reached only after the BFGS Hessian was discarded
        // (reset_count > 0).
        if(zero_step_detected)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = grad_L_null_norm,
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_null,
                .diagnostics = solver_diagnostics{.bfgs_reset_count = reset_count},
            };
        }

        // Post-loop: cap-exhausted null step. Armijo + SOC failed on
        // every retry and the BFGS-reset cap is hit. Mirror the kraft
        // cap-exhausted return shape: refresh KKT-residual buffers
        // from the most recent qp.lambda multipliers and surface the
        // cumulative reset count so callers can compose
        //   is_null_step && diagnostics.bfgs_reset_count >= options.bfgs_reset_max
        // to detect cap exhaustion deterministically.
        //
        // Reference: N&W 2e Definition 12.1, eq. 12.34 (full KKT
        //            first-order optimality E-measure).
        if(!ls_success)
        {
            Eigen::VectorXd lambda_capped = Eigen::VectorXd::Zero(m);
            if(m > 0 && qp.lambda.size() >= m)
                lambda_capped = qp.lambda.head(m);
            else if(m > 0 && qp.lambda.size() > 0)
                lambda_capped.head(qp.lambda.size()) = qp.lambda;

            Eigen::VectorXd lambda_eq_capped = s.n_eq > 0
                ? Eigen::VectorXd(lambda_capped.head(s.n_eq))
                : Eigen::VectorXd::Zero(0);
            Eigen::VectorXd mu_ineq_capped = s.n_ineq > 0
                ? Eigen::VectorXd(lambda_capped.segment(s.n_eq, s.n_ineq))
                : Eigen::VectorXd::Zero(0);

            const double kkt_capped = detail::kkt_residual<double,
                                                           Eigen::Dynamic,
                                                           Eigen::Dynamic,
                                                           Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_capped, mu_ineq_capped,
                s.c_eq, s.c_ineq);

            // Lagrangian gradient norm at the iter-start point using
            // the most recent QP multipliers (consistent with the
            // zero-step branch above).
            double grad_L_capped_norm = s.g.norm();
            if(m > 0)
            {
                Eigen::Vector<double, N> grad_L = s.g;
                if(s.n_eq > 0)
                    grad_L.noalias() -= s.J_eq.transpose() * lambda_eq_capped;
                if(s.n_ineq > 0)
                    grad_L.noalias() -= s.J_ineq.transpose() * mu_ineq_capped;
                grad_L_capped_norm = grad_L.norm();
            }

            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = grad_L_capped_norm,
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_capped,
                .diagnostics = solver_diagnostics{.bfgs_reset_count = reset_count},
            };
        }

        // Recompute lambda_new from the accepted-step qp.lambda for
        // the BFGS-update block below. The local lambda_new declared
        // inside the (now-looped) region went out of scope when the
        // loop closed; the BFGS update needs the multipliers from the
        // QP solve that produced the accepted direction.
        Eigen::VectorXd lambda_new = Eigen::VectorXd::Zero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // L1 merit at iter-start point for the post-step `improved`
        // flag below. Local phi0 declared inside the (now-looped)
        // region went out of scope; recompute at s.x using the
        // current sigma so phi_new vs phi0 comparison stays valid.
        // sigma is monotone across retries so this is the most
        // conservative reference value.
        double phi0 = detail::l1_merit(s.objective_value,
                                       s.c_eq, s.c_ineq, s.sigma);

        // --- 5. Compute BFGS update vectors ---
        // Reuse objective and constraint values from line search (already
        // evaluated at x_trial). Only gradient and Jacobian need fresh eval.
        Eigen::Vector<double, N> x_old = s.x;
        Eigen::Vector<double, N> g_old = s.g;
        double f_old = s.objective_value;

        s.x = x_trial;
        s.objective_value = f_trial;
        s.problem->gradient(s.x, s.g);
        s.c_eq = s.c_eq_trial;
        s.c_ineq = s.c_ineq_trial;

        // Save old Jacobian before re-evaluation for BFGS update (N&W eq. 18.13)
        if(m > 0)
            s.J_all_old.noalias() = s.J_all;

        if(m > 0)
            s.problem->constraint_jacobian(s.x, s.J_all);
        s.J_eq = s.J_all.topRows(s.n_eq);
        s.J_ineq = s.J_all.bottomRows(s.n_ineq);

        Eigen::Vector<double, N> sk = s.x - x_old;

        // Lagrangian gradient at new and old points using new multipliers.
        // grad_L_old must use the Jacobian at x_old (not x_new) so that
        // y_k = grad_L_new - grad_L_old captures constraint curvature.
        // Reference: N&W eq. 18.13.
        Eigen::Vector<double, N> grad_L_new, grad_L_old;
        if(m > 0)
        {
            // grad_L = grad_f - J^T * lambda (N&W eq. 18.2-18.3)
            // Use J_all directly for new point (already updated above)
            grad_L_new = s.g;
            grad_L_new.noalias() -= s.J_all.transpose() * lambda_new;

            // Use J_all_old for old point (saved before Jacobian re-evaluation)
            grad_L_old = g_old;
            grad_L_old.noalias() -= s.J_all_old.transpose() * lambda_new;
        }
        else
        {
            grad_L_new = s.g;
            grad_L_old = g_old;
        }

        Eigen::Vector<double, N> yk = grad_L_new - grad_L_old;

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
        Eigen::VectorXd lambda_eq_kkt = Eigen::VectorXd::Zero(s.n_eq);
        Eigen::VectorXd mu_ineq_kkt = Eigen::VectorXd::Zero(s.n_ineq);
        if(m > 0)
        {
            Eigen::Vector<double, N> g_fixed = s.g;
            Eigen::VectorXd lambda_reest =
                detail::estimate_multipliers_active_set<double,
                                                        N,
                                                        Eigen::Dynamic,
                                                        Eigen::Dynamic>(
                    g_fixed, s.J_eq, s.J_ineq, s.c_ineq);
            if(s.n_eq > 0)
                lambda_eq_kkt = lambda_reest.head(s.n_eq);
            if(s.n_ineq > 0)
                mu_ineq_kkt = lambda_reest.segment(s.n_eq, s.n_ineq);
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq_kkt, mu_ineq_kkt,
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
            // Surface BFGS-reset count for telemetry. Zero on the
            // common success path (Armijo + SOC accepted on the
            // first try); non-zero only when the BFGS-reset retry
            // loop above performed at least one reset before the
            // QP + Armijo + SOC chain accepted a descent direction.
            // Caller telemetry can distinguish "step accepted on
            // first try" from "step accepted only after BFGS
            // curvature was discarded".
            .diagnostics = solver_diagnostics{.bfgs_reset_count = reset_count},
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
            s.problem->constraints(x0, s.c_all);
        s.c_eq = s.c_all.head(s.n_eq);
        s.c_ineq = s.c_all.tail(s.n_ineq);
        if(m > 0)
            s.problem->constraint_jacobian(x0, s.J_all);
        s.J_eq = s.J_all.topRows(s.n_eq);
        s.J_ineq = s.J_all.bottomRows(s.n_ineq);
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
