#ifndef HPP_GUARD_NABLAPP_SOLVER_NW_SQP_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_NW_SQP_POLICY_H

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

#include "nablapp/detail/lagrangian.h"
#include "nablapp/detail/adaptive_bfgs.h"
#include "nablapp/detail/kkt_residual.h"
#include "nablapp/detail/merit_function.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/options/qp_options.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp
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
        Eigen::MatrixXd J_eq;
        Eigen::MatrixXd J_ineq;
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
        detail::adaptive_bfgs<double, N, 10> hessian;
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
        s.hessian = detail::adaptive_bfgs<double, N, 10>(n);
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

        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        Eigen::Matrix<double, Eigen::Dynamic, N> A_eq = s.J_eq;
        s.b_eq_workspace = -s.c_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> A_ineq = s.J_ineq;
        s.b_ineq_workspace = -s.c_ineq;

        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Solve (A_eq A_eq^T) w = b_eq, then p0 = A_eq^T w
            // A*A^T is PSD but may be rank-deficient with numerical noise;
            // LDLT handles indefinite/singular cases that LLT cannot.
            s.AAt_workspace.noalias() = A_eq * A_eq.transpose();
            s.ldlt_feasibility.compute(s.AAt_workspace);
            auto w = s.ldlt_feasibility.solve(s.b_eq_workspace);
            p0.noalias() = A_eq.transpose() * w;
        }

        // Use embedded QP options with defaults
        nablapp::qp_options qp_opts;
        qp_opts.max_iterations = options.qp.max_iterations.has_value()
            ? options.qp.max_iterations
            : std::optional<std::uint16_t>{200};
        qp_opts.tolerance = options.qp.tolerance.has_value()
            ? options.qp.tolerance
            : std::optional<double>{1e-12};

        detail::qp_result<double, N> qp;
        bool has_finite_bounds = has_finite_box(s.lower, s.upper);

        if(has_finite_bounds)
        {
            Eigen::Vector<double, N> p_lower = (Eigen::Vector<double, N>(s.lower) -
                                                  Eigen::Vector<double, N>(s.x)).eval();
            Eigen::Vector<double, N> p_upper = (Eigen::Vector<double, N>(s.upper) -
                                                  Eigen::Vector<double, N>(s.x)).eval();
            p0 = p0.cwiseMax(p_lower).cwiseMin(p_upper);
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   A_eq, s.b_eq_workspace, A_ineq, s.b_ineq_workspace,
                                   p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = s.qp_solver.solve(s.hessian.hessian(), s.g,
                                   A_eq, s.b_eq_workspace, A_ineq, s.b_ineq_workspace,
                                   p0, qp_opts);
        }

        Eigen::Vector<double, N> p = qp.x;

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
        if(p.norm() < 1e-15)
        {
            Eigen::VectorXd lambda_null = Eigen::VectorXd::Zero(m);
            if(m > 0 && qp.lambda.size() >= m)
                lambda_null = qp.lambda.head(m);
            else if(m > 0 && qp.lambda.size() > 0)
                lambda_null.head(qp.lambda.size()) = qp.lambda;

            Eigen::VectorXd lambda_eq_null = s.n_eq > 0
                ? Eigen::VectorXd(lambda_null.head(s.n_eq))
                : Eigen::VectorXd::Zero(0);
            Eigen::VectorXd mu_ineq_null = s.n_ineq > 0
                ? Eigen::VectorXd(lambda_null.segment(s.n_eq, s.n_ineq))
                : Eigen::VectorXd::Zero(0);
            // Reference: N&W 2e Definition 12.1, eq. 12.34 (full KKT
            //            first-order optimality E-measure).
            double kkt_null = detail::kkt_residual<double,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_null, mu_ineq_null,
                s.c_eq, s.c_ineq);

            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_null,
            };
        }

        // --- 2. Extract QP multipliers ---
        Eigen::VectorXd lambda_new = Eigen::VectorXd::Zero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // --- 3. Update penalty parameter (N&W eq. 18.36) ---
        s.sigma = detail::update_penalty(s.sigma, lambda_new);

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

        // Initialize x_trial from s.x so that the post-loop read at
        // s.x = x_trial; is well-defined even in the pathological
        // max_ls == 0 configuration. GCC 15 flags an uninitialized SSE
        // packet load through this path if x_trial is left default-
        // constructed after only being assigned inside the loop body.
        Eigen::Vector<double, N> x_trial = s.x;
        double f_trial{s.objective_value};

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
                break;

            alpha *= ls_rho;
        }

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

        // --- 6. BFGS curvature push (adaptive_bfgs with Shanno rescaling) ---
        // Skip BFGS updates on non-positive curvature pairs.
        // In SQP the Lagrangian gradient difference y_k can easily
        // have s^T y < 0 when the constraint Hessian contribution
        // (A_{k+1} - A_k)^T lam dominates the objective curvature --
        // feeding such a pair into the BFGS formula distorts B
        // instead of improving it. adaptive_bfgs::push does its own
        // s^T y > 0 guard, but checking here keeps the semantics
        // explicit in the policy step.
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

        // KKT residual: full first-order optimality error E(x, lambda,
        // mu) via the active-set QP multipliers. Equality multipliers
        // occupy the first n_eq entries of lambda_new; inequality
        // multipliers follow. When m == 0 the helper collapses to
        // ||grad_f||_inf.
        //
        // Reference: N&W 2e Definition 12.1 (KKT conditions:
        //            stationarity, primal feasibility, dual feasibility,
        //            complementarity); eq. 12.34 (Lagrangian
        //            stationarity leg).
        Eigen::VectorXd lambda_eq_kkt = s.n_eq > 0
            ? Eigen::VectorXd(lambda_new.head(s.n_eq))
            : Eigen::VectorXd::Zero(0);
        Eigen::VectorXd mu_ineq_kkt = s.n_ineq > 0
            ? Eigen::VectorXd(lambda_new.segment(s.n_eq, s.n_ineq))
            : Eigen::VectorXd::Zero(0);
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq_kkt, mu_ineq_kkt,
            s.c_eq, s.c_ineq);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_L_norm,
            .step_size = sk.norm(),
            .objective_change = s.objective_value - f_old,
            .improved = phi_new < phi0,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
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
