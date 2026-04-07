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
#include "nablapp/detail/merit_function.h"
#include "nablapp/detail/bfgs_hessian.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/options/bfgs_options.h"
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
        bfgs_options bfgs{};               // BFGS Hessian update params (N&W Proc 18.2)
        qp_options qp{};                   // QP subproblem params
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
        double objective_value{};
        double sigma{1.0};
        detail::bfgs_hessian<double, N> B;
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

        // Evaluate constraints: single vector, split into eq/ineq
        Eigen::VectorXd c_all(m);
        if(m > 0)
            problem.constraints(x0, c_all);
        s.c_eq = c_all.head(s.n_eq);
        s.c_ineq = c_all.tail(s.n_ineq);

        // Evaluate Jacobian: single matrix, split into eq/ineq
        Eigen::MatrixXd J_all(m, n);
        if(m > 0)
            problem.constraint_jacobian(x0, J_all);
        s.J_eq = J_all.topRows(s.n_eq);
        s.J_ineq = J_all.bottomRows(s.n_ineq);

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
        s.B = detail::bfgs_hessian<double, N>{n};
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
        Eigen::VectorXd b_eq = -s.c_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Solve (A_eq A_eq^T) w = b_eq, then p0 = A_eq^T w
            // A*A^T is PSD but may be rank-deficient with numerical noise;
            // LDLT handles indefinite/singular cases that LLT cannot.
            auto AAt = (A_eq * A_eq.transpose()).eval();
            Eigen::LDLT<decltype(AAt)> ldlt(AAt);
            auto w = ldlt.solve(b_eq);
            p0 = A_eq.transpose() * w;
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
            qp = s.qp_solver.solve(s.B.hessian(), s.g,
                                   A_eq, b_eq, A_ineq, b_ineq,
                                   p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = s.qp_solver.solve(s.B.hessian(), s.g,
                                   A_eq, b_eq, A_ineq, b_ineq,
                                   p0, qp_opts);
        }

        Eigen::Vector<double, N> p = qp.x;

        // Zero step check
        if(p.norm() < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
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

        // Backtracking Armijo on L1 merit using embedded line search options
        double alpha = 1.0;
        const double ls_c1 = options.line_search.c1;
        const double ls_rho = options.line_search.rho;
        const std::uint16_t max_ls = options.line_search.max_iterations;

        Eigen::Vector<double, N> x_trial(n);
        Eigen::VectorXd c_eq_trial, c_ineq_trial;
        double f_trial{};

        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * p;

            if(has_finite_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            f_trial = s.problem->value(x_trial);
            {
                Eigen::VectorXd c_all(m);
                if(m > 0)
                    s.problem->constraints(x_trial, c_all);
                c_eq_trial = c_all.head(s.n_eq);
                c_ineq_trial = c_all.tail(s.n_ineq);
            }
            double phi_trial = detail::l1_merit(f_trial, c_eq_trial,
                                                c_ineq_trial, s.sigma);

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
        s.c_eq = c_eq_trial;
        s.c_ineq = c_ineq_trial;
        {
            Eigen::MatrixXd J_all(m, n);
            if(m > 0)
                s.problem->constraint_jacobian(s.x, J_all);
            s.J_eq = J_all.topRows(s.n_eq);
            s.J_ineq = J_all.bottomRows(s.n_ineq);
        }

        Eigen::Vector<double, N> sk = s.x - x_old;

        // Lagrangian gradient at new and old points using new multipliers
        Eigen::Vector<double, N> grad_L_new, grad_L_old;
        if(m > 0)
        {
            Eigen::Matrix<double, Eigen::Dynamic, N> A_new_full(m, n);
            if(s.n_eq > 0) A_new_full.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_new_full.bottomRows(s.n_ineq) = s.J_ineq;

            grad_L_new = detail::lagrangian_gradient(s.g, A_new_full, lambda_new);
            grad_L_old = detail::lagrangian_gradient(g_old, A_new_full, lambda_new);
        }
        else
        {
            grad_L_new = s.g;
            grad_L_old = g_old;
        }

        Eigen::Vector<double, N> yk = grad_L_new - grad_L_old;

        // --- 6. Damped BFGS update (N&W Procedure 18.2) ---
        if(sk.norm() > 1e-15)
            s.B.update(sk, yk, options.bfgs);

        // --- 7. Update multipliers and iteration ---
        s.lambda = lambda_new;
        ++s.iteration;

        double grad_L_norm = grad_L_new.norm();

        double phi_new = detail::l1_merit(s.objective_value,
                                            s.c_eq, s.c_ineq, s.sigma);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_L_norm,
            .step_size = sk.norm(),
            .objective_change = s.objective_value - f_old,
            .improved = phi_new < phi0,
            .x_norm = s.x.norm(),
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
        {
            Eigen::VectorXd c_all(m);
            if(m > 0)
                s.problem->constraints(x0, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }
        {
            Eigen::MatrixXd J_all(m, n);
            if(m > 0)
                s.problem->constraint_jacobian(x0, J_all);
            s.J_eq = J_all.topRows(s.n_eq);
            s.J_ineq = J_all.bottomRows(s.n_ineq);
        }
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS Hessian.
    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.B.reset();
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
