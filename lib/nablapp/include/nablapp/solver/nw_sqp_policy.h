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
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
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

    struct state_type
    {
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
        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};

        std::function<double(const Eigen::Vector<double, N>&)> eval_value;
        std::function<void(const Eigen::Vector<double, N>&,
                           Eigen::Vector<double, N>&)> eval_gradient;
        std::function<void(const Eigen::Vector<double, N>&, Eigen::VectorXd&,
                           Eigen::VectorXd&)> eval_constraints;
        std::function<void(const Eigen::Vector<double, N>&, Eigen::MatrixXd&,
                           Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts,
                    const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "nw_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "nw_sqp_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type s;

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
        s.sigma = 1.0;
        s.iteration = 0;

        // Initial multiplier estimate via least-squares (N&W eq. 18.15)
        if(m > 0)
        {
            Eigen::MatrixXd A_all(m, n);
            if(s.n_eq > 0) A_all.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_all.bottomRows(s.n_ineq) = s.J_ineq;
            s.lambda = detail::estimate_multipliers(s.g, A_all);
        }
        else
        {
            s.lambda = Eigen::VectorXd::Zero(0);
        }

        // Closures capturing problem by const reference
        s.eval_value = [&problem](const Eigen::Vector<double, N>& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::Vector<double, N>& v,
                                     Eigen::Vector<double, N>& grad) {
            problem.gradient(v, grad);
        };

        const int n_eq_cap = s.n_eq;
        const int n_ineq_cap = s.n_ineq;

        s.eval_constraints = [&problem, n_eq_cap, n_ineq_cap](
            const Eigen::Vector<double, N>& v,
            Eigen::VectorXd& ceq, Eigen::VectorXd& cineq)
        {
            const int mtot = n_eq_cap + n_ineq_cap;
            Eigen::VectorXd c(mtot);
            if(mtot > 0)
                problem.constraints(v, c);
            ceq = c.head(n_eq_cap);
            cineq = c.tail(n_ineq_cap);
        };

        s.eval_jacobian = [&problem, n_eq_cap, n_ineq_cap](
            const Eigen::Vector<double, N>& v,
            Eigen::MatrixXd& Jeq, Eigen::MatrixXd& Jineq)
        {
            const int mtot = n_eq_cap + n_ineq_cap;
            const int n_dim = v.size();
            Eigen::MatrixXd J(mtot, n_dim);
            if(mtot > 0)
                problem.constraint_jacobian(v, J);
            Jeq = J.topRows(n_eq_cap);
            Jineq = J.bottomRows(n_ineq_cap);
        };

        return s;
    }

    step_result<double> step(state_type& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;

        // Re-evaluate at current x (skip on first iteration -- init did it)
        if(s.iteration != 0)
        {
            s.objective_value = s.eval_value(s.x);
            s.eval_gradient(s.x, s.g);
            s.eval_constraints(s.x, s.c_eq, s.c_ineq);
            s.eval_jacobian(s.x, s.J_eq, s.J_ineq);
        }

        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        Eigen::MatrixXd A_eq = s.J_eq;
        Eigen::VectorXd b_eq = -s.c_eq;
        Eigen::MatrixXd A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        Eigen::VectorXd p0 = Eigen::VectorXd::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            Eigen::MatrixXd AAt = A_eq * A_eq.transpose();
            auto qr = AAt.colPivHouseholderQr();
            Eigen::VectorXd w = qr.solve(b_eq);
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

        detail::qp_result<double> qp;
        bool has_finite_bounds = has_finite_box(s.lower, s.upper);

        if(has_finite_bounds)
        {
            Eigen::VectorXd p_lower = Eigen::VectorXd(s.lower) - Eigen::VectorXd(s.x);
            Eigen::VectorXd p_upper = Eigen::VectorXd(s.upper) - Eigen::VectorXd(s.x);
            p0 = p0.cwiseMax(p_lower).cwiseMin(p_upper);
            qp = detail::solve_qp(s.B.hessian(), Eigen::VectorXd(s.g),
                                  A_eq, b_eq, A_ineq, b_ineq,
                                  p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = detail::solve_qp(s.B.hessian(), Eigen::VectorXd(s.g),
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

        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * p;

            if(has_finite_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            double f_trial = s.eval_value(x_trial);
            s.eval_constraints(x_trial, c_eq_trial, c_ineq_trial);
            double phi_trial = detail::l1_merit(f_trial, c_eq_trial,
                                                c_ineq_trial, s.sigma);

            if(phi_trial <= phi0 + ls_c1 * alpha * dphi0)
                break;

            alpha *= ls_rho;
        }

        // --- 5. Compute BFGS update vectors ---
        Eigen::Vector<double, N> x_old = s.x;
        Eigen::Vector<double, N> g_old = s.g;
        double f_old = s.objective_value;

        s.x = x_trial;
        s.objective_value = s.eval_value(s.x);
        s.eval_gradient(s.x, s.g);
        s.eval_constraints(s.x, s.c_eq, s.c_ineq);
        s.eval_jacobian(s.x, s.J_eq, s.J_ineq);

        Eigen::Vector<double, N> sk = s.x - x_old;

        // Lagrangian gradient at new and old points using new multipliers
        Eigen::Vector<double, N> grad_L_new, grad_L_old;
        if(m > 0)
        {
            Eigen::MatrixXd A_new_full(m, n);
            if(s.n_eq > 0) A_new_full.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_new_full.bottomRows(s.n_ineq) = s.J_ineq;

            grad_L_new = detail::lagrangian_gradient(s.g, A_new_full,
                                                     lambda_new);
            grad_L_old = detail::lagrangian_gradient(g_old, A_new_full,
                                                     lambda_new);
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
    void reset(state_type& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        s.eval_constraints(x0, s.c_eq, s.c_ineq);
        s.eval_jacobian(x0, s.J_eq, s.J_ineq);
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS Hessian.
    void reset_clear(state_type& s,
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

    static double lagrangian_gradient_norm(const state_type& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;
        if(m == 0)
            return s.g.norm();

        Eigen::MatrixXd A(m, n);
        if(s.n_eq > 0) A.topRows(s.n_eq) = s.J_eq;
        if(s.n_ineq > 0) A.bottomRows(s.n_ineq) = s.J_ineq;
        return detail::lagrangian_gradient(s.g, A, s.lambda).norm();
    }
};

}

#endif
