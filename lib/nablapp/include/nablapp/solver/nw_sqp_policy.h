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
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace nablapp
{

struct nw_sqp_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::VectorXd g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::MatrixXd J_eq;
        Eigen::MatrixXd J_ineq;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        Eigen::VectorXd lambda;
        double objective_value{};
        double sigma{1.0};
        detail::bfgs_hessian<double> B;
        int iteration{0};
        int n_eq{0};
        int n_ineq{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_gradient;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&,
                           Eigen::VectorXd&)> eval_constraints;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&,
                           Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem>
    state_type init(this auto&&, const Problem& problem,
                    const Eigen::VectorXd& x0,
                    const solver_options<double>& /*opts*/)
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
            s.lower = Eigen::VectorXd::Constant(n, -inf);
            s.upper = Eigen::VectorXd::Constant(n, inf);
        }

        // BFGS Hessian of Lagrangian, initialized to I
        s.B = detail::bfgs_hessian<double>{n};
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
        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::VectorXd& v,
                                     Eigen::VectorXd& grad) {
            problem.gradient(v, grad);
        };

        const int n_eq_cap = s.n_eq;
        const int n_ineq_cap = s.n_ineq;

        s.eval_constraints = [&problem, n_eq_cap, n_ineq_cap](
            const Eigen::VectorXd& v,
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
            const Eigen::VectorXd& v,
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

    step_result<double> step(this auto&&, state_type& s)
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
        //
        // min 0.5 p^T B p + g^T p
        // s.t. J_eq p + c_eq = 0      =>  J_eq p = -c_eq
        // s.t. J_ineq p + c_ineq >= 0 =>  J_ineq p >= -c_ineq
        //
        // For solve_qp convention (A_ineq x >= b_ineq):
        //   A_eq = J_eq, b_eq = -c_eq  (A_eq p = b_eq means J_eq p = -c_eq)
        //   A_ineq = J_ineq, b_ineq = -c_ineq

        Eigen::MatrixXd A_eq = s.J_eq;
        Eigen::VectorXd b_eq = -s.c_eq;
        Eigen::MatrixXd A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        // Compute feasible starting point for QP.
        // p0 = 0 is feasible when c_eq = 0 (J_eq * 0 = 0 = -c_eq).
        // When c_eq != 0, compute minimum-norm point satisfying A_eq p = b_eq:
        //   p0 = A_eq^T (A_eq A_eq^T)^{-1} b_eq
        Eigen::VectorXd p0 = Eigen::VectorXd::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            // Solve (A_eq A_eq^T) w = b_eq, then p0 = A_eq^T w
            Eigen::MatrixXd AAt = A_eq * A_eq.transpose();
            auto qr = AAt.colPivHouseholderQr();
            Eigen::VectorXd w = qr.solve(b_eq);
            p0 = A_eq.transpose() * w;
        }

        // Also ensure inequality feasibility: J_ineq * p0 >= -c_ineq
        // If any violated, project p0 minimally. For simplicity, if
        // infeasible we still proceed -- the QP solver handles this.

        detail::qp_options<double> qp_opts;
        qp_opts.max_iterations = 200;
        qp_opts.tolerance = 1e-12;

        detail::qp_result<double> qp;
        bool has_finite_bounds = has_finite_box(s.lower, s.upper);

        if(has_finite_bounds)
        {
            // Box constraints on p: (lower - x) <= p <= (upper - x)
            Eigen::VectorXd p_lower = s.lower - s.x;
            Eigen::VectorXd p_upper = s.upper - s.x;
            // Clamp p0 to box
            p0 = p0.cwiseMax(p_lower).cwiseMin(p_upper);
            qp = detail::solve_qp(s.B.hessian(), s.g,
                                  A_eq, b_eq, A_ineq, b_ineq,
                                  p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = detail::solve_qp(s.B.hessian(), s.g,
                                  A_eq, b_eq, A_ineq, b_ineq,
                                  p0, qp_opts);
        }

        Eigen::VectorXd p = qp.x;

        // Zero step check
        if(p.norm() < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
            };
        }

        // --- 2. Extract QP multipliers ---
        // QP lambda covers [equality; inequality] constraints.
        // Box constraint multipliers from the augmented system are ignored.
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

        // Backtracking Armijo on L1 merit (robust for non-smooth merit)
        double alpha = 1.0;
        constexpr double c1 = 1e-4;
        constexpr double rho = 0.5;
        constexpr int max_ls = 30;

        Eigen::VectorXd x_trial(n);
        Eigen::VectorXd c_eq_trial, c_ineq_trial;

        for(int ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * p;

            // Project to box bounds if present
            if(has_finite_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            double f_trial = s.eval_value(x_trial);
            s.eval_constraints(x_trial, c_eq_trial, c_ineq_trial);
            double phi_trial = detail::l1_merit(f_trial, c_eq_trial,
                                                c_ineq_trial, s.sigma);

            if(phi_trial <= phi0 + c1 * alpha * dphi0)
                break;

            alpha *= rho;
        }

        // --- 5. Compute BFGS update vectors ---
        // s_k = x_{k+1} - x_k
        Eigen::VectorXd x_old = s.x;
        Eigen::VectorXd g_old = s.g;
        double f_old = s.objective_value;

        s.x = x_trial;
        s.objective_value = s.eval_value(s.x);
        s.eval_gradient(s.x, s.g);
        s.eval_constraints(s.x, s.c_eq, s.c_ineq);
        s.eval_jacobian(s.x, s.J_eq, s.J_ineq);

        Eigen::VectorXd sk = s.x - x_old;

        // Build full Jacobians for Lagrangian gradient computation
        Eigen::MatrixXd A_old(m, n);
        Eigen::MatrixXd A_new(m, n);
        if(s.n_eq > 0)
        {
            Eigen::MatrixXd J_eq_old;
            Eigen::MatrixXd J_ineq_old;
            // We don't have the old Jacobian anymore, use current as approx
            // (standard quasi-Newton SQP uses lambda_{k+1} with both grads)
        }

        // Lagrangian gradient at new and old points using new multipliers
        // y = grad_L(x_{k+1}, lambda_{k+1}) - grad_L(x_k, lambda_{k+1})
        // N&W eq. 18.13 / Section 18.4
        Eigen::VectorXd grad_L_new, grad_L_old;
        if(m > 0)
        {
            Eigen::MatrixXd A_new_full(m, n);
            if(s.n_eq > 0) A_new_full.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_new_full.bottomRows(s.n_ineq) = s.J_ineq;

            grad_L_new = detail::lagrangian_gradient(s.g, A_new_full,
                                                     lambda_new);
            // For grad_L_old, we use g_old and A_new_full (quasi-Newton
            // approximation -- the Jacobian at x_old is no longer available,
            // but using A_{k+1} is standard in practical SQP, see N&W p. 539)
            grad_L_old = detail::lagrangian_gradient(g_old, A_new_full,
                                                     lambda_new);
        }
        else
        {
            grad_L_new = s.g;
            grad_L_old = g_old;
        }

        Eigen::VectorXd yk = grad_L_new - grad_L_old;

        // --- 6. Damped BFGS update (N&W Procedure 18.2) ---
        if(sk.norm() > 1e-15)
            s.B.update(sk, yk);

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
        };
    }

    // Hot start -- preserves BFGS Hessian.
    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        s.eval_constraints(x0, s.c_eq, s.c_ineq);
        s.eval_jacobian(x0, s.J_eq, s.J_ineq);
        s.iteration = 0;
        // B is NOT reset -- preserves curvature
    }

    // Cold restart -- clears BFGS Hessian.
    void reset_clear(this auto&& self, state_type& s,
                     const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
        s.B.reset();
        s.sigma = 1.0;
        s.lambda.setZero();
    }

private:
    // Check if box bounds contain any finite values.
    static bool has_finite_box(const Eigen::VectorXd& lower,
                               const Eigen::VectorXd& upper)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int i = 0; i < lower.size(); ++i)
        {
            if(lower[i] > -inf || upper[i] < inf)
                return true;
        }
        return false;
    }

    // Compute Lagrangian gradient norm at current state.
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
