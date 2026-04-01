#ifndef HPP_GUARD_NABLAPP_SOLVER_KRAFT_SLSQP_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_KRAFT_SLSQP_POLICY_H

// Kraft 1988 SLSQP policy for basic_solver.
//
// Implements Sequential Least Squares Programming (SLSQP) using the L-BFGS
// compact Hessian approximation instead of dense BFGS. Each step: (1) builds
// the QP subproblem with the compact L-BFGS Hessian, (2) solves via the
// dense active-set QP solver, (3) performs backtracking on an L1 merit
// function, (4) updates the L-BFGS curvature pairs.
//
// The L-BFGS Hessian gives O(mn) storage vs O(n^2) for dense BFGS, making
// Kraft SLSQP suitable for larger problems while matching NLopt LD_SLSQP
// behavior on robotics-scale problems (n=6-200).
//
// Supports: unconstrained, equality, inequality, box, and mixed constraints.
// When the problem satisfies only differentiable (no constraints/bounds),
// constraints default to empty and the solver reduces to L-BFGS with QP
// step computation.
//
// Reference: Kraft, D. (1988) "A Software Package for Sequential Quadratic
//            Programming", DFVLR-FB 88-28, Deutsche Forschungs- und
//            Versuchsanstalt fuer Luft- und Raumfahrt.
//            N&W Chapter 18 (SQP methods), Section 18.3 (L1 merit function).
//            K&W Section 10.4 (penalty methods, augmented Lagrangian).

#include "nablapp/detail/active_set_qp.h"
#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/line_search/armijo.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace nablapp
{

struct kraft_slsqp_policy
{
    using scalar_type = double;

    struct options_type
    {
        int history_depth{10};
        double initial_penalty{1.0};
        double penalty_growth{10.0};
    };

    options_type options{};

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
        detail::compact_lbfgs<double> lbfgs;
        int iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int n{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_gradient;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&)> eval_constraints;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&, Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem, typename Convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        self.options = policy_opts;
        return self.init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        const int n = problem.dimension();
        state_type s;

        s.n = n;
        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // Bounds
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

        // Constraints
        if constexpr(constrained<Problem>)
        {
            s.n_eq = problem.num_equality();
            s.n_ineq = problem.num_inequality();

            // The constrained concept uses a single constraints() call that fills
            // a combined vector. We split into eq and ineq parts.
            Eigen::VectorXd c_all(s.n_eq + s.n_ineq);
            problem.constraints(x0, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);

            Eigen::MatrixXd J_all(s.n_eq + s.n_ineq, n);
            problem.constraint_jacobian(x0, J_all);
            s.J_eq = J_all.topRows(s.n_eq);
            s.J_ineq = J_all.bottomRows(s.n_ineq);

            s.lambda = Eigen::VectorXd::Zero(s.n_eq + s.n_ineq);

            s.eval_constraints = [&problem, n_eq = s.n_eq, n_ineq = s.n_ineq](
                const Eigen::VectorXd& v, Eigen::VectorXd& ceq, Eigen::VectorXd& cineq) {
                Eigen::VectorXd c_all(n_eq + n_ineq);
                problem.constraints(v, c_all);
                ceq = c_all.head(n_eq);
                cineq = c_all.tail(n_ineq);
            };

            s.eval_jacobian = [&problem, n_eq = s.n_eq, n_ineq = s.n_ineq, n](
                const Eigen::VectorXd& v, Eigen::MatrixXd& Jeq, Eigen::MatrixXd& Jineq) {
                Eigen::MatrixXd J_all(n_eq + n_ineq, n);
                problem.constraint_jacobian(v, J_all);
                Jeq = J_all.topRows(n_eq);
                Jineq = J_all.bottomRows(n_ineq);
            };
        }
        else
        {
            s.n_eq = 0;
            s.n_ineq = 0;
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
            s.J_eq.resize(0, n);
            s.J_ineq.resize(0, n);
            s.lambda.resize(0);

            s.eval_constraints = [](const Eigen::VectorXd&, Eigen::VectorXd& ceq,
                                    Eigen::VectorXd& cineq) {
                ceq.resize(0);
                cineq.resize(0);
            };

            s.eval_jacobian = [n](const Eigen::VectorXd&, Eigen::MatrixXd& Jeq,
                                  Eigen::MatrixXd& Jineq) {
                Jeq.resize(0, n);
                Jineq.resize(0, n);
            };
        }

        s.lbfgs = detail::compact_lbfgs<double>{self.options.history_depth};
        s.sigma = self.options.initial_penalty;
        s.iteration = 0;

        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::VectorXd& v, Eigen::VectorXd& grad) {
            problem.gradient(v, grad);
        };

        return s;
    }

    step_result<double> step(this auto&&, state_type& s)
    {
        // Re-evaluate gradient (skip on first iteration -- init already computed it)
        if(s.iteration != 0)
        {
            s.eval_gradient(s.x, s.g);
            if(s.n_eq + s.n_ineq > 0)
            {
                s.eval_constraints(s.x, s.c_eq, s.c_ineq);
                s.eval_jacobian(s.x, s.J_eq, s.J_ineq);
            }
        }

        const int n = s.n;

        // Build QP Hessian from compact L-BFGS: B[:,j] = lbfgs.multiply(e_j).
        // For small n (liepp uses n=6-7), this column-by-column approach is O(n*m*n)
        // which is negligible.
        Eigen::MatrixXd B(n, n);
        Eigen::VectorXd ej = Eigen::VectorXd::Zero(n);
        for(int j = 0; j < n; ++j)
        {
            ej[j] = 1.0;
            B.col(j) = s.lbfgs.multiply(ej);
            ej[j] = 0.0;
        }

        // Symmetrize for numerical safety
        B = (0.5 * (B + B.transpose())).eval();

        // Build QP subproblem:
        //   min  0.5 * p^T B p + g^T p
        //   s.t. J_eq * p + c_eq = 0          (linearised equality)
        //        J_ineq * p + c_ineq >= 0      (linearised inequality)
        //        lower - x <= p <= upper - x    (box on step)
        //
        // In solve_qp convention: A_eq * p = b_eq, A_ineq * p >= b_ineq
        Eigen::MatrixXd A_eq = s.J_eq;
        Eigen::VectorXd b_eq = -s.c_eq;

        Eigen::MatrixXd A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        // Box bounds on the step: lower - x <= p <= upper - x
        Eigen::VectorXd p_lower = s.lower - s.x;
        Eigen::VectorXd p_upper = s.upper - s.x;

        // Clamp infinities to large finite for QP feasibility
        constexpr double big = 1e20;
        for(int i = 0; i < n; ++i)
        {
            if(p_lower[i] < -big) p_lower[i] = -big;
            if(p_upper[i] > big) p_upper[i] = big;
        }

        // Feasible starting point for QP: p0 = 0 is feasible for box since
        // lower - x <= 0 <= upper - x when x is feasible. But it may violate
        // equality constraints. Use p0 = 0 and let the active-set solver handle it.
        Eigen::VectorXd p0 = Eigen::VectorXd::Zero(n);

        // Solve QP with box constraints
        detail::qp_options<double> qp_opts;
        qp_opts.max_iterations = 10 * n;
        auto qp_res = detail::solve_qp(B, s.g, A_eq, b_eq, A_ineq, b_ineq,
                                        p_lower, p_upper, p0, qp_opts);

        Eigen::VectorXd p = qp_res.x;

        // Zero step check
        double p_norm = p.norm();
        if(p_norm < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
            };
        }

        // Update penalty parameter sigma.
        // N&W eq. 18.36: sigma >= max_i |lambda_i| + delta for some delta > 0.
        // This ensures the L1 merit function is exact.
        if(qp_res.lambda.size() > 0)
        {
            double lambda_max = qp_res.lambda.cwiseAbs().maxCoeff();
            if(lambda_max + 0.5 > s.sigma)
                s.sigma = lambda_max + 1.0;
        }

        // L1 merit function for line search:
        //   phi_1(x) = f(x) + sigma * (||c_eq(x)||_1 + ||max(0, -c_ineq(x))||_1)
        auto merit = [&](const Eigen::VectorXd& xk) -> double {
            double fval = s.eval_value(xk);
            double viol = 0.0;

            if(s.n_eq + s.n_ineq > 0)
            {
                Eigen::VectorXd ceq, cineq;
                s.eval_constraints(xk, ceq, cineq);

                if(ceq.size() > 0)
                    viol += ceq.cwiseAbs().sum();
                if(cineq.size() > 0)
                    viol += (-cineq).cwiseMax(0.0).sum();
            }

            return fval + s.sigma * viol;
        };

        double merit_0 = merit(s.x);

        // Directional derivative of L1 merit along p:
        //   D phi_1 = g^T p - sigma * (||c_eq||_1 + ||max(0,-c_ineq)||_1)
        // The second term is the constraint violation reduction (approximate).
        double constraint_viol_0 = 0.0;
        if(s.c_eq.size() > 0)
            constraint_viol_0 += s.c_eq.cwiseAbs().sum();
        if(s.c_ineq.size() > 0)
            constraint_viol_0 += (-s.c_ineq).cwiseMax(0.0).sum();

        double dphi_merit = s.g.dot(p) - s.sigma * constraint_viol_0;

        // Ensure descent direction for merit function
        if(dphi_merit >= 0.0)
            dphi_merit = -1e-8;

        // Backtracking Armijo line search on L1 merit
        auto phi_ls = [&](double alpha) {
            Eigen::VectorXd x_trial = s.x + alpha * p;
            // Project onto bounds for safety
            for(int i = 0; i < n; ++i)
            {
                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
            }
            return merit(x_trial);
        };

        line_search_options ls_opts;
        ls_opts.max_alpha = 1.0;
        ls_opts.c1 = 1e-4;
        ls_opts.rho = 0.5;
        ls_opts.max_iterations = 20;

        auto ls = armijo(phi_ls, merit_0, dphi_merit, ls_opts);
        double alpha = ls.alpha;

        // Ensure minimum step
        if(alpha < 1e-15)
            alpha = 1e-4;

        // Update iterate
        Eigen::VectorXd x_old = s.x;
        double old_f = s.objective_value;
        s.x = s.x + alpha * p;

        // Project onto bounds
        for(int i = 0; i < n; ++i)
        {
            if(s.x[i] < s.lower[i]) s.x[i] = s.lower[i];
            if(s.x[i] > s.upper[i]) s.x[i] = s.upper[i];
        }

        // Evaluate new objective
        s.objective_value = s.eval_value(s.x);

        // Evaluate new gradient for L-BFGS update
        Eigen::VectorXd g_old = s.g;
        s.eval_gradient(s.x, s.g);

        // Update constraints
        if(s.n_eq + s.n_ineq > 0)
        {
            s.eval_constraints(s.x, s.c_eq, s.c_ineq);
            s.eval_jacobian(s.x, s.J_eq, s.J_ineq);
        }

        // L-BFGS update using Lagrangian gradient:
        //   grad_L = grad_f - J_eq^T lambda_eq - J_ineq^T lambda_ineq
        // For simplicity (and matching Kraft's original), we use the objective
        // gradient directly when unconstrained, and the Lagrangian gradient
        // when constrained.
        Eigen::VectorXd grad_L_old = g_old;
        Eigen::VectorXd grad_L_new = s.g;

        if(s.n_eq + s.n_ineq > 0 && qp_res.lambda.size() > 0)
        {
            // Extract multipliers: first m_eq are equality, rest are inequality
            // Note: qp_res.lambda includes box constraint multipliers at the end
            // We only use the first n_eq + n_ineq entries for the Lagrangian
            int m_total = s.n_eq + s.n_ineq;
            Eigen::VectorXd lam = qp_res.lambda.head(
                std::min(m_total, static_cast<int>(qp_res.lambda.size())));

            // Update stored multipliers
            if(lam.size() == m_total)
                s.lambda = lam;

            if(s.n_eq > 0 && lam.size() >= s.n_eq)
            {
                Eigen::VectorXd lam_eq = lam.head(s.n_eq);

                // Recompute old Jacobian at old point for grad_L_old
                Eigen::MatrixXd J_eq_old, J_ineq_old;
                s.eval_jacobian(x_old, J_eq_old, J_ineq_old);
                grad_L_old -= J_eq_old.transpose() * lam_eq;
                grad_L_new -= s.J_eq.transpose() * lam_eq;
            }

            if(s.n_ineq > 0 && lam.size() >= s.n_eq + s.n_ineq)
            {
                Eigen::VectorXd lam_ineq = lam.segment(s.n_eq, s.n_ineq);

                Eigen::MatrixXd J_eq_old, J_ineq_old;
                s.eval_jacobian(x_old, J_eq_old, J_ineq_old);
                grad_L_old -= J_ineq_old.transpose() * lam_ineq;
                grad_L_new -= s.J_ineq.transpose() * lam_ineq;
            }
        }

        // Curvature pair for L-BFGS
        Eigen::VectorXd sk = s.x - x_old;
        Eigen::VectorXd yk = grad_L_new - grad_L_old;
        s.lbfgs.push(sk, yk);

        ++s.iteration;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.g.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
        };
    }

    // Hot start -- preserves L-BFGS history.
    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        if(s.n_eq + s.n_ineq > 0)
        {
            s.eval_constraints(x0, s.c_eq, s.c_ineq);
            s.eval_jacobian(x0, s.J_eq, s.J_ineq);
        }
        s.iteration = 0;
    }

    // Cold restart -- clears L-BFGS curvature history.
    void reset_clear(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
        s.lbfgs.reset();
        s.sigma = 1.0;
    }
};

}

#endif
