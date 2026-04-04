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
#include "nablapp/options/qp_options.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace nablapp
{

template <int N = dynamic_dimension>
struct kraft_slsqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = kraft_slsqp_policy<M>;

    struct options_type
    {
        std::optional<std::uint8_t> history_depth{};   // L-BFGS memory (default: 10, N&W 7.2)
        std::optional<double> initial_penalty{};       // penalty weight (default: 1.0, Kraft 1988)
        std::optional<double> penalty_growth{};        // penalty multiplier (default: 10.0, Kraft 1988)
        line_search_options line_search{};              // Embedded line search params
        qp_options qp{};                                // QP subproblem params
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
        detail::compact_lbfgs<double, N> lbfgs;
        detail::active_set_qp_solver<double, N> qp_solver;
        Eigen::Matrix<double, N, N> B_workspace;
        Eigen::Vector<double, N> ej_workspace;
        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int n{0};

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
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
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
            s.lower = Eigen::Vector<double, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(n, inf);
        }

        // Constraints
        if constexpr(constrained<Problem>)
        {
            s.n_eq = problem.num_equality();
            s.n_ineq = problem.num_inequality();

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
                const Eigen::Vector<double, N>& v, Eigen::VectorXd& ceq, Eigen::VectorXd& cineq) {
                Eigen::VectorXd c_all(n_eq + n_ineq);
                problem.constraints(v, c_all);
                ceq = c_all.head(n_eq);
                cineq = c_all.tail(n_ineq);
            };

            s.eval_jacobian = [&problem, n_eq = s.n_eq, n_ineq = s.n_ineq, n](
                const Eigen::Vector<double, N>& v, Eigen::MatrixXd& Jeq, Eigen::MatrixXd& Jineq) {
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

            s.eval_constraints = [](const Eigen::Vector<double, N>&, Eigen::VectorXd& ceq,
                                    Eigen::VectorXd& cineq) {
                ceq.resize(0);
                cineq.resize(0);
            };

            s.eval_jacobian = [n](const Eigen::Vector<double, N>&, Eigen::MatrixXd& Jeq,
                                  Eigen::MatrixXd& Jineq) {
                Jeq.resize(0, n);
                Jineq.resize(0, n);
            };
        }

        std::uint8_t depth = options.history_depth.value_or(10);
        s.lbfgs = detail::compact_lbfgs<double, N>{depth};
        s.sigma = options.initial_penalty.value_or(1.0);
        s.iteration = 0;

        // Pre-allocate QP solver workspace
        int max_constraints = s.n_eq + s.n_ineq + 2 * n;
        s.qp_solver = detail::active_set_qp_solver<double, N>(n, max_constraints);
        s.B_workspace.setZero(n, n);
        s.ej_workspace = Eigen::Vector<double, N>::Zero(n);

        s.eval_value = [&problem](const Eigen::Vector<double, N>& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::Vector<double, N>& v,
                                     Eigen::Vector<double, N>& grad) {
            problem.gradient(v, grad);
        };

        return s;
    }

    step_result<double> step(state_type& s)
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

        // Build QP Hessian from compact L-BFGS using pre-allocated workspace
        for(int j = 0; j < n; ++j)
        {
            s.ej_workspace[j] = 1.0;
            s.B_workspace.col(j) = s.lbfgs.multiply(s.ej_workspace);
            s.ej_workspace[j] = 0.0;
        }

        // Symmetrize for numerical safety
        s.B_workspace = (0.5 * (s.B_workspace + s.B_workspace.transpose())).eval();

        Eigen::MatrixXd A_eq = s.J_eq;
        Eigen::VectorXd b_eq = -s.c_eq;
        Eigen::MatrixXd A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        // Box bounds on the step
        Eigen::VectorXd p_lower = Eigen::VectorXd(s.lower) - Eigen::VectorXd(s.x);
        Eigen::VectorXd p_upper = Eigen::VectorXd(s.upper) - Eigen::VectorXd(s.x);

        constexpr double big = 1e20;
        for(int i = 0; i < n; ++i)
        {
            if(p_lower[i] < -big) p_lower[i] = -big;
            if(p_upper[i] > big) p_upper[i] = big;
        }

        // Solve QP with embedded options
        nablapp::qp_options qp_opts;
        qp_opts.max_iterations = options.qp.max_iterations.has_value()
            ? options.qp.max_iterations
            : std::optional<std::uint16_t>{static_cast<std::uint16_t>(10 * n)};
        qp_opts.tolerance = options.qp.tolerance.has_value()
            ? options.qp.tolerance
            : std::optional<double>{1e-12};
        Eigen::Vector<double, N> p_lo(p_lower);
        Eigen::Vector<double, N> p_hi(p_upper);
        Eigen::Vector<double, N> p_start = Eigen::Vector<double, N>::Zero(n);
        auto qp_res = s.qp_solver.solve(s.B_workspace, Eigen::Vector<double, N>(s.g),
                                         Eigen::Matrix<double, Eigen::Dynamic, N>(A_eq),
                                         b_eq,
                                         Eigen::Matrix<double, Eigen::Dynamic, N>(A_ineq),
                                         b_ineq, p_lo, p_hi, p_start, qp_opts);

        Eigen::Vector<double, N> p = qp_res.x;

        double p_norm = p.norm();
        if(p_norm < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
            };
        }

        // Update penalty parameter sigma
        if(qp_res.lambda.size() > 0)
        {
            double lambda_max = qp_res.lambda.cwiseAbs().maxCoeff();
            if(lambda_max + 0.5 > s.sigma)
                s.sigma = lambda_max + 1.0;
        }

        // L1 merit function for line search
        auto merit = [&](const Eigen::Vector<double, N>& xk) -> double {
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

        double constraint_viol_0 = 0.0;
        if(s.c_eq.size() > 0)
            constraint_viol_0 += s.c_eq.cwiseAbs().sum();
        if(s.c_ineq.size() > 0)
            constraint_viol_0 += (-s.c_ineq).cwiseMax(0.0).sum();

        double dphi_merit = s.g.dot(p) - s.sigma * constraint_viol_0;

        if(dphi_merit >= 0.0)
            dphi_merit = -1e-8;

        // Backtracking Armijo line search on L1 merit
        auto phi_ls = [&](double alpha) {
            Eigen::Vector<double, N> x_trial = s.x + alpha * p;
            for(int i = 0; i < n; ++i)
            {
                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
            }
            return merit(x_trial);
        };

        auto ls = armijo(phi_ls, merit_0, dphi_merit, options.line_search);
        double alpha = ls.alpha;

        if(alpha < 1e-15)
            alpha = 1e-4;

        // Update iterate
        Eigen::Vector<double, N> x_old = s.x;
        double old_f = s.objective_value;
        s.x = s.x + alpha * p;

        for(int i = 0; i < n; ++i)
        {
            if(s.x[i] < s.lower[i]) s.x[i] = s.lower[i];
            if(s.x[i] > s.upper[i]) s.x[i] = s.upper[i];
        }

        s.objective_value = s.eval_value(s.x);

        Eigen::Vector<double, N> g_old = s.g;
        s.eval_gradient(s.x, s.g);

        if(s.n_eq + s.n_ineq > 0)
        {
            s.eval_constraints(s.x, s.c_eq, s.c_ineq);
            s.eval_jacobian(s.x, s.J_eq, s.J_ineq);
        }

        // L-BFGS update using Lagrangian gradient
        Eigen::Vector<double, N> grad_L_old = g_old;
        Eigen::Vector<double, N> grad_L_new = s.g;

        if(s.n_eq + s.n_ineq > 0 && qp_res.lambda.size() > 0)
        {
            int m_total = s.n_eq + s.n_ineq;
            Eigen::VectorXd lam = qp_res.lambda.head(
                std::min(m_total, static_cast<int>(qp_res.lambda.size())));

            if(lam.size() == m_total)
                s.lambda = lam;

            if(s.n_eq > 0 && lam.size() >= s.n_eq)
            {
                Eigen::VectorXd lam_eq = lam.head(s.n_eq);

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

        Eigen::Vector<double, N> sk = s.x - x_old;
        Eigen::Vector<double, N> yk = grad_L_new - grad_L_old;
        s.lbfgs.push(sk, yk);

        ++s.iteration;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.g.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .x_norm = s.x.norm(),
        };
    }

    // Hot start -- preserves L-BFGS history.
    void reset(state_type& s, const Eigen::Vector<double, N>& x0)
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
    void reset_clear(state_type& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.lbfgs.reset();
        s.sigma = options.initial_penalty.value_or(1.0);
    }
};

}

#endif
