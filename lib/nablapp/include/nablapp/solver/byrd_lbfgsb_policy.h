#ifndef HPP_GUARD_NABLAPP_SOLVER_BYRD_LBFGSB_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_BYRD_LBFGSB_POLICY_H

// Byrd 1995 variant L-BFGS-B solver policy for basic_solver.
//
// Structurally identical to lbfgsb_policy but with different defaults
// motivated by non-convex landscapes (cartan IK benchmarks):
//   - Armijo backtracking line search (instead of Strong Wolfe)
//   - 5-pair curvature history (instead of 10)
//
// The Armijo line search avoids the curvature condition that can
// over-constrain step acceptance on non-smooth or non-convex objectives.
// Shorter history (m=5) reduces the cost of each two-loop recursion
// and limits stale curvature influence.
//
// Reference: Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), pp. 1190-1208.
//            N&W Sections 9.2 (compact representation), 16.6 (GCP +
//            subspace minimization).

#include "nablapp/solver/lbfgsb_policy.h"

namespace nablapp
{

struct byrd_lbfgsb_policy
{
    using scalar_type = double;

    struct options_type
    {
        int history_depth{5};
        lbfgsb_line_search line_search_type{lbfgsb_line_search::armijo};
    };

    options_type options{};

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::VectorXd g;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        double objective_value{};
        detail::compact_lbfgs<double> B;
        int iteration{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_gradient;
    };

    template <typename Problem>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& opts, const options_type& policy_opts)
    {
        self.options = policy_opts;
        return self.init(problem, x0, opts);
    }

    template <typename Problem>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& opts)
    {
        const int n = problem.dimension();
        state_type s;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

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

        s.B = detail::compact_lbfgs<double>{self.options.history_depth};
        s.iteration = 0;

        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::VectorXd& v, Eigen::VectorXd& grad) {
            problem.gradient(v, grad);
        };

        return s;
    }

    step_result<double> step(this auto&& self, state_type& s)
    {
        if(s.iteration != 0)
            s.eval_gradient(s.x, s.g);

        auto gcp = detail::cauchy_point(s.x, s.g, s.lower, s.upper, s.B);

        Eigen::VectorXd x_new = detail::subspace_minimize(
            s.x, gcp.x_cauchy, s.g, s.lower, s.upper, gcp.free_indices, s.B);

        Eigen::VectorXd d = (x_new - s.x).eval();

        if(d.norm() < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
            };
        }

        double alpha_max = detail::compute_alpha_max(s.x, d, s.lower, s.upper);

        if(alpha_max < 1e-15)
        {
            d = (gcp.x_cauchy - s.x).eval();
            if(d.norm() < 1e-15)
            {
                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = s.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                };
            }
            alpha_max = detail::compute_alpha_max(s.x, d, s.lower, s.upper);
            if(alpha_max < 1e-15)
            {
                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = s.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                };
            }
        }

        auto phi = [&](double a) {
            return s.eval_value((s.x + a * d).eval());
        };
        auto dphi = [&](double a) {
            Eigen::VectorXd g_temp(s.x.size());
            s.eval_gradient((s.x + a * d).eval(), g_temp);
            return g_temp.dot(d);
        };

        line_search_options<double> ls_opts{.max_alpha = std::min(1.0, alpha_max)};
        double dphi0 = s.g.dot(d);
        line_search_result<double> ls;
        if(self.options.line_search_type == lbfgsb_line_search::armijo)
            ls = armijo(phi, s.objective_value, dphi0, ls_opts);
        else
            ls = strong_wolfe(phi, dphi, s.objective_value, dphi0, ls_opts);

        Eigen::VectorXd x_old = s.x;
        double old_f = s.objective_value;
        s.x = detail::project((s.x + ls.alpha * d).eval(), s.lower, s.upper);

        s.objective_value = s.eval_value(s.x);
        Eigen::VectorXd new_g(s.x.size());
        s.eval_gradient(s.x, new_g);

        Eigen::VectorXd sk = (s.x - x_old).eval();
        Eigen::VectorXd yk = (new_g - s.g).eval();
        s.B.push(sk, yk);

        s.g = new_g;
        ++s.iteration;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = new_g.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        s.iteration = 0;
    }

    void reset_clear(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
        s.B.reset();
    }
};

}

#endif
