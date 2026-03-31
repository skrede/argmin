#ifndef HPP_GUARD_NABLAPP_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H

// Augmented Lagrangian (method of multipliers) wrapper policy.
//
// Converts any unconstrained or bound-constrained inner policy (e.g.,
// lbfgsb_policy, bobyqa_policy) into a constrained solver. Each outer
// step() performs a full inner solve on the augmented Lagrangian
// subproblem followed by a multiplier/penalty update. The inner
// subproblem satisfies differentiable + bound_constrained so that
// gradient-based and derivative-free inner solvers both work.
//
// Reference: K&W Section 10.9, Algorithm 10.2;
//            N&W Section 17.4, Algorithm 17.4.

#include "nablapp/detail/augmented_lagrangian.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/options.h"
#include "nablapp/result/step_result.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace nablapp
{

template <typename InnerPolicy = lbfgsb_policy>
struct augmented_lagrangian_policy
{
    using scalar_type = typename InnerPolicy::scalar_type;

    struct options_type
    {
        scalar_type mu_init = scalar_type(1);
        scalar_type mu_decrease = scalar_type(0.1);
        scalar_type mu_min = scalar_type(1e-12);
        int inner_max_iter = 200;
        int max_outer_iter = 50;
        scalar_type constraint_tol = scalar_type(1e-6);
        scalar_type inner_grad_tol = scalar_type(1e-6);
        typename InnerPolicy::options_type inner_opts = {};
    };

    struct state_type
    {
        Eigen::VectorX<scalar_type> x;
        scalar_type f{};
        Eigen::VectorX<scalar_type> c_eq;
        Eigen::VectorX<scalar_type> c_ineq;

        Eigen::VectorX<scalar_type> lambda_eq;
        Eigen::VectorX<scalar_type> lambda_ineq;

        scalar_type mu{};
        scalar_type prev_viol{};

        std::function<scalar_type(const Eigen::VectorX<scalar_type>&)> eval_value;
        std::function<void(const Eigen::VectorX<scalar_type>&,
                           Eigen::VectorX<scalar_type>&)> eval_gradient;
        std::function<void(const Eigen::VectorX<scalar_type>&,
                           Eigen::VectorX<scalar_type>&,
                           Eigen::VectorX<scalar_type>&)> eval_constraints;
        std::function<void(const Eigen::VectorX<scalar_type>&,
                           Eigen::MatrixX<scalar_type>&,
                           Eigen::MatrixX<scalar_type>&)> eval_jacobian;

        Eigen::VectorX<scalar_type> lower;
        Eigen::VectorX<scalar_type> upper;

        int n_eq = 0;
        int n_ineq = 0;
        int outer_iter = 0;

        options_type opts;
    };

    template <typename Problem>
    state_type init(this auto&& self, const Problem& problem,
                    const Eigen::VectorX<scalar_type>& x0,
                    const solver_options<scalar_type>& solver_opts,
                    const options_type& policy_opts)
    {
        auto s = self.init(problem, x0, solver_opts);
        s.opts = policy_opts;
        s.mu = policy_opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        return s;
    }

    template <typename Problem>
    state_type init(this auto&&, const Problem& problem,
                    const Eigen::VectorX<scalar_type>& x0,
                    const solver_options<scalar_type>& /*solver_opts*/)
    {
        static_assert(constrained<Problem, scalar_type>,
                      "augmented_lagrangian_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type s;
        s.opts = options_type{};

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();

        s.x = x0;
        s.f = problem.value(x0);

        // Evaluate constraints
        const int m = s.n_eq + s.n_ineq;
        Eigen::VectorX<scalar_type> c_all(m);
        if(m > 0)
            problem.constraints(x0, c_all);
        s.c_eq = c_all.head(s.n_eq);
        s.c_ineq = c_all.tail(s.n_ineq);

        // Initialize multipliers to zero
        s.lambda_eq = Eigen::VectorX<scalar_type>::Zero(s.n_eq);
        s.lambda_ineq = Eigen::VectorX<scalar_type>::Zero(s.n_ineq);

        // Penalty
        s.mu = s.opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.outer_iter = 0;

        // Box bounds
        if constexpr(bound_constrained<Problem, scalar_type>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr scalar_type inf = std::numeric_limits<scalar_type>::infinity();
            s.lower = Eigen::VectorX<scalar_type>::Constant(n, -inf);
            s.upper = Eigen::VectorX<scalar_type>::Constant(n, inf);
        }

        // Closures capturing problem by const reference
        s.eval_value = [&problem](const Eigen::VectorX<scalar_type>& v) {
            return problem.value(v);
        };

        if constexpr(differentiable<Problem, scalar_type>)
        {
            s.eval_gradient = [&problem](const Eigen::VectorX<scalar_type>& v,
                                         Eigen::VectorX<scalar_type>& grad) {
                problem.gradient(v, grad);
            };
        }

        const int n_eq_cap = s.n_eq;
        const int n_ineq_cap = s.n_ineq;

        s.eval_constraints = [&problem, n_eq_cap, n_ineq_cap](
            const Eigen::VectorX<scalar_type>& v,
            Eigen::VectorX<scalar_type>& ceq,
            Eigen::VectorX<scalar_type>& cineq)
        {
            const int mtot = n_eq_cap + n_ineq_cap;
            Eigen::VectorX<scalar_type> c(mtot);
            if(mtot > 0)
                problem.constraints(v, c);
            ceq = c.head(n_eq_cap);
            cineq = c.tail(n_ineq_cap);
        };

        s.eval_jacobian = [&problem, n_eq_cap, n_ineq_cap](
            const Eigen::VectorX<scalar_type>& v,
            Eigen::MatrixX<scalar_type>& Jeq,
            Eigen::MatrixX<scalar_type>& Jineq)
        {
            const int mtot = n_eq_cap + n_ineq_cap;
            const int ndim = v.size();
            Eigen::MatrixX<scalar_type> J(mtot, ndim);
            if(mtot > 0)
                problem.constraint_jacobian(v, J);
            Jeq = J.topRows(n_eq_cap);
            Jineq = J.bottomRows(n_ineq_cap);
        };

        return s;
    }

    step_result<scalar_type> step(this auto&&, state_type& s)
    {
        const int n = static_cast<int>(s.x.size());

        // Evaluate constraints and Jacobian at current x
        s.eval_constraints(s.x, s.c_eq, s.c_ineq);
        Eigen::MatrixX<scalar_type> J_eq, J_ineq;
        s.eval_jacobian(s.x, J_eq, J_ineq);

        // Capture state for the inner subproblem. These references are
        // safe because the inner solver is local to this function.
        const auto& lambda_eq = s.lambda_eq;
        const auto& lambda_ineq = s.lambda_ineq;
        const scalar_type mu = s.mu;
        const auto& eval_value = s.eval_value;
        const auto& eval_gradient = s.eval_gradient;
        const auto& eval_constraints = s.eval_constraints;
        const auto& eval_jacobian = s.eval_jacobian;
        const int n_eq = s.n_eq;
        const int n_ineq = s.n_ineq;

        // Synthetic subproblem struct for the inner solver.
        // Satisfies differentiable + bound_constrained.
        struct subproblem
        {
            int dim;
            const Eigen::VectorX<scalar_type>& lo;
            const Eigen::VectorX<scalar_type>& hi;

            const Eigen::VectorX<scalar_type>& lam_eq;
            const Eigen::VectorX<scalar_type>& lam_ineq;
            scalar_type pen;
            int neq;
            int nineq;

            const std::function<scalar_type(const Eigen::VectorX<scalar_type>&)>& ev_value;
            const std::function<void(const Eigen::VectorX<scalar_type>&,
                                     Eigen::VectorX<scalar_type>&)>& ev_gradient;
            const std::function<void(const Eigen::VectorX<scalar_type>&,
                                     Eigen::VectorX<scalar_type>&,
                                     Eigen::VectorX<scalar_type>&)>& ev_constraints;
            const std::function<void(const Eigen::VectorX<scalar_type>&,
                                     Eigen::MatrixX<scalar_type>&,
                                     Eigen::MatrixX<scalar_type>&)>& ev_jacobian;

            int dimension() const { return dim; }

            scalar_type value(const Eigen::VectorX<scalar_type>& x) const
            {
                scalar_type fval = ev_value(x);
                Eigen::VectorX<scalar_type> ceq, cineq;
                ev_constraints(x, ceq, cineq);
                return detail::augmented_lagrangian_value(
                    fval, ceq, cineq, lam_eq, lam_ineq, pen);
            }

            void gradient(const Eigen::VectorX<scalar_type>& x,
                          Eigen::VectorX<scalar_type>& g) const
            {
                ev_gradient(x, g);
                Eigen::VectorX<scalar_type> ceq, cineq;
                Eigen::MatrixX<scalar_type> Jeq, Jineq;
                ev_constraints(x, ceq, cineq);
                ev_jacobian(x, Jeq, Jineq);
                g = detail::augmented_lagrangian_gradient(
                    g, Jeq, Jineq, ceq, cineq, lam_eq, lam_ineq, pen);
            }

            Eigen::VectorX<scalar_type> lower_bounds() const { return lo; }
            Eigen::VectorX<scalar_type> upper_bounds() const { return hi; }
        };

        subproblem sub{
            .dim = n,
            .lo = s.lower,
            .hi = s.upper,
            .lam_eq = lambda_eq,
            .lam_ineq = lambda_ineq,
            .pen = mu,
            .neq = n_eq,
            .nineq = n_ineq,
            .ev_value = eval_value,
            .ev_gradient = eval_gradient,
            .ev_constraints = eval_constraints,
            .ev_jacobian = eval_jacobian,
        };

        // Inner solver: solve the augmented Lagrangian subproblem
        solver_options<scalar_type> inner_opts;
        inner_opts.max_iterations = s.opts.inner_max_iter;
        inner_opts.gradient_tolerance = s.opts.inner_grad_tol;
        inner_opts.objective_tolerance = scalar_type(1e-15);
        inner_opts.step_tolerance = scalar_type(1e-15);

        basic_solver<InnerPolicy> inner_solver{sub, s.x, inner_opts};
        auto inner_result = inner_solver.solve();

        // Extract solution from inner solver
        scalar_type f_old = s.f;
        Eigen::VectorX<scalar_type> x_old = s.x;
        s.x = inner_result.x;
        s.f = eval_value(s.x);
        eval_constraints(s.x, s.c_eq, s.c_ineq);

        // Constraint violation
        scalar_type viol = detail::constraint_violation(s.c_eq, s.c_ineq);

        // K&W Algorithm 10.2 / N&W Algorithm 17.4:
        // Always update multipliers. If violation is not decreasing
        // sufficiently, also decrease mu (increase penalty).
        detail::update_multipliers(
            s.lambda_eq, s.lambda_ineq, s.c_eq, s.c_ineq, s.mu);

        if(viol > scalar_type(0.25) * s.prev_viol)
        {
            // Insufficient feasibility progress: decrease mu
            s.mu = std::max(s.mu * s.opts.mu_decrease, s.opts.mu_min);
        }
        s.prev_viol = viol;

        ++s.outer_iter;

        // Use the max of constraint violation and step change as
        // gradient_norm proxy. This prevents basic_solver from stopping
        // early when constraints happen to be satisfied at a non-optimal
        // feasible point. The outer loop must continue until both
        // feasibility and optimality are achieved.
        scalar_type step_norm = (s.x - x_old).norm();
        scalar_type grad_proxy = std::max(viol, step_norm);

        return step_result<scalar_type>{
            .objective_value = s.f,
            .gradient_norm = grad_proxy,
            .step_size = step_norm,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old || viol < s.opts.constraint_tol,
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorX<scalar_type>& x0)
    {
        s.x = x0;
        s.f = s.eval_value(x0);
        s.eval_constraints(x0, s.c_eq, s.c_ineq);
        s.outer_iter = 0;
        // Preserve multipliers and penalty (hot start)
    }

    void reset_clear(this auto&& self, state_type& s,
                     const Eigen::VectorX<scalar_type>& x0)
    {
        self.reset(s, x0);
        s.lambda_eq.setZero();
        s.lambda_ineq.setZero();
        s.mu = s.opts.mu_init;
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
    }
};

}

#endif
