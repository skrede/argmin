#ifndef HPP_GUARD_NABLAPP_SOLVER_COBYLA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_COBYLA_POLICY_H

// COBYLA solver policy for basic_solver.
//
// Implements Powell's Constrained Optimization BY Linear Approximation.
// A trust-region derivative-free method that maintains a simplex of n+1
// points and builds linear models of the objective and all constraints
// by interpolation. Each step solves a linearised trust-region subproblem,
// evaluates the trial point, updates the simplex, and adjusts the trust
// radius based on model accuracy.
//
// Requires: objective<P,S> && constrained_values<P,S> && bound_constrained<P,S>.
// No gradient or constraint Jacobian is needed -- COBYLA uses only function
// and constraint value evaluations.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization
//            method that models the objective and constraint functions
//            by linear interpolation."
//            K&W 2e, Section 10.7 (derivative-free constrained optimization).

#include "nablapp/detail/cobyla_simplex.h"
#include "nablapp/detail/cobyla_trust_region.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace nablapp
{

struct cobyla_policy
{
    using scalar_type = double;

    struct options_type
    {
        double initial_trust_radius{0.0};   // 0 means auto (10% of max bound range)
        double final_trust_radius{1e-8};
        double step_convergence_factor{1e-3};
    };

    options_type options{};

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::MatrixXd simplex;       // n x (n+1)
        Eigen::VectorXd f_simplex;     // objective at each vertex
        Eigen::MatrixXd c_simplex;     // all constraints at each vertex, m x (n+1)
        double rho{};
        double rho_end{};
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int m_total{0};
        int best_idx{0};

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_constraints;
    };

    template <typename Problem>
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& opts, const options_type& policy_opts)
    {
        self.options = policy_opts;
        return self.init(problem, x0, opts);
    }

    template <typename Problem>
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& /*opts*/)
    {
        const int n = problem.dimension();
        state_type s;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();
        s.x = detail::project(x0, s.lower, s.upper);

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        s.m_total = s.n_eq + s.n_ineq;

        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };

        s.eval_constraints = [&problem](const Eigen::VectorXd& v, Eigen::VectorXd& c) {
            problem.constraints(v, c);
        };

        // Initial trust radius (same auto logic as BOBYQA)
        double h = self.options.initial_trust_radius;
        if(h <= 0.0)
        {
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper[i] - s.lower[i];
                if(std::isfinite(range_i))
                    max_range = std::max(max_range, range_i);
            }
            h = (max_range > 0.0) ? 0.1 * max_range : 1.0;
        }
        s.rho = h;
        s.rho_end = self.options.final_trust_radius;

        // Build initial simplex
        s.simplex = detail::build_simplex(s.x, s.rho, s.lower, s.upper);

        // Evaluate objective and constraints at all simplex vertices
        s.f_simplex.resize(n + 1);
        s.c_simplex.resize(s.m_total, n + 1);

        for(int i = 0; i <= n; ++i)
        {
            s.f_simplex[i] = s.eval_value(s.simplex.col(i));
            if(s.m_total > 0)
            {
                Eigen::VectorXd c(s.m_total);
                s.eval_constraints(s.simplex.col(i), c);
                s.c_simplex.col(i) = c;
            }
        }

        // Find best point using a merit combining objective and feasibility
        s.best_idx = find_best_point(s);
        s.x = s.simplex.col(s.best_idx);
        s.objective_value = s.f_simplex[s.best_idx];

        // Populate c_eq / c_ineq for basic_solver constraint_violation
        populate_constraint_state(s);

        s.iteration = 0;
        return s;
    }

    step_result<double> step(this auto&& self, state_type& s)
    {
        const int n = static_cast<int>(s.x.size());
        double old_f = s.objective_value;

        // Build linear models from current simplex
        auto models = detail::build_linear_models(
            s.simplex, s.f_simplex, s.c_simplex, s.best_idx);

        // Compute linearised constraint offsets at best point
        Eigen::VectorXd constraint_offsets = compute_constraint_offsets(s);

        // Solve trust-region subproblem
        Eigen::VectorXd d = detail::solve_linear_subproblem(
            models.objective_gradient,
            models.constraint_gradients,
            constraint_offsets,
            s.n_eq,
            s.rho,
            s.lower,
            s.upper,
            s.x);

        double d_norm = d.norm();

        // Convergence: step too small relative to trust radius
        if(d_norm < s.rho * self.options.step_convergence_factor &&
           s.rho <= s.rho_end)
        {
            return make_converged_result(s, models.objective_gradient.norm());
        }

        // Trial point
        Eigen::VectorXd x_trial = detail::project(
            (s.x + d).eval(), s.lower, s.upper);
        double f_trial = s.eval_value(x_trial);

        Eigen::VectorXd c_trial(s.m_total);
        if(s.m_total > 0)
            s.eval_constraints(x_trial, c_trial);

        // Predicted vs actual reduction (using merit: f + penalty * violation)
        double penalty = 10.0;
        double viol_old = current_violation(s);
        double viol_trial = compute_violation(c_trial, s.n_eq, s.n_ineq);

        double actual = (old_f + penalty * viol_old) - (f_trial + penalty * viol_trial);
        double predicted = -models.objective_gradient.dot(d) +
                           penalty * (viol_old - linearised_violation_at(
                               models.constraint_gradients, constraint_offsets,
                               s.n_eq, d));

        // Update trust radius
        s.rho = detail::compute_rho_update(s.rho, s.rho_end, actual, predicted);

        // Select vertex to replace and update simplex
        int k = detail::select_replacement_vertex(
            s.simplex, s.f_simplex, s.best_idx, x_trial);
        detail::replace_vertex(s.simplex, s.f_simplex, s.c_simplex, k, x_trial, f_trial, c_trial);

        // Update best point
        bool improved = false;
        int new_best = find_best_point(s);
        if(new_best != s.best_idx || f_trial < old_f)
        {
            s.best_idx = new_best;
            s.x = s.simplex.col(s.best_idx);
            s.objective_value = s.f_simplex[s.best_idx];
            improved = (s.objective_value < old_f);
        }

        // Geometry maintenance
        int geo_idx = detail::check_simplex_geometry(s.simplex, s.best_idx, s.rho);
        if(geo_idx >= 0)
            repair_geometry(s, geo_idx);

        populate_constraint_state(s);
        ++s.iteration;

        // Derivative-free convergence signalling (same pattern as BOBYQA)
        double obj_change = s.objective_value - old_f;
        double effective_step = improved ? d_norm : s.rho;
        double effective_change = improved ? obj_change : s.rho;

        double grad_proxy = models.objective_gradient.norm();
        if(s.rho > s.rho_end)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
        };
    }

    void reset(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset_clear(s, x0);
    }

    void reset_clear(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        const int n = static_cast<int>(x0.size());
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;

        s.simplex = detail::build_simplex(s.x, s.rho, s.lower, s.upper);

        for(int i = 0; i <= n; ++i)
        {
            s.f_simplex[i] = s.eval_value(s.simplex.col(i));
            if(s.m_total > 0)
            {
                Eigen::VectorXd c(s.m_total);
                s.eval_constraints(s.simplex.col(i), c);
                s.c_simplex.col(i) = c;
            }
        }

        s.best_idx = find_best_point(s);
        s.x = s.simplex.col(s.best_idx);
        s.objective_value = s.f_simplex[s.best_idx];
        populate_constraint_state(s);
    }

private:

    // Find the best simplex vertex, preferring feasible points.
    static int find_best_point(const state_type& s)
    {
        const int nv = static_cast<int>(s.f_simplex.size());
        int best = 0;
        double best_merit = std::numeric_limits<double>::infinity();

        for(int i = 0; i < nv; ++i)
        {
            double viol = 0.0;
            if(s.m_total > 0)
                viol = compute_violation(s.c_simplex.col(i), s.n_eq, s.n_ineq);

            // Feasibility-first: feasible beats infeasible regardless of objective
            double merit = s.f_simplex[i] + 1e6 * viol;
            if(merit < best_merit)
            {
                best_merit = merit;
                best = i;
            }
        }
        return best;
    }

    // Compute constraint violation from a combined constraint vector.
    static double compute_violation(const Eigen::VectorXd& c, int n_eq, int n_ineq)
    {
        Eigen::VectorXd ceq = c.head(n_eq);
        Eigen::VectorXd cineq = c.tail(n_ineq);
        return detail::constraint_violation(ceq, cineq);
    }

    // Current violation at the best point.
    static double current_violation(const state_type& s)
    {
        return detail::constraint_violation(s.c_eq, s.c_ineq);
    }

    // Compute linearised constraint offsets (constraint values at best point).
    static Eigen::VectorXd compute_constraint_offsets(const state_type& s)
    {
        if(s.m_total == 0)
            return Eigen::VectorXd{};
        return s.c_simplex.col(s.best_idx);
    }

    // Compute linearised violation at displacement d.
    static double linearised_violation_at(
        const Eigen::MatrixXd& constraint_gradients,
        const Eigen::VectorXd& constraint_offsets,
        int n_eq,
        const Eigen::VectorXd& d)
    {
        return detail::linearised_violation(constraint_gradients, constraint_offsets, n_eq, d);
    }

    // Populate c_eq and c_ineq from the best point's constraint values.
    static void populate_constraint_state(state_type& s)
    {
        if(s.m_total > 0)
        {
            Eigen::VectorXd c = s.c_simplex.col(s.best_idx);
            s.c_eq = c.head(s.n_eq);
            s.c_ineq = c.tail(s.n_ineq);
        }
        else
        {
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
        }
    }

    // Repair degenerate simplex geometry at vertex geo_idx.
    static void repair_geometry(state_type& s, int geo_idx)
    {
        Eigen::VectorXd geo_pt = detail::geometry_improving_point(
            s.simplex, s.best_idx, geo_idx, s.rho, s.lower, s.upper);
        double geo_f = s.eval_value(geo_pt);

        Eigen::VectorXd geo_c(s.m_total);
        if(s.m_total > 0)
            s.eval_constraints(geo_pt, geo_c);

        detail::replace_vertex(s.simplex, s.f_simplex, s.c_simplex,
                               geo_idx, geo_pt, geo_f, geo_c);
    }

    static step_result<double> make_converged_result(const state_type& s, double grad_norm)
    {
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_norm,
            .step_size = 0.0,
            .objective_change = 0.0,
            .improved = false,
            .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
        };
    }
};

}

#endif
