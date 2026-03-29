#ifndef HPP_GUARD_NABLAPP_SOLVER_BOBYQA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_BOBYQA_POLICY_H

// BOBYQA solver policy for basic_solver.
//
// Implements Powell's Bound Optimization BY Quadratic Approximation.
// A trust-region derivative-free method that maintains a quadratic
// interpolation model Q(x) interpolating f at m points (default m = 2n+1).
// Each step solves the trust-region subproblem min Q(x_k + d) subject to
// ||d|| <= delta and box constraints, then updates the model and radius
// based on the accuracy ratio rho.
//
// Requires: objective<P,S> && bound_constrained<P,S> (D-08).
// No gradient is needed -- BOBYQA uses only objective evaluations.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.
//            K&W Section 8.4 (surrogate model framework).

#include "nablapp/detail/quadratic_model.h"
#include "nablapp/detail/trust_region.h"
#include "nablapp/detail/bound_projection.h"
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

struct bobyqa_policy
{
    using scalar_type = double;

    struct options_type
    {
        int num_interpolation_points{0};   // 0 means 2n+1 default (D-06)
        double initial_trust_radius{0.0};  // 0 means auto (10% of max bound range)
        double final_trust_radius{1e-8};   // stopping criterion on delta
    };

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        double objective_value{};
        Eigen::MatrixXd Y;
        Eigen::VectorXd f_values;
        detail::quadratic_model<double> model;
        double delta{};
        double delta_max{};
        int iteration{0};
        int m{};
        bool initialized{false};

        std::function<double(const Eigen::VectorXd&)> eval_value;
    };

    template <typename Problem>
        requires objective<Problem> && bound_constrained<Problem>
    state_type init(this auto&&, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<double>& opts)
    {
        const int n = problem.dimension();
        state_type s;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Project x0 to feasible region
        s.x = detail::project(x0, s.lower, s.upper);

        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };

        // Number of interpolation points (D-06)
        options_type policy_opts;
        s.m = (policy_opts.num_interpolation_points > 0)
                  ? policy_opts.num_interpolation_points
                  : 2 * n + 1;

        // Initial trust-region radius
        double h = policy_opts.initial_trust_radius;
        if(h <= 0.0)
        {
            // Auto: 10% of the maximum bound range
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper[i] - s.lower[i];
                if(std::isfinite(range_i))
                    max_range = std::max(max_range, range_i);
            }
            h = (max_range > 0.0) ? 0.1 * max_range : 1.0;
        }
        s.delta = h;
        s.delta_max = 10.0 * h;

        // Build initial interpolation set
        // Y[:,0] = x0 (projected), Y[:,i] = x0 + h*e_i, Y[:,n+i] = x0 - h*e_i
        s.Y.resize(n, s.m);
        s.f_values.resize(s.m);

        s.Y.col(0) = s.x;
        s.f_values[0] = s.eval_value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::VectorXd pt = s.x;
            pt[i] = std::min(pt[i] + h, s.upper[i]);
            // If positive step hits the bound, try negative
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::max(s.x[i] - h, s.lower[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.eval_value(pt);
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::VectorXd pt = s.x;
            pt[i] = std::max(pt[i] - h, s.lower[i]);
            // If negative step hits the bound, try positive
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::min(s.x[i] + h, s.upper[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.eval_value(pt);
        }

        // Find best point
        int best = 0;
        for(int i = 1; i < s.m; ++i)
        {
            if(s.f_values[i] < s.f_values[best])
                best = i;
        }
        s.x = s.Y.col(best);
        s.objective_value = s.f_values[best];

        // Build initial quadratic model
        s.model = detail::build_model(s.Y, s.f_values, s.x);
        s.initialized = true;
        s.iteration = 0;

        return s;
    }

    step_result<double> step(this auto&&, state_type& s)
    {
        double old_f = s.objective_value;

        // Model gradient at current best point
        Eigen::VectorXd mg = detail::model_gradient(s.model, s.x);

        // Solve trust-region subproblem
        Eigen::VectorXd d = detail::solve_trust_region_box(
            mg, s.model.H, s.x, s.delta, s.lower, s.upper);

        double d_norm = d.norm();

        // Convergence check: step too small relative to final trust radius
        if(d_norm < s.delta * 1e-3 && s.delta < options_type{}.final_trust_radius * 10.0)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = mg.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
            };
        }

        // Trial point
        Eigen::VectorXd x_new = detail::project(
            (s.x + d).eval(), s.lower, s.upper);
        double f_new = s.eval_value(x_new);

        // Model predictions
        double q_old = detail::evaluate_model(s.model, s.x);
        double q_new = detail::evaluate_model(s.model, x_new);

        // Accuracy ratio
        double rho = detail::compute_rho(old_f, f_new, q_old, q_new);

        // Update trust-region radius
        s.delta = detail::update_radius(s.delta, rho, d_norm, s.delta_max);

        // Select point to replace in interpolation set
        int k = detail::select_replacement(
            s.Y, s.f_values, x_new, f_new, s.x);

        // Update interpolation set
        s.Y.col(k) = x_new;
        s.f_values[k] = f_new;

        // Accept step if sufficient improvement or strictly better
        bool improved = false;
        if(f_new < old_f)
        {
            s.x = x_new;
            s.objective_value = f_new;
            improved = true;
        }

        // Update model
        detail::update_model(s.model, s.Y, s.f_values, k, s.x);

        // Geometry check: if a point is too far, replace it
        int g_idx = detail::check_geometry(s.Y, s.x, s.delta);
        if(g_idx >= 0)
        {
            // Replace with a geometry-improving point midway between x_k and the far point
            Eigen::VectorXd geo_pt = detail::project(
                (0.5 * (s.x + s.Y.col(g_idx))).eval(), s.lower, s.upper);
            double geo_f = s.eval_value(geo_pt);
            s.Y.col(g_idx) = geo_pt;
            s.f_values[g_idx] = geo_f;
            detail::update_model(s.model, s.Y, s.f_values, g_idx, s.x);

            if(geo_f < s.objective_value)
            {
                s.x = geo_pt;
                s.objective_value = geo_f;
                improved = true;
            }
        }

        ++s.iteration;

        // Derivative-free convergence signalling:
        // - gradient_norm: use model gradient as proxy, but ensure it stays large
        //   enough to prevent basic_solver from triggering gradient convergence
        //   while the trust region is still active
        // - objective_change: when step is rejected (objective unchanged), report
        //   delta as a proxy to prevent ftol_reached from firing prematurely
        // - step_size: report delta when step is rejected, to prevent stall detection
        double obj_change = s.objective_value - old_f;
        double effective_step = improved ? d_norm : s.delta;
        double effective_change = improved ? obj_change : s.delta;

        double grad_proxy = mg.norm();
        if(s.delta > options_type{}.final_trust_radius)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
        };
    }

    // Cold restart -- BOBYQA has no warm-start mode since the interpolation
    // set is point-specific.
    void reset(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        // BOBYQA cannot warm-start; delegate to full reset
        self.reset_clear(s, x0);
    }

    void reset_clear(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        const int n = x0.size();
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;
        s.initialized = false;

        double h = s.delta;

        // Rebuild interpolation set from new starting point
        s.Y.col(0) = s.x;
        s.f_values[0] = s.eval_value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::VectorXd pt = s.x;
            pt[i] = std::min(pt[i] + h, s.upper[i]);
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::max(s.x[i] - h, s.lower[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.eval_value(pt);
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::VectorXd pt = s.x;
            pt[i] = std::max(pt[i] - h, s.lower[i]);
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::min(s.x[i] + h, s.upper[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.eval_value(pt);
        }

        // Find best point
        int best = 0;
        for(int i = 1; i < s.m; ++i)
        {
            if(s.f_values[i] < s.f_values[best])
                best = i;
        }
        s.x = s.Y.col(best);
        s.objective_value = s.f_values[best];

        s.model = detail::build_model(s.Y, s.f_values, s.x);
        s.initialized = true;
    }
};

}

#endif
