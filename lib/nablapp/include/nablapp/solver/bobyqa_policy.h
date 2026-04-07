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
#include "nablapp/options/trust_region_options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace nablapp
{

template <int N = dynamic_dimension>
struct bobyqa_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = bobyqa_policy<M>;

    struct options_type
    {
        std::optional<std::uint16_t> num_interpolation_points{};  // default: 2n+1 (Powell 2009)
        std::optional<double> initial_trust_radius{};             // default: auto, 10% of max bound range (Powell 2009)
        std::optional<double> final_trust_radius{};               // default: 1e-8, stopping criterion on delta (Powell 2009)
        std::optional<double> step_convergence_factor{};          // default: 1e-3
        trust_region_options trust{};                              // Embedded trust region params
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        double objective_value{};
        Eigen::Matrix<double, N, Eigen::Dynamic> Y;
        Eigen::VectorXd f_values;
        detail::quadratic_model<double, N> model;
        double delta{};
        double delta_max{};
        double final_trust_radius{1e-8};
        double rho{};       // Current minimum trust radius (contracts toward rho_end)
        double rho_end{};   // Final rho target (= final_trust_radius)
        std::uint32_t iteration{0};
        std::uint16_t itest{0};  // Consecutive non-improving iterations (Powell 2009, Section 4)
        std::uint16_t rescue_counter{0};  // Consecutive rho contractions without improvement (Powell 2009, RESCUE)
        int m{};
        bool initialized{false};
    };

    template <typename Problem, typename Convergence>
        requires objective<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
        requires objective<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Project x0 to feasible region
        s.x = detail::project(x0, s.lower, s.upper);

        // Number of interpolation points (Powell 2009)
        s.m = options.num_interpolation_points.has_value()
                  ? static_cast<int>(options.num_interpolation_points.value())
                  : 2 * n + 1;

        // Final trust radius
        s.final_trust_radius = options.final_trust_radius.value_or(1e-8);

        // Initial trust-region radius (Powell 2009)
        double h = options.initial_trust_radius.value_or(0.0);
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
        s.f_values[0] = s.problem->value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x;
            pt[i] = std::min(pt[i] + h, s.upper[i]);
            // If positive step hits the bound, try negative
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::max(s.x[i] - h, s.lower[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.problem->value(pt);
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x;
            pt[i] = std::max(pt[i] - h, s.lower[i]);
            // If negative step hits the bound, try positive
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::min(s.x[i] + h, s.upper[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.problem->value(pt);
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

        // Powell 2009, Section 5: two-radius rho contraction scheme.
        // rho starts at the initial trust radius (rhobeg) and contracts toward rho_end.
        s.rho = s.delta;
        s.rho_end = s.final_trust_radius;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        double old_f = s.objective_value;
        double step_conv_factor = options.step_convergence_factor.value_or(1e-3);

        // Model gradient at current best point
        Eigen::Vector<double, N> mg = detail::model_gradient(s.model, s.x);

        // Solve trust-region subproblem
        Eigen::Vector<double, N> d = detail::solve_trust_region_box(
            mg, s.model.H, s.x, s.delta, s.lower, s.upper);

        double d_norm = d.norm();

        // Powell 2009, Section 5: convergence via rho contraction.
        // Only declare convergence when rho has contracted to rho_end and
        // delta is also exhausted. This ensures the solver goes through the
        // full rho contraction sequence before stopping.
        if(s.rho <= s.rho_end && s.delta <= s.rho_end)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = mg.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
            };
        }

        // Trial point
        Eigen::Vector<double, N> x_new = detail::project(
            (s.x + d).eval(), s.lower, s.upper);
        double f_new = s.problem->value(x_new);

        // Model predictions
        double q_old = detail::evaluate_model(s.model, s.x);
        double q_new = detail::evaluate_model(s.model, x_new);

        // Accuracy ratio
        double accuracy_ratio = detail::compute_rho(old_f, f_new, q_old, q_new);

        // Update trust-region radius (delta), passing embedded trust region options.
        s.delta = detail::update_radius(s.delta, accuracy_ratio, d_norm, s.delta_max,
                                        options.trust);

        // Powell 2009, Section 5: two-radius rho contraction.
        // When delta has shrunk to rho, contract rho by halving toward rho_end.
        // Then ensure delta >= rho so the solver continues with a meaningful radius.
        if(s.delta <= s.rho)
        {
            s.rho = std::max(s.rho * 0.5, s.rho_end);
            s.delta = std::max(s.delta, s.rho);
        }

        // Powell 2009, Section 4: Lagrange-based point replacement.
        // Compute Lagrange values at x_new and select the point k that
        // maximizes |L_k(x_new)| for best-conditioned interpolation update.
        Eigen::VectorXd lv_xnew = detail::compute_lagrange_at_point(
            s.Y, s.f_values, s.x, x_new);
        int k = detail::select_replacement(
            s.Y, s.f_values, x_new, f_new, s.x, lv_xnew);

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

        // Powell 2009, Section 4: itest model staleness counter.
        if(improved)
            s.itest = 0;
        else
            ++s.itest;

        // Update model
        detail::update_model(s.model, s.Y, s.f_values, k, s.x);

        // Powell 2009, Section 4: when itest >= 3 and NPT > 2N+1, zero H
        // to force minimum-Frobenius-norm model. For default NPT = 2N+1,
        // the SVD already produces minimum-norm, so this is a no-op.
        if(s.itest >= 3)
        {
            const int n = s.x.size();
            if(s.m > 2 * n + 1)
                s.model.H.setZero();
            s.itest = 0;
        }

        // Powell 2009, Section 6 (ALTMOV concept): geometry improvement.
        // When the trust-region step is short and rho > rho_end, find the
        // interpolation point k_geo with largest |L_k(x_opt)| and replace
        // it with an objective evaluation at a geometry-improving point.
        if(d_norm < 0.5 * s.rho && s.rho > s.rho_end)
        {
            // Find point with largest |L_k(x_opt)| (excluding best)
            int best_idx = 0;
            for(int i = 1; i < s.m; ++i)
            {
                if(s.f_values[i] < s.f_values[best_idx])
                    best_idx = i;
            }

            int k_geo = (best_idx == 0) ? 1 : 0;
            double max_abs_lk = std::abs(s.model.lagrange_values[k_geo]);
            for(int i = 0; i < s.m; ++i)
            {
                if(i == best_idx) continue;
                double abs_lk = std::abs(s.model.lagrange_values[i]);
                if(abs_lk > max_abs_lk)
                {
                    max_abs_lk = abs_lk;
                    k_geo = i;
                }
            }

            // Geometry-improving point: move from x_opt toward Y[:,k_geo]
            // at distance rho, projected onto bounds.
            Eigen::Vector<double, N> dir = (s.Y.col(k_geo) - s.x).eval();
            double dir_norm = dir.norm();
            if(dir_norm > 1e-15)
            {
                Eigen::Vector<double, N> geo_pt = detail::project(
                    (s.x + s.rho * dir / dir_norm).eval(), s.lower, s.upper);
                double geo_f = s.problem->value(geo_pt);
                s.Y.col(k_geo) = geo_pt;
                s.f_values[k_geo] = geo_f;
                detail::update_model(s.model, s.Y, s.f_values, k_geo, s.x);

                if(geo_f < s.objective_value)
                {
                    s.x = geo_pt;
                    s.objective_value = geo_f;
                    improved = true;
                }
            }
        }

        // Powell 2009, RESCUE concept: full interpolation set rebuild.
        // Track consecutive rho contractions without objective improvement.
        // When rescue_counter >= 3, rebuild the entire interpolation set
        // around x_opt using current rho as the step size.
        if(s.delta <= s.rho && !improved)
            ++s.rescue_counter;
        else if(improved)
            s.rescue_counter = 0;

        if(s.rescue_counter >= 3)
        {
            const int n = s.x.size();
            double h = s.rho;

            s.Y.col(0) = s.x;
            s.f_values[0] = s.problem->value(s.x);

            for(int i = 0; i < n && (1 + i) < s.m; ++i)
            {
                Eigen::Vector<double, N> pt = s.x;
                pt[i] = std::min(pt[i] + h, s.upper[i]);
                if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                    pt[i] = std::max(s.x[i] - h, s.lower[i]);
                s.Y.col(1 + i) = pt;
                s.f_values[1 + i] = s.problem->value(pt);
            }

            for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
            {
                Eigen::Vector<double, N> pt = s.x;
                pt[i] = std::max(pt[i] - h, s.lower[i]);
                if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                    pt[i] = std::min(s.x[i] + h, s.upper[i]);
                s.Y.col(1 + n + i) = pt;
                s.f_values[1 + n + i] = s.problem->value(pt);
            }

            s.model = detail::build_model(s.Y, s.f_values, s.x);
            s.rescue_counter = 0;
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
        if(s.delta > s.final_trust_radius)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
            .x_norm = s.x.norm(),
        };
    }

    // Cold restart -- BOBYQA has no warm-start mode since the interpolation
    // set is point-specific.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        // BOBYQA cannot warm-start; delegate to full reset
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = x0.size();
        s.x = detail::project(x0, s.lower, s.upper);
        s.iteration = 0;
        s.initialized = false;

        double h = s.delta;

        // Rebuild interpolation set from new starting point
        s.Y.col(0) = s.x;
        s.f_values[0] = s.problem->value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x;
            pt[i] = std::min(pt[i] + h, s.upper[i]);
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::max(s.x[i] - h, s.lower[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.problem->value(pt);
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x;
            pt[i] = std::max(pt[i] - h, s.lower[i]);
            if(std::abs(pt[i] - s.x[i]) < 1e-15 * h)
                pt[i] = std::min(s.x[i] + h, s.upper[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.problem->value(pt);
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
