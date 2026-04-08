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
        Eigen::Vector<double, N> scale;          // scale[i] = normalized bound range (Powell 2009, PRELIM)
        Eigen::Vector<double, N> x_scaled;       // x in scaled space: x_scaled[i] = x[i] / scale[i]
        Eigen::Vector<double, N> lower_scaled;   // lower[i] / scale[i]
        Eigen::Vector<double, N> upper_scaled;   // upper[i] / scale[i]
        double objective_value{};
        Eigen::Matrix<double, N, Eigen::Dynamic> Y;  // Interpolation points in scaled coordinates
        Eigen::VectorXd f_values;
        detail::quadratic_model<double, N> model;     // Model in scaled coordinates
        double delta{};
        double delta_max{};
        double final_trust_radius{1e-8};
        double rho{};       // Current minimum trust radius (contracts toward rho_end)
        double rho_end{};   // Final rho target (= final_trust_radius)
        std::uint32_t iteration{0};
        std::uint16_t itest{0};  // Consecutive non-improving iterations (Powell 2009, Section 4)
        std::uint16_t rescue_counter{0};  // Consecutive rho contractions without improvement (Powell 2009, RESCUE)
        bool last_improved{false};       // Whether the previous step improved the objective
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

        // Variable rescaling (Powell 2009, PRELIM concept).
        // Compute scale factors from bound ranges so that the trust region
        // treats all variables equally regardless of their physical scales.
        // scale[i] = (upper[i] - lower[i]) when both bounds are finite, else 1.0.
        // Normalize by maxCoeff so the largest range maps to 1.0.
        s.scale.resize(n);
        for(int i = 0; i < n; ++i)
        {
            double range = s.upper[i] - s.lower[i];
            s.scale[i] = std::isfinite(range) ? range : 1.0;
        }
        double max_scale = s.scale.maxCoeff();
        if(max_scale > 0.0)
            s.scale /= max_scale;
        // Guard against zero-range dimensions (lower == upper)
        for(int i = 0; i < n; ++i)
        {
            if(s.scale[i] < 1e-15)
                s.scale[i] = 1.0;
        }

        // Transform bounds to scaled space
        s.lower_scaled = (s.lower.array() / s.scale.array()).matrix();
        s.upper_scaled = (s.upper.array() / s.scale.array()).matrix();

        // Project x0 to feasible region, then transform to scaled space
        s.x = detail::project(x0, s.lower, s.upper);
        s.x_scaled = (s.x.array() / s.scale.array()).matrix();

        // Number of interpolation points (Powell 2009)
        s.m = options.num_interpolation_points.has_value()
                  ? static_cast<int>(options.num_interpolation_points.value())
                  : 2 * n + 1;

        // Final trust radius
        s.final_trust_radius = options.final_trust_radius.value_or(1e-8);

        // Initial trust-region radius in SCALED space (Powell 2009).
        // In scaled space, all dimensions have comparable ranges, so a
        // uniform h makes the initial simplex well-conditioned.
        double h = options.initial_trust_radius.value_or(0.0);
        if(h <= 0.0)
        {
            // Auto: 10% of the maximum scaled bound range.
            // Since scaling normalizes the largest range to 1.0, max_range
            // in scaled space is approximately 1/max_scale * max_range_orig.
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper_scaled[i] - s.lower_scaled[i];
                if(std::isfinite(range_i))
                    max_range = std::max(max_range, range_i);
            }
            h = (max_range > 0.0) ? 0.1 * max_range : 1.0;
        }
        s.delta = h;
        s.delta_max = 10.0 * h;

        // Build initial interpolation set in SCALED coordinates.
        // The perturbation h is uniform in scaled space, meaning the actual
        // perturbation in original space is h * scale[i] for variable i.
        s.Y.resize(n, s.m);
        s.f_values.resize(s.m);

        s.Y.col(0) = s.x_scaled;
        s.f_values[0] = s.problem->value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x_scaled;
            pt[i] = std::min(pt[i] + h, s.upper_scaled[i]);
            if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                pt[i] = std::max(s.x_scaled[i] - h, s.lower_scaled[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.problem->value(
                (pt.array() * s.scale.array()).matrix());
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x_scaled;
            pt[i] = std::max(pt[i] - h, s.lower_scaled[i]);
            if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                pt[i] = std::min(s.x_scaled[i] + h, s.upper_scaled[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.problem->value(
                (pt.array() * s.scale.array()).matrix());
        }

        // Find best point
        int best = 0;
        for(int i = 1; i < s.m; ++i)
        {
            if(s.f_values[i] < s.f_values[best])
                best = i;
        }
        s.x_scaled = s.Y.col(best);
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.f_values[best];

        // Build initial quadratic model in scaled coordinates
        s.model = detail::build_model(s.Y, s.f_values, s.x_scaled);
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

        // All internal computation in scaled coordinates.
        // Model gradient at current best point (scaled space).
        Eigen::Vector<double, N> mg = detail::model_gradient(s.model, s.x_scaled);

        // Solve trust-region subproblem in scaled coordinates
        Eigen::Vector<double, N> d = detail::solve_trust_region_box(
            mg, s.model.H, s.x_scaled, s.delta, s.lower_scaled, s.upper_scaled);

        double d_norm = d.norm();

        // Powell 2009, Section 5: convergence via rho contraction.
        // Only terminate when the previous step was also non-improving,
        // preventing premature exit when the solver is still making progress.
        if(s.rho <= s.rho_end && s.delta <= s.rho_end && !s.last_improved)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = mg.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
                .policy_status = solver_status::converged,
            };
        }

        // Trial point in scaled coordinates, then unscale for objective evaluation
        Eigen::Vector<double, N> x_new_scaled = detail::project(
            (s.x_scaled + d).eval(), s.lower_scaled, s.upper_scaled);
        Eigen::Vector<double, N> x_new_orig =
            (x_new_scaled.array() * s.scale.array()).matrix();
        double f_new = s.problem->value(x_new_orig);

        // Model predictions (scaled coordinates)
        double q_old = detail::evaluate_model(s.model, s.x_scaled);
        double q_new = detail::evaluate_model(s.model, x_new_scaled);

        // Accuracy ratio
        double accuracy_ratio = detail::compute_rho(old_f, f_new, q_old, q_new);

        // Update trust-region radius (delta)
        s.delta = detail::update_radius(s.delta, accuracy_ratio, d_norm, s.delta_max,
                                        options.trust);

        // Powell 2009, Section 5: two-radius rho contraction.
        // Only contract rho when the solver has genuinely stalled at this
        // scale (non-improving step AND rescue_counter indicates repeated
        // failure), not from a single bad accuracy ratio shrinking delta.
        if(s.delta <= s.rho)
        {
            if(!s.last_improved && s.rescue_counter > 0)
            {
                s.rho = std::max(s.rho * 0.5, s.rho_end);
            }
            s.delta = std::max(s.delta, s.rho);
        }

        // Powell 2009, Section 4: Lagrange-based point replacement.
        // Full SVD evaluation retained for accurate point selection;
        // the main O(m*p^2) saving is in update_model_incremental below.
        Eigen::VectorXd lv_xnew = detail::compute_lagrange_at_point(
            s.Y, s.f_values, s.x_scaled, x_new_scaled);
        int k = detail::select_replacement(
            s.Y, s.f_values, x_new_scaled, f_new, s.x_scaled, lv_xnew);

        // Update interpolation set (scaled coordinates)
        s.Y.col(k) = x_new_scaled;
        s.f_values[k] = f_new;

        // Accept step if strictly better
        bool improved = false;
        if(f_new < old_f)
        {
            s.x_scaled = x_new_scaled;
            s.x = x_new_orig;
            s.objective_value = f_new;
            improved = true;
        }

        // Powell 2009, Section 4: itest model staleness counter.
        if(improved)
            s.itest = 0;
        else
            ++s.itest;

        // Model update: incremental path for rejected steps, full SVD
        // rebuild for accepted steps (re-centers x_base for conditioning).
        // Reference: Powell 2009, Section 4.
        if(improved)
        {
            s.model = detail::build_model(s.Y, s.f_values, s.x_scaled);
        }
        else
        {
            auto lv_result = detail::update_model_incremental(
                s.model, s.f_values, x_new_scaled, k, s.model.x_base);
            if(lv_result.size() == 0)
                s.model = detail::build_model(s.Y, s.f_values, s.model.x_base);
        }

        // Powell 2009, Section 4: when itest >= 3 and NPT > 2N+1, zero H
        if(s.itest >= 3)
        {
            const int n = s.x.size();
            if(s.m > 2 * n + 1)
                s.model.H.setZero();
            s.itest = 0;
        }

        // Powell 2009, Section 6 (ALTMOV concept): geometry improvement.
        // All geometry operations in scaled coordinates.
        if(d_norm < 0.5 * s.rho && s.rho > s.rho_end)
        {
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

            // Geometry-improving point in scaled space
            Eigen::Vector<double, N> dir = (s.Y.col(k_geo) - s.x_scaled).eval();
            double dir_norm = dir.norm();
            if(dir_norm > 1e-15)
            {
                Eigen::Vector<double, N> geo_pt_scaled = detail::project(
                    (s.x_scaled + s.rho * dir / dir_norm).eval(),
                    s.lower_scaled, s.upper_scaled);
                Eigen::Vector<double, N> geo_pt_orig =
                    (geo_pt_scaled.array() * s.scale.array()).matrix();
                double geo_f = s.problem->value(geo_pt_orig);
                s.Y.col(k_geo) = geo_pt_scaled;
                s.f_values[k_geo] = geo_f;

                // Full rebuild for geometry improvement: geometry points can be far
                // from x_base, where LDLT pseudoinverse refresh loses accuracy.
                s.model = detail::build_model(s.Y, s.f_values, s.x_scaled);

                if(geo_f < s.objective_value)
                {
                    s.x_scaled = geo_pt_scaled;
                    s.x = geo_pt_orig;
                    s.objective_value = geo_f;
                    improved = true;
                }
            }
        }

        // Powell 2009, RESCUE concept: full interpolation set rebuild.
        // All in scaled coordinates.
        if(s.delta <= s.rho && !improved)
            ++s.rescue_counter;
        else if(improved)
            s.rescue_counter = 0;

        if(s.rescue_counter >= 3)
        {
            const int n = s.x.size();
            double h = s.rho;

            s.Y.col(0) = s.x_scaled;
            s.f_values[0] = s.problem->value(s.x);

            for(int i = 0; i < n && (1 + i) < s.m; ++i)
            {
                Eigen::Vector<double, N> pt = s.x_scaled;
                pt[i] = std::min(pt[i] + h, s.upper_scaled[i]);
                if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                    pt[i] = std::max(s.x_scaled[i] - h, s.lower_scaled[i]);
                s.Y.col(1 + i) = pt;
                s.f_values[1 + i] = s.problem->value(
                    (pt.array() * s.scale.array()).matrix());
            }

            for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
            {
                Eigen::Vector<double, N> pt = s.x_scaled;
                pt[i] = std::max(pt[i] - h, s.lower_scaled[i]);
                if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                    pt[i] = std::min(s.x_scaled[i] + h, s.upper_scaled[i]);
                s.Y.col(1 + n + i) = pt;
                s.f_values[1 + n + i] = s.problem->value(
                    (pt.array() * s.scale.array()).matrix());
            }

            s.model = detail::build_model(s.Y, s.f_values, s.x_scaled);
            s.rescue_counter = 0;
        }

        ++s.iteration;
        s.last_improved = improved;

        // Derivative-free convergence signalling
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
        s.x_scaled = (s.x.array() / s.scale.array()).matrix();
        s.iteration = 0;
        s.initialized = false;

        double h = s.delta;

        // Rebuild interpolation set in scaled coordinates
        s.Y.col(0) = s.x_scaled;
        s.f_values[0] = s.problem->value(s.x);

        for(int i = 0; i < n && (1 + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x_scaled;
            pt[i] = std::min(pt[i] + h, s.upper_scaled[i]);
            if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                pt[i] = std::max(s.x_scaled[i] - h, s.lower_scaled[i]);
            s.Y.col(1 + i) = pt;
            s.f_values[1 + i] = s.problem->value(
                (pt.array() * s.scale.array()).matrix());
        }

        for(int i = 0; i < n && (1 + n + i) < s.m; ++i)
        {
            Eigen::Vector<double, N> pt = s.x_scaled;
            pt[i] = std::max(pt[i] - h, s.lower_scaled[i]);
            if(std::abs(pt[i] - s.x_scaled[i]) < 1e-15 * h)
                pt[i] = std::min(s.x_scaled[i] + h, s.upper_scaled[i]);
            s.Y.col(1 + n + i) = pt;
            s.f_values[1 + n + i] = s.problem->value(
                (pt.array() * s.scale.array()).matrix());
        }

        // Find best point
        int best = 0;
        for(int i = 1; i < s.m; ++i)
        {
            if(s.f_values[i] < s.f_values[best])
                best = i;
        }
        s.x_scaled = s.Y.col(best);
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.f_values[best];

        s.model = detail::build_model(s.Y, s.f_values, s.x_scaled);
        s.initialized = true;
    }
};

}

#endif
