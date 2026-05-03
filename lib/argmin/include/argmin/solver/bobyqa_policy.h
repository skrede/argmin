#ifndef HPP_GUARD_ARGMIN_SOLVER_BOBYQA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_BOBYQA_POLICY_H

// BOBYQA solver policy for basic_solver.
//
// Implements Powell's Bound Optimization BY Quadratic Approximation.
// A trust-region derivative-free method that maintains a quadratic
// interpolation model Q(x) interpolating f at m points (default m = 2n+1).
// Each step solves the trust-region subproblem min Q(x_k + d) subject to
// ||d|| <= delta and box constraints, then updates the model and radius
// based on the accuracy ratio rho.
//
// Uses Powell's BMAT/ZMAT factored interpolation system for O(m*n) model
// updates instead of O(m*p^2) SVD. ALTMOV geometry improvement is wired
// using exact Lagrange values from BMAT/ZMAT.
//
// Requires: objective<P,S> && bound_constrained<P,S>.
// No gradient is needed -- BOBYQA uses only objective evaluations.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.
//            K&W Section 8.4 (surrogate model framework).

#include "argmin/detail/interpolation_system.h"
#include "argmin/detail/trust_region.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/options/trust_region_options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace argmin
{

// Build explicit N x N Hessian from HQ (packed upper triangle) and PQ
// (implicit second derivative via outer products of interpolation points).
//
// H = HQ_matrix + sum_k pq[k] * xpt[k] * xpt[k]^T
//
// Cost: O(n^2 * m), acceptable for n < 20.
//
// Reference: Powell 2009, Section 2 (equation 2.2).
template <typename Scalar, int N>
Eigen::Matrix<Scalar, N, N> build_explicit_hessian(
    const detail::interpolation_system<Scalar, N>& sys)
{
    const int32_t n = sys.xbase.size();
    const int32_t m = sys.m_points;

    Eigen::Matrix<Scalar, N, N> H;
    if constexpr(N == Eigen::Dynamic)
        H.setZero(n, n);
    else
        H.setZero();

    // Unpack HQ upper triangle into symmetric matrix.
    int32_t ih = 0;
    for(int32_t j = 0; j < n; ++j)
    {
        for(int32_t i = 0; i <= j; ++i)
        {
            H(i, j) = sys.hq[ih];
            H(j, i) = sys.hq[ih];
            ++ih;
        }
    }

    // Add PQ outer products: sum_k pq[k] * xpt[k] * xpt[k]^T.
    for(int32_t k = 0; k < m; ++k)
    {
        if(sys.pq[k] == Scalar(0)) continue;
        auto xk = sys.xpt.col(k).head(n);
        H.template selfadjointView<Eigen::Upper>().rankUpdate(xk, sys.pq[k]);
    }
    // Copy upper to lower.
    H.template triangularView<Eigen::StrictlyLower>() =
        H.template triangularView<Eigen::StrictlyUpper>().transpose();

    return H;
}

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
        std::uint16_t stall_window{200};
        double feasibility_gate{std::numeric_limits<double>::infinity()};
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
        detail::interpolation_system<double, N> sys;  // BMAT/ZMAT factored interpolation system
        double delta{};
        double delta_max{};
        double final_trust_radius{1e-8};
        double rho{};       // Current minimum trust radius (contracts toward rho_end)
        double rho_end{};   // Final rho target (= final_trust_radius)
        std::uint32_t iteration{0};
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
        double h = options.initial_trust_radius.value_or(0.0);
        if(h <= 0.0)
        {
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

        // Bootstrap the BMAT/ZMAT interpolation system.
        // The bootstrap evaluates f at 2n+1 coordinate-perturbation points
        // around x_scaled and initializes BMAT, ZMAT, GOPT, HQ, PQ.
        //
        // Reference: Powell 2009, Section 2.
        //   Adapted from NLopt prelim_() lines 1710-1950.
        //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1710
        s.sys = detail::bootstrap_interpolation_system<double, N>(
            s.x_scaled, h, s.lower_scaled, s.upper_scaled,
            [&](const Eigen::Vector<double, N>& x_sc) {
                return s.problem->value(
                    (x_sc.array() * s.scale.array()).matrix());
            });

        // Update x to the best point found during bootstrap.
        s.x_scaled = s.sys.xbase + s.sys.xopt;
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.sys.fval[s.sys.kopt];

        s.initialized = true;
        s.iteration = 0;

        // Powell 2009, Section 5: two-radius rho contraction scheme.
        s.rho = s.delta;
        s.rho_end = s.final_trust_radius;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        double old_f = s.objective_value;
        const int n = s.x.size();

        // Model gradient at xopt (d=0 gives gopt directly).
        Eigen::Vector<double, N> zero_d;
        if constexpr(N == Eigen::Dynamic)
            zero_d.setZero(n);
        else
            zero_d.setZero();

        Eigen::Vector<double, N> mg = detail::model_gradient_at(s.sys, zero_d);

        // Build explicit Hessian for the trust-region subproblem.
        // Cost: O(n^2 * m), acceptable for n < 20.
        Eigen::Matrix<double, N, N> H = build_explicit_hessian<double, N>(s.sys);

        // x_opt in absolute scaled coordinates for the trust-region solver.
        Eigen::Vector<double, N> x_opt_abs = s.sys.xbase + s.sys.xopt;

        // Solve trust-region subproblem in scaled coordinates.
        Eigen::Vector<double, N> d = detail::solve_trust_region_box(
            mg, H, x_opt_abs, s.delta, s.lower_scaled, s.upper_scaled);

        double d_norm = d.norm();

        // Powell 2009, Section 5: convergence via rho contraction.
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

        // Trial point: xopt + d in shifted coordinates (relative to xbase).
        Eigen::Vector<double, N> x_new_shifted = s.sys.xopt + d;
        Eigen::Vector<double, N> x_new_abs = detail::project(
            (s.sys.xbase + x_new_shifted).eval(), s.lower_scaled, s.upper_scaled);
        // Recompute the shifted coordinate after projection.
        x_new_shifted = x_new_abs - s.sys.xbase;
        // Recompute d after projection for accurate model prediction.
        d = x_new_shifted - s.sys.xopt;

        Eigen::Vector<double, N> x_new_orig =
            (x_new_abs.array() * s.scale.array()).matrix();
        double f_new = s.problem->value(x_new_orig);

        // Model predictions: Q(xopt + d) - Q(xopt) = Q(d) since gopt is at xopt.
        double q_predicted = detail::evaluate_interpolation_model(s.sys, d);

        // Accuracy ratio.
        //
        // Reference: Powell 2009, eq. (3.1).
        double accuracy_ratio = 0.0;
        if(std::abs(q_predicted) > std::numeric_limits<double>::epsilon() * 100.0)
            accuracy_ratio = (old_f - f_new) / (-q_predicted);

        // (model predictions used for accuracy ratio above)

        // Update trust-region radius (delta).
        s.delta = detail::update_radius(s.delta, accuracy_ratio, d_norm, s.delta_max,
                                        options.trust);

        // Powell 2009, Section 5: rho contraction using three-regime schedule.
        // Adapted from NLopt bobyqa.c lines 2993-3017.
        // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2993
        if(s.delta <= s.rho)
        {
            if(accuracy_ratio <= 0.0 && std::max(s.delta, d_norm) <= s.rho && s.rho > s.rho_end)
                std::tie(s.rho, s.delta) = detail::contract_rho(s.rho, s.rho_end);
            s.delta = std::max(s.delta, s.rho);
        }

        // Powell 2009, Section 4: BMAT/ZMAT-based point replacement.
        // Compute VLAG and BETA for the trial step.
        auto [vlag, beta] = detail::compute_vlag_beta(s.sys, d);

        // Lagrange values at the trial point (first m entries of vlag).
        Eigen::VectorXd lv_xnew(s.m);
        for(int k = 0; k < s.m; ++k)
            lv_xnew[k] = vlag[k];

        // Select replacement point using denominator*distance^4 weighting.
        //
        // Reference: Powell 2009, Section 4.
        //   Adapted from NLopt bobyqa.c lines 2493-2549.
        //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2493
        //
        // Use xpt columns (shifted coords) and x_opt_abs for distances
        // in the select_replacement interface that expects absolute coords.
        Eigen::Matrix<double, N, Eigen::Dynamic> Y_abs(n, s.m);
        Eigen::VectorXd fval_vec(s.m);
        for(int k = 0; k < s.m; ++k)
        {
            Y_abs.col(k) = s.sys.xbase + s.sys.xpt.col(k).head(n);
            fval_vec[k] = s.sys.fval[k];
        }

        int knew = detail::select_replacement(
            Y_abs, fval_vec, x_new_abs, f_new, x_opt_abs, lv_xnew, s.delta);

        // Compute denominator for BMAT/ZMAT update.
        // alpha = sum_j zmat[knew, j]^2 (diagonal element of Z*Z^T).
        double alpha = 0.0;
        {
            int32_t nptm = s.m - n - 1;
            for(int32_t jj = 0; jj < nptm; ++jj)
                alpha += s.sys.zmat(knew, jj) * s.sys.zmat(knew, jj);
        }
        double denom = detail::compute_denom(vlag[knew], alpha, beta);

        // Update BMAT/ZMAT if denominator is healthy.
        // If denominator collapses, skip the factored update and trigger rescue.
        bool denom_ok = (denom > 1e-20);
        if(denom_ok)
        {
            detail::update_bmat_zmat(s.sys, vlag, beta, denom, knew);
            detail::update_model_on_replacement(s.sys, x_new_shifted, f_new, knew, d);
        }
        else
        {
            // Denominator collapse: just update point data for consistency.
            s.sys.fval[knew] = f_new;
            for(int i = 0; i < n; ++i)
                s.sys.xpt(i, knew) = x_new_shifted[i];
            ++s.rescue_counter;
        }

        // Accept step if strictly better.
        bool improved = false;
        if(f_new < old_f)
        {
            s.x_scaled = x_new_abs;
            s.x = x_new_orig;
            s.objective_value = f_new;
            improved = true;

            // If update_model_on_replacement didn't already update kopt
            // (it does when f_new < fval[kopt]), ensure consistency.
            if(denom_ok && s.sys.kopt != knew && f_new < s.sys.fval[s.sys.kopt])
            {
                s.sys.kopt = knew;
                s.sys.xopt = s.sys.xpt.col(knew).head(n);
            }
        }

        // Powell 2009, Section 6 (ALTMOV): geometry improvement.
        // With BMAT/ZMAT, the exact Lagrange values enable effective ALTMOV
        // placement, eliminating the numerical drift that made it unusable
        // under the SVD model representation.
        //
        // Reference: Powell 2009, Section 6.
        //   Adapted from NLopt bobyqa.c lines 743-1159.
        //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L743
        if(d_norm < 0.5 * s.rho && s.rho > s.rho_end)
        {
            // Select knew_geo as the point with largest |L_k(xopt)|.
            auto lagrange_at_xopt = detail::compute_lagrange_at(s.sys, s.sys.xopt);

            int knew_geo = (s.sys.kopt == 0) ? 1 : 0;
            double max_abs_lk = std::abs(lagrange_at_xopt[knew_geo]);
            for(int i = 0; i < s.m; ++i)
            {
                if(i == s.sys.kopt) continue;
                double abs_lk = std::abs(lagrange_at_xopt[i]);
                if(abs_lk > max_abs_lk)
                {
                    max_abs_lk = abs_lk;
                    knew_geo = i;
                }
            }

            // ALTMOV geometry step: find the point maximizing |L_knew_geo|.
            Eigen::Vector<double, N> geo_pt_shifted = detail::altmov_geometry_step<double, N>(
                s.sys, knew_geo, s.rho, s.lower_scaled - s.sys.xbase,
                s.upper_scaled - s.sys.xbase);

            Eigen::Vector<double, N> geo_pt_abs = detail::project(
                (s.sys.xbase + geo_pt_shifted).eval(), s.lower_scaled, s.upper_scaled);
            geo_pt_shifted = geo_pt_abs - s.sys.xbase;

            Eigen::Vector<double, N> geo_pt_orig =
                (geo_pt_abs.array() * s.scale.array()).matrix();
            double geo_f = s.problem->value(geo_pt_orig);

            // Compute VLAG/BETA for the geometry step.
            Eigen::Vector<double, N> d_geo = geo_pt_shifted - s.sys.xopt;
            auto [vlag_geo, beta_geo] = detail::compute_vlag_beta(s.sys, d_geo);

            double alpha_geo = 0.0;
            {
                int32_t nptm = s.m - n - 1;
                for(int32_t jj = 0; jj < nptm; ++jj)
                    alpha_geo += s.sys.zmat(knew_geo, jj) * s.sys.zmat(knew_geo, jj);
            }
            double denom_geo = detail::compute_denom(vlag_geo[knew_geo], alpha_geo, beta_geo);

            if(denom_geo > 1e-20)
            {
                detail::update_bmat_zmat(s.sys, vlag_geo, beta_geo, denom_geo, knew_geo);
                detail::update_model_on_replacement(s.sys, geo_pt_shifted, geo_f, knew_geo, d_geo);

                if(geo_f < s.objective_value)
                {
                    s.x_scaled = geo_pt_abs;
                    s.x = geo_pt_orig;
                    s.objective_value = geo_f;
                    improved = true;
                }
            }
        }

        // Powell 2009, RESCUE concept: full interpolation system rebuild.
        if(s.delta <= s.rho && !improved)
            ++s.rescue_counter;
        else if(improved)
            s.rescue_counter = 0;

        if(s.rescue_counter >= 3)
        {
            double h = s.rho;
            Eigen::Vector<double, N> x_base_new = s.sys.xbase + s.sys.xopt;

            s.sys = detail::bootstrap_interpolation_system<double, N>(
                x_base_new, h, s.lower_scaled, s.upper_scaled,
                [&](const Eigen::Vector<double, N>& x_sc) {
                    return s.problem->value(
                        (x_sc.array() * s.scale.array()).matrix());
                });

            s.x_scaled = s.sys.xbase + s.sys.xopt;
            s.x = (s.x_scaled.array() * s.scale.array()).matrix();
            s.objective_value = s.sys.fval[s.sys.kopt];
            s.rescue_counter = 0;
        }

        ++s.iteration;
        s.last_improved = improved;

        // Derivative-free convergence signalling.
        double obj_change = s.objective_value - old_f;
        double effective_step = improved ? d_norm : s.delta;
        double effective_change = improved ? obj_change : s.delta;

        double grad_proxy = mg.norm();
        if(s.rho > s.rho_end)
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
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        s.x_scaled = (s.x.array() / s.scale.array()).matrix();
        s.iteration = 0;
        s.initialized = false;

        double h = s.delta;

        s.sys = detail::bootstrap_interpolation_system<double, N>(
            s.x_scaled, h, s.lower_scaled, s.upper_scaled,
            [&](const Eigen::Vector<double, N>& x_sc) {
                return s.problem->value(
                    (x_sc.array() * s.scale.array()).matrix());
            });

        s.x_scaled = s.sys.xbase + s.sys.xopt;
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.sys.fval[s.sys.kopt];
        s.initialized = true;
    }
};

}

#endif
