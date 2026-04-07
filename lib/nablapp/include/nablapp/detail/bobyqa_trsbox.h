#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_TRSBOX_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_TRSBOX_H

// Powell's TRSBOX: bounded trust-region subproblem solver for BOBYQA.
//
// Solves min_d Q(x_opt + d) subject to ||d|| <= delta and
// lower <= x_base + x_opt + d <= upper. Uses truncated conjugate
// gradient with XBDI active-set tracking and a 2D alternative
// iteration at the trust-region boundary for non-convex directions.
//
// This replaces the simplified solve_trust_region_box from trust_region.h
// with the full Powell TRSBOX including CG restarts on bound hits and
// the angular search at the boundary.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Section 3 (TRSBOX subroutine).

#include "nablapp/detail/bobyqa_model.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp::detail
{

// Solve the bounded trust-region subproblem per Powell 2009, Sec. 3.
//
// Returns the step d from x_opt. The new point is x_base + x_opt + d,
// projected to bounds. The step satisfies ||d|| <= delta and
// lower <= x_base + x_opt + d <= upper.
//
// The algorithm:
//   1. Initialize XBDI bound indicators
//   2. CG iteration over the free subspace with restarts on bound hits
//   3. 2D alternative iteration at the trust-region boundary
//
// Reference: Powell 2009, Sec. 3.
template <typename Scalar = double,
          int N = nablapp::dynamic_dimension,
          int NPT = (N == nablapp::dynamic_dimension
                         ? nablapp::dynamic_dimension
                         : 2 * N + 1)>
Eigen::Vector<Scalar, N> trsbox(
    const bobyqa_model<Scalar, N, NPT>& model,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Scalar delta)
{
    const int n = model.gopt.size();
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    constexpr int max_cg_iters = 200;

    // Effective bounds on d: (lower - x_base - x_opt) <= d <= (upper - x_base - x_opt).
    Eigen::Vector<Scalar, N> d_lo = lower - model.x_base - model.x_opt;
    Eigen::Vector<Scalar, N> d_hi = upper - model.x_base - model.x_opt;

    // Step 1: Initialize XBDI bound indicators (Powell 2009, Sec. 3).
    //
    // xbdi[i] = -1 if at lower bound and gradient pushes toward it
    // xbdi[i] = +1 if at upper bound and gradient pushes toward it
    // xbdi[i] =  0 if free
    Eigen::Vector<std::int8_t, N> xbdi(n);
    Eigen::Vector<Scalar, N> d = Eigen::Vector<Scalar, N>::Zero(n);
    Eigen::Vector<Scalar, N> g = model.gopt;  // Gradient at x_opt (d=0)

    auto initialize_xbdi = [&]()
    {
        for(int i = 0; i < n; ++i)
        {
            xbdi[i] = 0;
            if(d[i] <= d_lo[i] + eps * std::abs(d_lo[i]) && g[i] > Scalar(0))
                xbdi[i] = -1;
            else if(d[i] >= d_hi[i] - eps * std::abs(d_hi[i]) && g[i] < Scalar(0))
                xbdi[i] = 1;
        }
    };

    initialize_xbdi();

    // Clamp d to box bounds.
    auto clamp_to_box = [&](Eigen::Vector<Scalar, N>& v)
    {
        for(int i = 0; i < n; ++i)
            v[i] = std::clamp(v[i], d_lo[i], d_hi[i]);
    };

    // Count free variables.
    auto count_free = [&]() -> int
    {
        int count = 0;
        for(int i = 0; i < n; ++i)
            if(xbdi[i] == 0) ++count;
        return count;
    };

    // Zero out components at active bounds.
    auto zero_active = [&](Eigen::Vector<Scalar, N>& v)
    {
        for(int i = 0; i < n; ++i)
            if(xbdi[i] != 0) v[i] = Scalar(0);
    };

    // Step 2: CG iteration over the free subspace (Powell 2009, Sec. 3).
    //
    // CG is restarted when a new bound becomes active. The search
    // direction is projected onto the free subspace at each step.
    bool at_trust_boundary = false;
    Eigen::Vector<Scalar, N> s(n);
    Scalar rr_old = Scalar(0);

    for(int cg_restart = 0; cg_restart < n; ++cg_restart)
    {
        if(count_free() == 0) break;

        // Compute gradient at current d.
        g = model.gradient(d);
        zero_active(g);

        // Initial search direction.
        s = -g;
        rr_old = g.squaredNorm();

        if(rr_old < eps * eps) break;

        for(int iter = 0; iter < max_cg_iters; ++iter)
        {
            // Hessian-vector product H*s.
            Eigen::Vector<Scalar, N> hs = model.hessian_vector_product(s);

            Scalar shs = s.dot(hs);
            Scalar ss = s.squaredNorm();

            if(ss < eps * eps) break;

            // Max step to trust-region boundary: ||d + alpha*s||^2 = delta^2.
            Scalar dd = d.squaredNorm();
            Scalar ds = d.dot(s);
            Scalar disc = ds * ds - ss * (dd - delta * delta);
            Scalar alpha_tr = std::numeric_limits<Scalar>::infinity();
            if(disc > Scalar(0))
                alpha_tr = (-ds + std::sqrt(disc)) / ss;
            if(alpha_tr < Scalar(0))
                alpha_tr = std::numeric_limits<Scalar>::infinity();

            // Max step to box boundaries.
            Scalar alpha_box = std::numeric_limits<Scalar>::infinity();
            int bound_var = -1;
            for(int i = 0; i < n; ++i)
            {
                if(xbdi[i] != 0 || std::abs(s[i]) < eps) continue;

                Scalar ab;
                if(s[i] > Scalar(0))
                    ab = (d_hi[i] - d[i]) / s[i];
                else
                    ab = (d_lo[i] - d[i]) / s[i];

                if(ab < alpha_box)
                {
                    alpha_box = ab;
                    bound_var = i;
                }
            }

            // CG step.
            Scalar alpha_cg = std::numeric_limits<Scalar>::infinity();
            if(shs > eps * ss)
                alpha_cg = rr_old / shs;

            if(shs <= eps * ss)
            {
                // Non-positive curvature: move to boundary.
                Scalar alpha = std::min(alpha_tr, alpha_box);
                if(!std::isfinite(alpha)) alpha = Scalar(1);
                d += alpha * s;
                clamp_to_box(d);
                if(alpha >= alpha_tr - eps)
                    at_trust_boundary = true;
                if(alpha >= alpha_box - eps && bound_var >= 0)
                {
                    xbdi[bound_var] = (s[bound_var] > Scalar(0))
                                          ? static_cast<std::int8_t>(1)
                                          : static_cast<std::int8_t>(-1);
                }
                break;
            }

            if(alpha_cg >= alpha_tr)
            {
                // CG step hits trust-region boundary.
                d += alpha_tr * s;
                clamp_to_box(d);
                at_trust_boundary = true;
                break;
            }

            if(alpha_cg >= alpha_box)
            {
                // CG step hits box boundary; activate variable and restart CG.
                d += alpha_box * s;
                clamp_to_box(d);
                if(bound_var >= 0)
                {
                    xbdi[bound_var] = (s[bound_var] > Scalar(0))
                                          ? static_cast<std::int8_t>(1)
                                          : static_cast<std::int8_t>(-1);
                }
                break;  // Will restart CG in outer loop.
            }

            // Standard CG update.
            d += alpha_cg * s;
            clamp_to_box(d);

            // Update residual.
            g += alpha_cg * hs;
            zero_active(g);

            Scalar rr_new = g.squaredNorm();
            if(rr_new < eps * eps) { at_trust_boundary = false; break; }

            Scalar beta_cg = rr_new / rr_old;
            rr_old = rr_new;
            s = -g + beta_cg * s;
            zero_active(s);
        }

        if(at_trust_boundary || count_free() == 0) break;
    }

    // Step 3: 2D alternative iteration at the trust-region boundary
    // (Powell 2009, Sec. 3).
    //
    // When CG terminates at the trust-region boundary, search along
    // the arc defined by the current direction and an alternative
    // direction to find a better point on the boundary. This handles
    // non-convex directions where CG terminates early.
    if(at_trust_boundary)
    {
        alternative_iteration_2d(model, d, d_lo, d_hi, xbdi, delta, n);
    }

    // Final safety projection to bounds.
    clamp_to_box(d);

    // Ensure trust-region constraint.
    Scalar d_norm = d.norm();
    if(d_norm > delta)
        d *= delta / d_norm;

    return d;
}

// 2D alternative iteration at the trust-region boundary.
//
// Searches along arcs on the trust-region boundary sphere to find
// improvements in the model value. Uses the gradient projection and
// an orthogonal alternative direction.
//
// Reference: Powell 2009, Sec. 3 (alternative iteration).
template <typename Scalar, int N, int NPT>
void alternative_iteration_2d(
    const bobyqa_model<Scalar, N, NPT>& model,
    Eigen::Vector<Scalar, N>& d,
    const Eigen::Vector<Scalar, N>& d_lo,
    const Eigen::Vector<Scalar, N>& d_hi,
    const Eigen::Vector<std::int8_t, N>& xbdi,
    Scalar delta,
    int n)
{
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    constexpr int max_alt_iters = 50;

    // Compute gradient at current d.
    Eigen::Vector<Scalar, N> g = model.gradient(d);

    // Zero out active components.
    for(int i = 0; i < n; ++i)
        if(xbdi[i] != 0) g[i] = Scalar(0);

    Scalar g_norm = g.norm();
    if(g_norm < eps) return;

    // Alternative direction: the gradient projection orthogonal to d.
    Eigen::Vector<Scalar, N> alt = g;
    Scalar dg = d.dot(g);
    Scalar dd = d.squaredNorm();
    if(dd > eps)
        alt -= (dg / dd) * d;

    Scalar alt_norm = alt.norm();
    if(alt_norm < eps * g_norm) return;
    alt /= alt_norm;
    alt *= delta;

    // Search along the arc d(theta) = d*cos(theta) + alt*sin(theta).
    // We parametrize theta in [0, pi] and find the minimum of Q.
    //
    // Reference: Powell 2009, Sec. 3.
    Scalar best_val = model.evaluate(d);
    Eigen::Vector<Scalar, N> best_d = d;

    for(int iter = 1; iter <= max_alt_iters; ++iter)
    {
        Scalar theta = static_cast<Scalar>(iter) / static_cast<Scalar>(max_alt_iters + 1)
                       * Scalar(3.14159265358979323846);
        Scalar ct = std::cos(theta);
        Scalar st = std::sin(theta);

        Eigen::Vector<Scalar, N> d_trial = ct * d + st * alt;

        // Project to box bounds.
        for(int i = 0; i < n; ++i)
            d_trial[i] = std::clamp(d_trial[i], d_lo[i], d_hi[i]);

        // Scale back to trust region if needed.
        Scalar trial_norm = d_trial.norm();
        if(trial_norm > delta)
            d_trial *= delta / trial_norm;

        Scalar val = model.evaluate(d_trial);
        if(val < best_val)
        {
            best_val = val;
            best_d = d_trial;
        }
    }

    d = best_d;
}

}

#endif
