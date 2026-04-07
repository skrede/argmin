#ifndef HPP_GUARD_NABLAPP_DETAIL_TRUST_REGION_H
#define HPP_GUARD_NABLAPP_DETAIL_TRUST_REGION_H

// Trust-region subproblem solver with box constraints.
//
// Provides the trust-region step computation, accuracy ratio, radius
// update, geometry checking, and point replacement selection used by
// BOBYQA. The trust-region subproblem uses truncated projected conjugate
// gradient (a simplified version of Powell's TRSBOX subroutine).
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Sections 3 (TRSBOX), 4 (point replacement), 5 (radius update),
//            6 (geometry check).

#include "nablapp/types.h"
#include "nablapp/options/trust_region_options.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp::detail
{

// Solve trust-region subproblem with box constraints using XBDI active-set CG.
//
// min_d  g^T d + 0.5 d^T H d
// s.t.   ||d|| <= delta
//        lower <= x_k + d <= upper
//
// Uses Powell's TRSBOX approach with XBDI bound indicators:
// - xbdi[i] = -1 if variable i is active at lower bound
// - xbdi[i] = +1 if variable i is active at upper bound
// - xbdi[i] = 0 if variable i is free
//
// CG runs over the free subspace. When a variable hits a bound during CG,
// it is fixed (xbdi updated) and CG restarts over the reduced free set.
// This preserves conjugacy within each CG run, unlike the simpler approach
// of zeroing search direction components which loses conjugacy.
//
// Returns step d (not new point). x_new = x_k + d.
//
// Reference: Powell 2009, Section 3 (TRSBOX subroutine).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> solve_trust_region_box(
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Matrix<Scalar, N, N>& H,
    const Eigen::Vector<Scalar, N>& x_k,
    Scalar delta,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    const int n = g.size();
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();

    Eigen::Vector<Scalar, N> d = Eigen::Vector<Scalar, N>::Zero(n);

    // Effective box bounds on d: lower - x_k <= d <= upper - x_k
    Eigen::Vector<Scalar, N> d_lo = (lower - x_k).eval();
    Eigen::Vector<Scalar, N> d_hi = (upper - x_k).eval();

    // Clamp delta-ball to box bounds
    for(int i = 0; i < n; ++i)
    {
        d_lo[i] = std::max(d_lo[i], -delta);
        d_hi[i] = std::min(d_hi[i], delta);
    }

    // Initialize XBDI bound indicators.
    // A variable is initially bound-active if it is at a box bound AND the
    // gradient pushes it further into the bound (no benefit from freeing it).
    Eigen::Vector<int8_t, N> xbdi = Eigen::Vector<int8_t, N>::Zero(n);
    for(int i = 0; i < n; ++i)
    {
        if(d_lo[i] >= -eps && g[i] >= Scalar(0))
            xbdi[i] = -1;
        else if(d_hi[i] <= eps && g[i] <= Scalar(0))
            xbdi[i] = 1;
    }

    // Outer loop: restart CG when a new variable hits a bound.
    // At most n restarts (each fixes one variable).
    for(int outer = 0; outer < n; ++outer)
    {
        // Count free variables
        int n_free = 0;
        for(int i = 0; i < n; ++i)
        {
            if(xbdi[i] == 0)
                ++n_free;
        }
        if(n_free == 0)
            break;

        // CG residual: projected gradient (zero bound-active components)
        Eigen::Vector<Scalar, N> grad_d = (g + H * d).eval();
        Eigen::Vector<Scalar, N> r(n);
        for(int i = 0; i < n; ++i)
            r[i] = (xbdi[i] == 0) ? grad_d[i] : Scalar(0);

        Eigen::Vector<Scalar, N> s = -r;

        bool bound_hit = false;

        // Inner CG loop over the free subspace
        for(int cg_iter = 0; cg_iter < n_free; ++cg_iter)
        {
            Scalar rr = r.dot(r);
            if(rr < eps * eps * Scalar(100) * std::max(Scalar(1), g.squaredNorm()))
                break;

            Scalar ss = s.squaredNorm();
            if(ss < eps * eps)
                break;

            Scalar sHs = s.dot(H * s);

            // Trust-region boundary step
            Scalar dd = d.squaredNorm();
            Scalar ds = d.dot(s);
            Scalar alpha_tr = std::numeric_limits<Scalar>::infinity();
            Scalar disc = ds * ds - ss * (dd - delta * delta);
            if(disc > Scalar(0))
                alpha_tr = (-ds + std::sqrt(disc)) / ss;

            // Box boundary step (free variables only)
            Scalar alpha_box = std::numeric_limits<Scalar>::infinity();
            int bound_var = -1;
            int8_t bound_dir = 0;
            for(int i = 0; i < n; ++i)
            {
                if(xbdi[i] != 0) continue;
                Scalar ab = std::numeric_limits<Scalar>::infinity();
                int8_t dir = 0;
                if(s[i] > eps)
                {
                    ab = (d_hi[i] - d[i]) / s[i];
                    dir = 1;
                }
                else if(s[i] < -eps)
                {
                    ab = (d_lo[i] - d[i]) / s[i];
                    dir = -1;
                }
                if(ab < alpha_box)
                {
                    alpha_box = ab;
                    bound_var = i;
                    bound_dir = dir;
                }
            }

            if(sHs <= eps * ss)
            {
                // Non-positive curvature: move to nearest boundary
                Scalar alpha = std::min(alpha_tr, alpha_box);
                if(!std::isfinite(alpha))
                    alpha = Scalar(1);
                d += alpha * s;
                if(alpha_box <= alpha_tr && bound_var >= 0)
                {
                    d[bound_var] = (bound_dir > 0) ? d_hi[bound_var] : d_lo[bound_var];
                    xbdi[bound_var] = bound_dir;
                    bound_hit = true;
                }
                break;
            }

            Scalar alpha_cg = rr / sHs;

            if(alpha_cg >= alpha_tr)
            {
                // CG hits trust-region boundary
                d += alpha_tr * s;
                break;
            }

            if(alpha_cg >= alpha_box)
            {
                // CG hits box bound — fix variable and restart
                d += alpha_box * s;
                if(bound_var >= 0)
                {
                    d[bound_var] = (bound_dir > 0) ? d_hi[bound_var] : d_lo[bound_var];
                    xbdi[bound_var] = bound_dir;
                }
                bound_hit = true;
                break;
            }

            // Standard CG update
            d += alpha_cg * s;

            // Project to box (numerical safety)
            for(int i = 0; i < n; ++i)
                d[i] = std::clamp(d[i], d_lo[i], d_hi[i]);

            Eigen::Vector<Scalar, N> r_new = (r + alpha_cg * (H * s)).eval();
            // Zero bound-active components
            for(int i = 0; i < n; ++i)
            {
                if(xbdi[i] != 0)
                    r_new[i] = Scalar(0);
            }

            Scalar rr_new = r_new.dot(r_new);
            if(rr < eps * eps)
                break;

            Scalar beta = rr_new / rr;
            s = (-r_new + beta * s).eval();
            // Zero bound-active components of search direction
            for(int i = 0; i < n; ++i)
            {
                if(xbdi[i] != 0)
                    s[i] = Scalar(0);
            }

            r = r_new;
        }

        if(!bound_hit)
            break;
    }

    // Post-CG check: try freeing recently-bound variables that would
    // improve the model if released. This captures the key benefit of
    // Powell's 2D alternative step (angbd) without full 2D geometry.
    {
        Eigen::Vector<Scalar, N> grad_d = (g + H * d).eval();
        for(int i = 0; i < n; ++i)
        {
            if(xbdi[i] == 0) continue;
            // Check if gradient at current d wants to move variable away from bound
            if(xbdi[i] == -1 && grad_d[i] < Scalar(0))
            {
                xbdi[i] = 0;
                // Allow one more CG pass (handled by outer loop continuing)
            }
            else if(xbdi[i] == 1 && grad_d[i] > Scalar(0))
            {
                xbdi[i] = 0;
            }
        }
    }

    // Final projection to box
    for(int i = 0; i < n; ++i)
        d[i] = std::clamp(d[i], d_lo[i], d_hi[i]);

    // Ensure trust-region constraint
    if(d.norm() > delta)
        d *= delta / d.norm();

    return d;
}

// Compute model accuracy ratio rho.
//
// rho = (f(x_k) - f(x_new)) / (Q(x_k) - Q(x_new))
//
// Returns 0 if predicted reduction is near zero (avoids division by zero).
//
// Reference: Powell 2009, eq. (3.1).
template <typename Scalar = double>
Scalar compute_rho(Scalar f_old, Scalar f_new, Scalar q_old, Scalar q_new)
{
    Scalar predicted = q_old - q_new;
    if(std::abs(predicted) < std::numeric_limits<Scalar>::epsilon() * Scalar(100))
        return Scalar(0);
    return (f_old - f_new) / predicted;
}

// Update trust-region radius based on rho.
//
// If rho > 0.7 and ||d|| close to delta: increase delta (up to delta_max).
// If rho < 0.1: decrease delta (multiply by 0.5).
// Else: keep delta unchanged.
//
// Reference: Powell 2009, Section 5.
template <typename Scalar = double>
Scalar update_radius(Scalar delta, Scalar rho, Scalar step_norm, Scalar delta_max,
                     const trust_region_options& opts = {})
{
    const auto eta_good = static_cast<Scalar>(opts.eta_good.value_or(0.7));
    const auto eta_poor = static_cast<Scalar>(opts.eta_poor.value_or(0.1));
    const auto step_thr = static_cast<Scalar>(opts.step_threshold.value_or(0.5));
    const auto expand = static_cast<Scalar>(opts.expand_factor.value_or(2.0));
    const auto shrink = static_cast<Scalar>(opts.shrink_factor.value_or(0.5));

    if(rho > eta_good && step_norm > step_thr * delta)
        return std::min(expand * delta, delta_max);
    if(rho < eta_poor)
        return shrink * delta;
    return delta;
}

// Check if interpolation geometry is adequate.
//
// Computes the distance of each interpolation point from x_k. Returns
// the index of the point farthest beyond 2*delta (worst geometry), or
// -1 if all points are within acceptable distance.
//
// Reference: Powell 2009, Section 6.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
int check_geometry(const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& Y,
                   const Eigen::Vector<Scalar, N>& x_k,
                   Scalar delta,
                   const trust_region_options& opts = {})
{
    const int m = Y.cols();
    const auto geo_factor = static_cast<Scalar>(opts.geometry_factor.value_or(2.0));
    Scalar worst_dist = geo_factor * delta;
    int worst_idx = -1;

    for(int i = 0; i < m; ++i)
    {
        Scalar dist = (Y.col(i) - x_k).norm();
        if(dist > worst_dist)
        {
            worst_dist = dist;
            worst_idx = i;
        }
    }

    return worst_idx;
}

// Select which interpolation point to replace using Lagrange values.
//
// Hybrid criterion inspired by Powell 2009 Section 4:
// - When f_new improves the best: replace worst-f non-best point (preserve
//   geometry, remove the least useful objective sample).
// - When f_new does NOT improve: choose the point k that maximizes
//   |L_k(x_new)| among non-best points. Replacing the point with largest
//   Lagrange value produces the best-conditioned interpolation update.
//
// Reference: Powell 2009, Section 4.
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
int select_replacement(const Eigen::Matrix<Scalar, N, P>& Y,
                       const Eigen::Vector<Scalar, P>& f_values,
                       const Eigen::Vector<Scalar, N>& x_new,
                       Scalar f_new,
                       const Eigen::Vector<Scalar, N>& x_k,
                       const Eigen::VectorXd& lagrange_values_at_xnew)
{
    const int m = Y.cols();

    // Find the best point (lowest f) -- never replace it
    int best_idx = 0;
    for(int i = 1; i < m; ++i)
    {
        if(f_values[i] < f_values[best_idx])
            best_idx = i;
    }

    // Improvement step: replace worst-f non-best point
    if(f_new < f_values[best_idx])
    {
        int worst_idx = (best_idx == 0) ? 1 : 0;
        for(int i = 0; i < m; ++i)
        {
            if(i == best_idx) continue;
            if(f_values[i] > f_values[worst_idx])
                worst_idx = i;
        }
        return worst_idx;
    }

    // Non-improvement step: Lagrange criterion, max |L_k(x_new)|
    int chosen = (best_idx == 0) ? 1 : 0;
    Scalar max_abs_lk = std::abs(lagrange_values_at_xnew[chosen]);

    for(int i = 0; i < m; ++i)
    {
        if(i == best_idx) continue;
        Scalar abs_lk = std::abs(lagrange_values_at_xnew[i]);
        if(abs_lk > max_abs_lk)
        {
            max_abs_lk = abs_lk;
            chosen = i;
        }
    }

    return chosen;
}

// Backward-compatible overload: farthest-point heuristic (no Lagrange values).
//
// Reference: Powell 2009, Section 4.
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
int select_replacement(const Eigen::Matrix<Scalar, N, P>& Y,
                       const Eigen::Vector<Scalar, P>& f_values,
                       const Eigen::Vector<Scalar, N>& x_new,
                       Scalar f_new,
                       const Eigen::Vector<Scalar, N>& x_k)
{
    const int m = Y.cols();

    // Find the best point (lowest f) -- never replace it
    int best_idx = 0;
    for(int i = 1; i < m; ++i)
    {
        if(f_values[i] < f_values[best_idx])
            best_idx = i;
    }

    // If the new point improves the best, replace the worst-f non-best point
    if(f_new < f_values[best_idx])
    {
        int worst_idx = (best_idx == 0) ? 1 : 0;
        for(int i = 0; i < m; ++i)
        {
            if(i == best_idx) continue;
            if(f_values[i] > f_values[worst_idx])
                worst_idx = i;
        }
        return worst_idx;
    }

    // Otherwise, replace the farthest point from x_k (excluding best)
    int farthest_idx = (best_idx == 0) ? 1 : 0;
    Scalar max_dist = (Y.col(farthest_idx) - x_k).squaredNorm();

    for(int i = 0; i < m; ++i)
    {
        if(i == best_idx) continue;
        Scalar dist = (Y.col(i) - x_k).squaredNorm();
        if(dist > max_dist)
        {
            max_dist = dist;
            farthest_idx = i;
        }
    }

    return farthest_idx;
}

}

#endif
