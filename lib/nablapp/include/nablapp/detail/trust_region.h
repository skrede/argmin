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
#include <limits>

namespace nablapp::detail
{

// Solve trust-region subproblem with box constraints.
//
// min_d  g^T d + 0.5 d^T H d
// s.t.   ||d|| <= delta
//        lower <= x_k + d <= upper
//
// Uses truncated projected conjugate gradient. For the n=6-7 dimensions
// typical in liepp, this converges in at most n CG steps.
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

    // Truncated projected CG
    Eigen::Vector<Scalar, N> r = g;
    Eigen::Vector<Scalar, N> s = -r;

    for(int iter = 0; iter < n; ++iter)
    {
        // Check if residual is small enough
        if(r.norm() < eps * Scalar(100) * std::max(Scalar(1), g.norm()))
            break;

        // Curvature along s
        Scalar sHs = s.dot(H * s);

        // Maximum step along s before hitting trust-region boundary
        // ||d + alpha * s||^2 <= delta^2
        Scalar dd = d.squaredNorm();
        Scalar ds = d.dot(s);
        Scalar ss = s.squaredNorm();

        if(ss < eps * eps)
            break;

        Scalar alpha_tr = std::numeric_limits<Scalar>::infinity();
        Scalar disc = ds * ds - ss * (dd - delta * delta);
        if(disc > Scalar(0))
            alpha_tr = (-ds + std::sqrt(disc)) / ss;

        // Maximum step before hitting box bounds
        Scalar alpha_box = std::numeric_limits<Scalar>::infinity();
        for(int i = 0; i < n; ++i)
        {
            if(s[i] > eps)
                alpha_box = std::min(alpha_box, (d_hi[i] - d[i]) / s[i]);
            else if(s[i] < -eps)
                alpha_box = std::min(alpha_box, (d_lo[i] - d[i]) / s[i]);
        }

        if(sHs <= eps * ss)
        {
            // Non-positive curvature: move to boundary
            Scalar alpha = std::min(alpha_tr, alpha_box);
            if(!std::isfinite(alpha))
                alpha = Scalar(1);
            d += alpha * s;
            break;
        }

        Scalar alpha_cg = r.dot(r) / sHs;

        if(alpha_cg >= alpha_tr || alpha_cg >= alpha_box)
        {
            // CG step would exceed boundary; truncate
            Scalar alpha = std::min(alpha_tr, alpha_box);
            d += alpha * s;
            break;
        }

        // Standard CG update
        Eigen::Vector<Scalar, N> d_new = (d + alpha_cg * s).eval();

        // Project to box (numerical safety)
        for(int i = 0; i < n; ++i)
            d_new[i] = std::clamp(d_new[i], d_lo[i], d_hi[i]);

        d = d_new;

        Scalar rr_old = r.dot(r);
        r += alpha_cg * (H * s);

        Scalar rr_new = r.dot(r);
        if(rr_old < eps * eps)
            break;

        Scalar beta = rr_new / rr_old;
        s = (-r + beta * s).eval();

        // Zero out search direction components at active bounds
        for(int i = 0; i < n; ++i)
        {
            if(d[i] <= d_lo[i] + eps && s[i] < Scalar(0))
                s[i] = Scalar(0);
            if(d[i] >= d_hi[i] - eps && s[i] > Scalar(0))
                s[i] = Scalar(0);
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

// Select which interpolation point to replace.
//
// Among non-best points, choose the one farthest from x_k whose removal
// would least degrade the interpolation set. For simplicity, we choose
// the farthest point from x_new (maximizing geometry refresh).
//
// If the new point improves the objective, prefer replacing the worst
// (highest f) point among those far from x_k. Otherwise, replace the
// farthest point.
//
// Reference: Powell 2009, Section 4.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
int select_replacement(const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& Y,
                       const Eigen::VectorX<Scalar>& f_values,
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
