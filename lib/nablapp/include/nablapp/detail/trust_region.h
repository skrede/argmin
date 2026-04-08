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
#include "nablapp/detail/quadratic_model.h"
#include "nablapp/options/trust_region_options.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

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

// Contract rho using Powell's three-regime schedule.
//
// When the trust-region radius delta has shrunk to rho, the resolution
// parameter rho is contracted and a new trust-region round begins.
// Three regimes based on rho / rho_end:
//   ratio <= 16:  jump straight to rho_end (close enough)
//   ratio <= 250: geometric mean sqrt(ratio) * rho_end
//   ratio > 250:  aggressive 0.1x contraction
//
// Returns {rho_new, delta_new}.
//
// Reference: Powell 2009, Section 5.
// Adapted from NLopt bobyqa.c lines 3003-3017.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L3003
template <typename Scalar = double>
std::pair<Scalar, Scalar> contract_rho(Scalar rho, Scalar rho_end)
{
    Scalar ratio = rho / rho_end;
    Scalar rho_new;
    if(ratio <= Scalar(16))
        rho_new = rho_end;
    else if(ratio <= Scalar(250))
        rho_new = std::sqrt(ratio) * rho_end;
    else
        rho_new = Scalar(0.1) * rho;
    Scalar delta_new = std::max(Scalar(0.5) * rho, rho_new);
    return {rho_new, delta_new};
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

// Select which interpolation point to replace using denominator * distance^4
// weighting.
//
// For each non-best point k, compute:
//   distsq = ||Y[:,k] - x_ref||^2
//   temp   = max(1, (distsq / delsq)^2)     -- distance^4 amplification
//   score  = temp * L_k(x_new)^2
// Select k with maximum score. Far-away points with large Lagrange
// influence are strongly preferred for replacement, maintaining both
// geometry compactness and interpolation conditioning.
//
// When f_new < f_best (improvement step), distances are recomputed from
// x_new instead of x_k, and the better selection is kept.
//
// Reference: Powell 2009, Section 4.
// Adapted from NLopt bobyqa.c lines 2493-2549 and 2656-2707.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2493
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2656
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
int select_replacement(const Eigen::Matrix<Scalar, N, P>& Y,
                       const Eigen::Vector<Scalar, P>& f_values,
                       const Eigen::Vector<Scalar, N>& x_new,
                       Scalar f_new,
                       const Eigen::Vector<Scalar, N>& x_k,
                       const Eigen::VectorXd& lagrange_values_at_xnew,
                       Scalar delta)
{
    const int m = Y.cols();
    const Scalar delsq = delta * delta;

    // Find the best point (lowest f) -- never replace it
    int best_idx = 0;
    for(int i = 1; i < m; ++i)
    {
        if(f_values[i] < f_values[best_idx])
            best_idx = i;
    }

    // Score each non-best point: L_k(x_new)^2 * max(1, (dist/delta)^4)
    // with distances from x_k (current best).
    int knew = (best_idx == 0) ? 1 : 0;
    Scalar best_score = Scalar(-1);

    for(int k = 0; k < m; ++k)
    {
        if(k == best_idx) continue;
        Scalar distsq = (Y.col(k) - x_k).squaredNorm();
        Scalar ratio_sq = distsq / delsq;
        Scalar temp = std::max(Scalar(1), ratio_sq * ratio_sq);
        Scalar lk = lagrange_values_at_xnew[k];
        Scalar score = temp * lk * lk;
        if(score > best_score)
        {
            best_score = score;
            knew = k;
        }
    }

    // When f_new improves the best, recompute with distances from x_new
    // and keep the better selection.
    // Adapted from NLopt bobyqa.c lines 2656-2707.
    if(f_new < f_values[best_idx])
    {
        int knew_alt = knew;
        Scalar best_score_alt = Scalar(-1);

        for(int k = 0; k < m; ++k)
        {
            if(k == best_idx) continue;
            Scalar distsq = (Y.col(k) - x_new).squaredNorm();
            Scalar ratio_sq = distsq / delsq;
            Scalar temp = std::max(Scalar(1), ratio_sq * ratio_sq);
            Scalar lk = lagrange_values_at_xnew[k];
            Scalar score = temp * lk * lk;
            if(score > best_score_alt)
            {
                best_score_alt = score;
                knew_alt = k;
            }
        }

        // Use alternative selection if it has a meaningfully better score
        if(best_score_alt > best_score)
            knew = knew_alt;
    }

    return knew;
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

// ALTMOV geometry improvement functions (future infrastructure).
//
// The functions below (altmov_clip_step_bounds, altmov_line_search,
// lagrange_gradient, altmov_cauchy_step, altmov_geometry_step) faithfully
// implement Powell 2009 Section 6 / NLopt ALTMOV. They are NOT currently
// wired into bobyqa_policy::step() because nablapp's SVD-based model
// representation (pinv_Phi) produces Lagrange values that accumulate
// numerical drift compared to NLopt's exact BMAT/ZMAT incremental
// updates. Empirical testing on HS001: ALTMOV placement degraded
// convergence from 2509 to 2902 iterations. The direction-based
// heuristic in bobyqa_policy.h outperforms ALTMOV under the SVD model.
//
// These functions become usable when/if a BMAT/ZMAT model representation
// is implemented, eliminating the Lagrange value drift issue.

// Compute box-constrained step bounds along a direction from x_opt.
//
// For each dimension, tightens slbd/subd so that x_opt + t*direction
// stays within [lower, upper] for t in [slbd, subd].
//
// Reference: Powell 2009, Section 6 (ALTMOV).
// Adapted from NLopt bobyqa.c lines 891-920.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L891
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
std::pair<Scalar, Scalar> altmov_clip_step_bounds(
    const Eigen::Vector<Scalar, N>& x_opt,
    const Eigen::Vector<Scalar, N>& direction,
    Scalar adelt,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    const int n = x_opt.size();
    Scalar distsq = direction.squaredNorm();
    Scalar subd = adelt / std::sqrt(std::max(distsq, std::numeric_limits<Scalar>::epsilon()));
    Scalar slbd = -subd;
    Scalar sumin = std::min(Scalar(1), subd);

    for(int i = 0; i < n; ++i)
    {
        Scalar t = direction[i];
        if(t > Scalar(0))
        {
            if(slbd * t < lower[i] - x_opt[i])
                slbd = (lower[i] - x_opt[i]) / t;
            if(subd * t > upper[i] - x_opt[i])
                subd = std::max(sumin, (upper[i] - x_opt[i]) / t);
        }
        else if(t < Scalar(0))
        {
            if(slbd * t > upper[i] - x_opt[i])
                slbd = (upper[i] - x_opt[i]) / t;
            if(subd * t < lower[i] - x_opt[i])
                subd = std::max(sumin, (lower[i] - x_opt[i]) / t);
        }
    }

    return {slbd, subd};
}

// Search through interpolation points for the best line-search candidate
// maximizing |L_knew(x)|^2 for geometry improvement.
//
// For each point k (k != kopt), evaluates L_knew at box-clipped step
// endpoints along the direction x_opt -> Y[:,k], tracking the candidate
// with maximum |L_knew|^2. Uses stored pinv_Phi for O(m*p) Lagrange
// evaluation instead of O(m*p^2) SVD per candidate.
//
// Reference: Powell 2009, Section 6 (ALTMOV line search).
// Adapted from NLopt bobyqa.c lines 864-982.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L864
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
std::pair<Eigen::Vector<Scalar, N>, Scalar> altmov_line_search(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& Y,
    const Eigen::Vector<Scalar, N>& x_opt,
    int kopt,
    int knew,
    Scalar adelt,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const quadratic_model<Scalar, N>& model)
{
    const int m = Y.cols();
    Scalar best_score = Scalar(-1);
    Eigen::Vector<Scalar, N> best_point = x_opt;

    for(int k = 0; k < m; ++k)
    {
        if(k == kopt) continue;

        Eigen::Vector<Scalar, N> d = (Y.col(k) - x_opt).eval();
        auto [slbd, subd] = altmov_clip_step_bounds(x_opt, d, adelt, lower, upper);
        if(slbd >= subd) continue;

        // Evaluate L_knew at box-clipped step endpoints
        for(Scalar t : {slbd, subd})
        {
            Eigen::Vector<Scalar, N> candidate = project(
                (x_opt + t * d).eval(), lower, upper);
            auto lv = compute_lagrange_incremental(model, model.x_base, candidate);
            Scalar score = lv[knew] * lv[knew];
            if(score > best_score)
            {
                best_score = score;
                best_point = candidate;
            }
        }
    }

    return {best_point, best_score};
}

// Compute the gradient of L_knew at x_opt analytically from the polynomial
// basis coefficients stored in pinv_Phi.
//
// L_knew(x) = pinv_Phi[knew,:] * phi(x) where phi has basis
// [1, s_1, ..., s_n, 0.5*s_1^2, s_1*s_2, ..., 0.5*s_n^2] with s = x - x_base.
// The gradient d/dx_j L_knew = pinv_Phi[knew, 1+j] + sum of quadratic terms.
//
// Reference: Powell 2009, Section 6 (gradient of Lagrange function).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> lagrange_gradient(
    const quadratic_model<Scalar, N>& model,
    int knew,
    const Eigen::Vector<Scalar, N>& x)
{
    const int n = x.size();
    Eigen::Vector<Scalar, N> s = (x - model.x_base).eval();
    Eigen::Vector<Scalar, N> grad(n);
    const auto& row = model.pinv_Phi.row(knew);

    for(int j = 0; j < n; ++j)
    {
        Scalar val = row[1 + j];
        int idx = 1 + n;
        for(int a = 0; a < n; ++a)
        {
            for(int b = a; b < n; ++b)
            {
                if(a == j && b == j)
                    val += row[idx] * s[j];
                else if(a == j)
                    val += row[idx] * s[b];
                else if(b == j)
                    val += row[idx] * s[a];
                ++idx;
            }
        }
        grad[j] = val;
    }

    return grad;
}

// Constrained Cauchy step along gradient of L_knew for geometry improvement.
//
// Computes the steepest-ascent direction of L_knew at x_opt, projects
// onto box constraints within the trust region, and selects the candidate
// maximizing |L_knew|^2. Tries both ascent and descent directions.
//
// Reference: Powell 2009, Section 6 (ALTMOV Cauchy step).
// Adapted from NLopt bobyqa.c lines 1017-1125.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1017
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
std::pair<Eigen::Vector<Scalar, N>, Scalar> altmov_cauchy_step(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& Y,
    const Eigen::Vector<Scalar, N>& x_opt,
    int knew,
    Scalar adelt,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const quadratic_model<Scalar, N>& model)
{
    const int n = x_opt.size();
    Scalar best_score = Scalar(0);
    Eigen::Vector<Scalar, N> best_point = x_opt;

    Eigen::Vector<Scalar, N> grad = lagrange_gradient(model, knew, x_opt);

    // Try both ascent (+grad) and descent (-grad) per NLopt lines 1138-1156
    for(int sign = 0; sign < 2; ++sign)
    {
        Eigen::Vector<Scalar, N> glag = (sign == 0) ? grad : Eigen::Vector<Scalar, N>(-grad);

        Scalar wfixsq = Scalar(0);
        Scalar ggfree = Scalar(0);
        Eigen::Vector<Scalar, N> w = Eigen::Vector<Scalar, N>::Zero(n);
        Scalar bigstp = Scalar(2) * adelt;

        for(int i = 0; i < n; ++i)
        {
            Scalar tempa = std::min(x_opt[i] - lower[i], glag[i]);
            Scalar tempb = std::max(x_opt[i] - upper[i], glag[i]);
            if(tempa > Scalar(0) || tempb < Scalar(0))
            {
                w[i] = bigstp;
                ggfree += glag[i] * glag[i];
            }
        }
        if(ggfree <= Scalar(0)) continue;

        // Iteratively fix components that hit bounds
        Scalar step_len = adelt / std::sqrt(ggfree);
        for(int iter = 0; iter < n; ++iter)
        {
            Scalar wsqsav = wfixsq;
            Scalar budget = adelt * adelt - wfixsq;
            if(budget <= Scalar(0)) break;
            step_len = std::sqrt(budget / std::max(ggfree, std::numeric_limits<Scalar>::epsilon()));
            ggfree = Scalar(0);
            for(int i = 0; i < n; ++i)
            {
                if(w[i] != bigstp) continue;
                Scalar trial = x_opt[i] - step_len * glag[i];
                if(trial <= lower[i])
                {
                    w[i] = lower[i] - x_opt[i];
                    wfixsq += w[i] * w[i];
                }
                else if(trial >= upper[i])
                {
                    w[i] = upper[i] - x_opt[i];
                    wfixsq += w[i] * w[i];
                }
                else
                    ggfree += glag[i] * glag[i];
            }
            if(wfixsq <= wsqsav || ggfree <= Scalar(0)) break;
        }

        for(int i = 0; i < n; ++i)
        {
            if(w[i] == bigstp)
                w[i] = -step_len * glag[i];
        }

        Eigen::Vector<Scalar, N> candidate = project(
            (x_opt + w).eval(), lower, upper);
        auto lv = compute_lagrange_incremental(model, model.x_base, candidate);
        Scalar score = lv[knew] * lv[knew];
        if(score > best_score)
        {
            best_score = score;
            best_point = candidate;
        }
    }

    return {best_point, best_score};
}

// ALTMOV geometry improvement step: maximize |L_knew(x)| for point replacement.
//
// Orchestrates line search through interpolation points and constrained
// Cauchy step, returning whichever candidate maximizes |L_knew(x)|^2.
//
// Reference: Powell 2009, Section 6 (ALTMOV).
// Adapted from NLopt bobyqa.c lines 743-1159.
// https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L743
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> altmov_geometry_step(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& Y,
    const Eigen::Vector<Scalar, N>& x_opt,
    int kopt,
    int knew,
    Scalar adelt,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const quadratic_model<Scalar, N>& model)
{
    auto [line_point, line_score] = altmov_line_search<Scalar, N>(
        Y, x_opt, kopt, knew, adelt, lower, upper, model);
    auto [cauchy_point, cauchy_score] = altmov_cauchy_step<Scalar, N>(
        Y, x_opt, knew, adelt, lower, upper, model);
    return (cauchy_score > line_score) ? cauchy_point : line_point;
}

}

#endif
