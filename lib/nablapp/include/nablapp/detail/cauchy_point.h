#ifndef HPP_GUARD_NABLAPP_DETAIL_CAUCHY_POINT_H
#define HPP_GUARD_NABLAPP_DETAIL_CAUCHY_POINT_H

#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nablapp::detail
{

// Result of the generalized Cauchy point computation.

template <typename Scalar = double>
struct cauchy_result
{
    Eigen::VectorX<Scalar> x_cauchy;
    std::vector<int> free_indices;    // indices NOT at bounds after GCP
    std::vector<int> active_indices;  // indices AT bounds after GCP
};

// Generalized Cauchy Point via breakpoint search along the projected gradient path.
//
// Walks the piecewise-linear path x(t) = P(x - t*g, l, u) and finds the first
// local minimum of the quadratic model m(x(t)) = f + g^T d(t) + 0.5 d(t)^T B d(t)
// where d(t) = x(t) - x.
//
// Reference: N&W Section 16.6, pp. 475-477, eq. 16.44-16.48.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

template <typename Scalar = double>
cauchy_result<Scalar> cauchy_point(
    const Eigen::VectorX<Scalar>& x,
    const Eigen::VectorX<Scalar>& g,
    const Eigen::VectorX<Scalar>& lower,
    const Eigen::VectorX<Scalar>& upper,
    const compact_lbfgs<Scalar>& B)
{
    const int n = x.size();
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

    // Breakpoint struct for sorting
    struct breakpoint
    {
        int index;
        Scalar t;
    };

    // Step 1: compute breakpoints (N&W eq. 16.46)
    std::vector<breakpoint> bps;
    bps.reserve(n);

    for(int i = 0; i < n; ++i)
    {
        Scalar ti;
        if(g[i] < Scalar(0) && upper[i] < inf)
            ti = (x[i] - upper[i]) / g[i];
        else if(g[i] > Scalar(0) && lower[i] > -inf)
            ti = (x[i] - lower[i]) / g[i];
        else
            continue;  // ti = +inf, variable never hits bound

        if(ti > Scalar(0))
            bps.push_back({i, ti});
    }

    // Unconstrained fallback (D-04): no finite breakpoints, all variables free.
    if(bps.empty())
    {
        // d = -g (steepest descent direction), find minimum of quadratic along d.
        Eigen::VectorX<Scalar> d = -g;
        Eigen::VectorX<Scalar> Bd = B.multiply(d);
        Scalar dTBd = d.dot(Bd);
        Scalar gTd = g.dot(d);

        Scalar t_star = Scalar(0);
        if(dTBd > Scalar(0))
            t_star = -gTd / dTBd;
        else if(gTd < Scalar(0))
            t_star = Scalar(1);  // Negative curvature: take unit step

        Eigen::VectorX<Scalar> x_cauchy = x + t_star * d;

        std::vector<int> free_idx(n);
        std::iota(free_idx.begin(), free_idx.end(), 0);

        return cauchy_result<Scalar>{
            .x_cauchy = std::move(x_cauchy),
            .free_indices = std::move(free_idx),
            .active_indices = {},
        };
    }

    // Step 2: sort breakpoints by t
    std::sort(bps.begin(), bps.end(),
              [](const auto& a, const auto& b) { return a.t < b.t; });

    // Step 3: walk the piecewise-linear path, tracking f'(t) and f''(t)
    // Initialize direction: d_i = -g_i for free components
    Eigen::VectorX<Scalar> d = -g;

    // Check which components are already at bounds at t=0
    for(int i = 0; i < n; ++i)
    {
        if(x[i] <= lower[i] && g[i] > Scalar(0))
            d[i] = Scalar(0);
        else if(x[i] >= upper[i] && g[i] < Scalar(0))
            d[i] = Scalar(0);
    }

    Scalar f_prime = g.dot(d);
    Eigen::VectorX<Scalar> Bd = B.multiply(d);
    Scalar f_double_prime = -d.dot(Bd);

    Scalar t_old = Scalar(0);

    for(const auto& bp : bps)
    {
        Scalar dt = bp.t - t_old;

        // Check if minimum is in the current interval [t_old, bp.t]
        if(f_prime * dt + Scalar(0.5) * f_double_prime * dt * dt >= Scalar(0)
           || (f_double_prime > Scalar(0) && f_prime < Scalar(0)))
        {
            // Minimum at t* = t_old - f'/f'' if f'' > 0
            if(f_double_prime > Scalar(0))
            {
                Scalar t_star = t_old - f_prime / f_double_prime;
                if(t_star > t_old && t_star < bp.t)
                {
                    Eigen::VectorX<Scalar> x_cauchy = project(x - t_star * g, lower, upper);
                    return classify_indices(x_cauchy, lower, upper);
                }
            }
        }

        // Update f' at the breakpoint
        f_prime += dt * f_double_prime;

        // If f' >= 0 at this breakpoint, the minimum is here
        if(f_prime >= Scalar(0))
        {
            Eigen::VectorX<Scalar> x_cauchy = project(x - bp.t * g, lower, upper);
            return classify_indices(x_cauchy, lower, upper);
        }

        // Variable bp.index becomes fixed at its bound
        d[bp.index] = Scalar(0);
        t_old = bp.t;

        // Recompute f' and f'' with updated d (simpler approach per plan)
        f_prime = g.dot(d);
        Bd = B.multiply(d);
        f_double_prime = -d.dot(Bd);

        // Adjust f' for position along the path
        // At position t_old, the model derivative is g^T d + t_old * d^T B d
        // but since we projected, we use the incremental form
        f_prime = g.dot(d) + t_old * (-f_double_prime);
    }

    // After all breakpoints: minimum is at or beyond the last breakpoint
    Scalar t_last = bps.back().t;
    Eigen::VectorX<Scalar> x_cauchy = project(x - t_last * g, lower, upper);
    return classify_indices(x_cauchy, lower, upper);
}

// Classify indices into free and active sets based on x_cauchy position.
template <typename Scalar = double>
cauchy_result<Scalar> classify_indices(const Eigen::VectorX<Scalar>& x_cauchy,
                                       const Eigen::VectorX<Scalar>& lower,
                                       const Eigen::VectorX<Scalar>& upper)
{
    const int n = x_cauchy.size();
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    std::vector<int> free_idx;
    std::vector<int> active_idx;
    free_idx.reserve(n);
    active_idx.reserve(n);

    for(int i = 0; i < n; ++i)
    {
        if(x_cauchy[i] <= lower[i] + eps * std::abs(lower[i])
           || x_cauchy[i] >= upper[i] - eps * std::abs(upper[i]))
            active_idx.push_back(i);
        else
            free_idx.push_back(i);
    }

    return cauchy_result<Scalar>{
        .x_cauchy = x_cauchy,
        .free_indices = std::move(free_idx),
        .active_indices = std::move(active_idx),
    };
}

}

#endif
