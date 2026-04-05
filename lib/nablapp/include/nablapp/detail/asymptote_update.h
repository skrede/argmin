#ifndef HPP_GUARD_NABLAPP_DETAIL_ASYMPTOTE_UPDATE_H
#define HPP_GUARD_NABLAPP_DETAIL_ASYMPTOTE_UPDATE_H

// Moving asymptote update heuristic for MMA/GCMMA.
//
// Updates lower and upper asymptotes based on oscillation detection
// of the design variables across consecutive iterations. When variables
// oscillate (sign change in consecutive differences), asymptotes are
// contracted to tighten the approximation. When variables change
// monotonically, asymptotes are expanded for a more aggressive step.
//
// Reference: Svanberg 1987, Section 3.

#include "nablapp/types.h"
#include "nablapp/options/asymptote_options.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// Update moving asymptotes L and U in-place.
//
// L, U:        lower/upper asymptotes (n), modified in-place
// x:           current iterate (n)
// x_old1:      previous iterate (n)
// x_old2:      iterate two steps back (n)
// x_min, x_max: variable bounds (n)
// iteration:   current outer iteration index (0-based)
// asyminit:    initial asymptote distance as fraction of range
// asymdec:     contraction factor on oscillation (< 1)
// asyminc:     expansion factor on monotone change (> 1)
//
// Reference: Svanberg 1987, Section 3.
template <typename Scalar, int N = nablapp::dynamic_dimension>
void update_asymptotes(
    Eigen::Vector<Scalar, N>& L,
    Eigen::Vector<Scalar, N>& U,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& x_old1,
    const Eigen::Vector<Scalar, N>& x_old2,
    const Eigen::Vector<Scalar, N>& x_min,
    const Eigen::Vector<Scalar, N>& x_max,
    int iteration,
    Scalar asyminit = Scalar(0.5),
    Scalar asymdec = Scalar(0.7),
    Scalar asyminc = Scalar(1.2),
    const asymptote_options& opts = {})
{
    const int n = static_cast<int>(x.size());
    L.resize(n);
    U.resize(n);

    for(int j = 0; j < n; ++j)
    {
        Scalar range = x_max[j] - x_min[j];
        if(range <= Scalar(0))
            range = Scalar(1);

        // Oscillation detection: sign of product of consecutive differences.
        // For iterations 0-1, d2 is typically zero (x_old2 == x_old1 or
        // x_old1 == x0), giving gamma = 1 (no change). The asymptotes
        // recenter on the current iterate preserving their init half-width.
        //
        // Reference: Svanberg 1987, Section 3.
        Scalar d1 = x[j] - x_old1[j];
        Scalar d2 = x_old1[j] - x_old2[j];
        Scalar product = d1 * d2;

        Scalar gamma;
        if(product < Scalar(0))
            gamma = asymdec;
        else if(product > Scalar(0))
            gamma = asyminc;
        else
            gamma = Scalar(1);

        L[j] = x[j] - gamma * (x_old1[j] - L[j]);
        U[j] = x[j] + gamma * (U[j] - x_old1[j]);

        // Safety clamp: asymptotes must not get too close to iterate.
        // Minimum distance is minimum_distance_fraction * range (Svanberg 1987).
        Scalar min_dist = static_cast<Scalar>(opts.minimum_distance_fraction.value_or(0.01)) * range;
        L[j] = std::min(L[j], x[j] - min_dist);
        U[j] = std::max(U[j], x[j] + min_dist);
    }
}

}

#endif
