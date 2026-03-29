#ifndef HPP_GUARD_NABLAPP_DETAIL_BOUND_PROJECTION_H
#define HPP_GUARD_NABLAPP_DETAIL_BOUND_PROJECTION_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

// Project x onto the box [lower, upper] component-wise.
//
// Reference: N&W Section 16.6, definition of P(x, l, u).

template <typename Scalar = double>
Eigen::VectorX<Scalar> project(const Eigen::VectorX<Scalar>& x,
                                const Eigen::VectorX<Scalar>& lower,
                                const Eigen::VectorX<Scalar>& upper)
{
    return x.cwiseMax(lower).cwiseMin(upper).eval();
}

// Compute maximum feasible step alpha such that x + alpha*d stays in [lower, upper].
//
// Returns infinity if no bound is hit (unconstrained direction).
//
// Reference: Byrd et al. 1995, N&W Section 16.6.

template <typename Scalar = double>
Scalar compute_alpha_max(const Eigen::VectorX<Scalar>& x,
                          const Eigen::VectorX<Scalar>& d,
                          const Eigen::VectorX<Scalar>& lower,
                          const Eigen::VectorX<Scalar>& upper)
{
    Scalar alpha = std::numeric_limits<Scalar>::infinity();
    const int n = x.size();

    for(int i = 0; i < n; ++i)
    {
        if(d[i] > Scalar(0) && upper[i] < std::numeric_limits<Scalar>::infinity())
            alpha = std::min(alpha, (upper[i] - x[i]) / d[i]);
        else if(d[i] < Scalar(0) && lower[i] > -std::numeric_limits<Scalar>::infinity())
            alpha = std::min(alpha, (lower[i] - x[i]) / d[i]);
    }

    return alpha;
}

}

#endif
