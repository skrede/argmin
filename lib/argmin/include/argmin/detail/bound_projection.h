#ifndef HPP_GUARD_ARGMIN_DETAIL_BOUND_PROJECTION_H
#define HPP_GUARD_ARGMIN_DETAIL_BOUND_PROJECTION_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

// Project x onto the box [lower, upper] component-wise.
//
// Reference: N&W Section 16.6, definition of P(x, l, u).

template <typename Scalar = double, int N = argmin::dynamic_dimension,
          typename XExpr>
Eigen::Vector<Scalar, N> project(const Eigen::MatrixBase<XExpr>& x,
                                 const Eigen::Vector<Scalar, N>& lower,
                                 const Eigen::Vector<Scalar, N>& upper)
{
    return x.cwiseMax(lower).cwiseMin(upper).eval();
}

// Compute maximum feasible step alpha such that x + alpha*d stays in [lower, upper].
//
// Returns infinity if no bound is hit (unconstrained direction).
//
// Reference: Byrd et al. 1995, N&W Section 16.6.

template <typename Scalar = double, int N = argmin::dynamic_dimension>
Scalar compute_alpha_max(const Eigen::Vector<Scalar, N>& x,
                          const Eigen::Vector<Scalar, N>& d,
                          const Eigen::Vector<Scalar, N>& lower,
                          const Eigen::Vector<Scalar, N>& upper)
{
    Scalar alpha = std::numeric_limits<Scalar>::infinity();
    const Eigen::Index n = x.size();

    for(Eigen::Index i = 0; i < n; ++i)
    {
        if(d[i] > Scalar(0) && upper[i] < std::numeric_limits<Scalar>::infinity())
            alpha = std::min(alpha, (upper[i] - x[i]) / d[i]);
        else if(d[i] < Scalar(0) && lower[i] > -std::numeric_limits<Scalar>::infinity())
            alpha = std::min(alpha, (lower[i] - x[i]) / d[i]);
    }

    return alpha;
}

// Projected gradient x - P(x - g, lower, upper) = clamp(g, x - upper, x - lower), the first-order measure of a
// bound-constrained problem (N&W 2e Section 16.7). Its norms reduce the clamp in place; a gradient no bound clips
// returns its own norm, bit-identical to the unprojected value.
template <typename Scalar, int N>
bool box_clips_gradient(const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g,
                        const Eigen::Ref<const Eigen::Vector<Scalar, N>>& x,
                        const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower,
                        const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper)
{
    return (g.array() < (x - upper).array()).any() || (g.array() > (x - lower).array()).any();
}

template <typename Scalar, int N>
Scalar projected_gradient_norm(const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g,
                               const Eigen::Ref<const Eigen::Vector<Scalar, N>>& x,
                               const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower,
                               const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper)
{
    if(!box_clips_gradient<Scalar, N>(g, x, lower, upper))
        return g.norm();
    return g.cwiseMax(x - upper).cwiseMin(x - lower).norm();
}

template <typename Scalar, int N>
Scalar projected_gradient_inf_norm(const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g,
                                   const Eigen::Ref<const Eigen::Vector<Scalar, N>>& x,
                                   const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower,
                                   const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper)
{
    if(!box_clips_gradient<Scalar, N>(g, x, lower, upper))
        return g.template lpNorm<Eigen::Infinity>();
    return g.cwiseMax(x - upper).cwiseMin(x - lower).template lpNorm<Eigen::Infinity>();
}

}

#endif
