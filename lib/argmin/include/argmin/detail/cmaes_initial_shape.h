#ifndef HPP_GUARD_ARGMIN_DETAIL_CMAES_INITIAL_SHAPE_H
#define HPP_GUARD_ARGMIN_DETAIL_CMAES_INITIAL_SHAPE_H

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

template <typename Scalar, typename Bound>
Scalar widest_finite_range(const Bound& lower, const Bound& upper)
{
    Scalar widest = Scalar(0.0);
    for(Eigen::Index i = 0; i < lower.size(); ++i)
    {
        const Scalar range = upper[i] - lower[i];
        if(std::isfinite(range))
            widest = std::max(widest, range);
    }
    return widest;
}

// Hansen (2023) arXiv:1604.00772: sigma = 1/4 to 1/3 of search space.
template <typename Scalar, typename Bound>
Scalar initial_step_size(const std::optional<Scalar>& requested, bool bounded, const Bound& lower, const Bound& upper)
{
    if(requested.has_value())
        return *requested;
    if(!bounded)
        return Scalar(0.3);
    const Scalar widest = widest_finite_range<Scalar>(lower, upper);
    return widest > Scalar(0.0) ? widest / Scalar(3.0) : Scalar(0.3);
}

// Zero selects the Hansen (2023) default 4 + floor(3 ln n). The bounded floor raises it to 4n, which Hansen (2023)
// recommends for multimodal landscape coverage.
template <typename Scalar>
int population_size(const std::optional<std::uint32_t>& requested, int max_population, int n, bool bounded_floor)
{
    int lambda = requested.has_value() ? static_cast<int>(std::min<std::uint32_t>(*requested, static_cast<std::uint32_t>(max_population) + 1u)) : 0;
    if(lambda == 0 && bounded_floor)
        lambda = std::max(4 * n, static_cast<int>(4 + std::floor(Scalar(3.0) * std::log(static_cast<Scalar>(n)))));
    return lambda;
}

// Hansen (2023) arXiv:1604.00772 sets the initial per-coordinate step to about 0.3 (ub - lb). With one scalar sigma
// that scale lives in the initial covariance diagonal, D_i = range_i / max_range, so the widest finite coordinate
// keeps unit scale and an isotropic box is unchanged.
template <typename Scalar, typename State>
void scale_to_box(State& s, int n)
{
    const Scalar widest = widest_finite_range<Scalar>(s.lower, s.upper);
    if(!(widest > Scalar(0.0)))
        return;
    for(int i = 0; i < n; ++i)
    {
        const Scalar range = s.upper[i] - s.lower[i];
        const Scalar ratio = (std::isfinite(range) && range > Scalar(0.0)) ? range / widest : Scalar(1.0);
        s.D[i] = ratio;
        s.C(i, i) = ratio * ratio;
    }
}

// The TolXUp baseline (Hansen 2023, arXiv:1604.00772 section B.3 item 8) is the initial max(D).
template <typename Scalar, typename State>
void initial_shape(State& s, int n, bool box_scaled)
{
    s.C.setIdentity(n, n);
    s.p_sigma.setZero(n);
    s.p_c.setZero(n);
    s.B.setIdentity(n, n);
    s.D.setOnes(n);
    if(box_scaled)
        scale_to_box<Scalar>(s, n);
    s.eigen_solver.compute(s.C);
    s.covariance_dirty = false;
    s.initial_d_max = s.D.maxCoeff();
    s.axis_cycle_index = 0;
}

}

#endif
