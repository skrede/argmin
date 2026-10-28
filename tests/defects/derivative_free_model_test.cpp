#include "argmin/detail/trust_region.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/interpolation_system.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>
#include <numbers>
#include <algorithm>

using namespace argmin;

namespace
{

using system2 = detail::interpolation_system<double, 2>;

double model_value(const Eigen::Vector2d& g, const Eigen::Matrix2d& H, const Eigen::Vector2d& d)
{
    return g.dot(d) + 0.5 * d.dot(H * d);
}

// The unconstrained minimizer lies outside the unit disk, so the minimum over the half disk d0 >= 0 is on its arc.
double half_disk_minimum(const Eigen::Vector2d& g, const Eigen::Matrix2d& H)
{
    double best = std::numeric_limits<double>::infinity();
    for(std::int32_t k = 0; k <= 200000; ++k)
    {
        const double angle = std::numbers::pi * (k / 200000.0 - 0.5);
        best = std::min(best, model_value(g, H, Eigen::Vector2d(std::cos(angle), std::sin(angle))));
    }
    return best;
}

system2 bootstrapped_unit_box()
{
    const auto f = [](const Eigen::Vector2d& x) { return (x[0] - 0.7) * (x[0] - 0.7) + (x[1] - 0.4) * (x[1] - 0.4) + 0.3 * x[0] * x[1]; };
    return detail::bootstrap_interpolation_system<double, 2>(Eigen::Vector2d::Zero(), 0.25, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), f);
}

double segment_maximum(const system2& sys, std::int32_t knew, std::int32_t k, double adelt, const Eigen::Vector2d& lower, const Eigen::Vector2d& upper)
{
    const Eigen::Vector2d direction = sys.xpt.col(k).head(2) - sys.xopt;
    const auto [from, to] = detail::altmov_clip_step_bounds<double, 2>(sys.xopt, direction, adelt, lower, upper);
    double best = -1.0;
    for(std::int32_t t = 0; from < to && t <= 1000; ++t)
    {
        const Eigen::Vector2d point = detail::project((sys.xopt + (from + (to - from) * t / 1000.0) * direction).eval(), lower, upper);
        const auto lagrange = detail::compute_lagrange_at(sys, point);
        best = std::max(best, lagrange[knew] * lagrange[knew]);
    }
    return best;
}

double dense_line_maximum(const system2& sys, std::int32_t knew, double adelt, const Eigen::Vector2d& lower, const Eigen::Vector2d& upper)
{
    double best = -1.0;
    for(std::int32_t k = 0; k < sys.m_points; ++k)
        if(k != sys.kopt)
            best = std::max(best, segment_maximum(sys, knew, k, adelt, lower, upper));
    return best;
}

}

TEST_CASE("the BOBYQA box trust-region step reaches the constrained model minimum when a bound coordinate should be released", "[known-defect][!shouldfail]")
{
    const Eigen::Vector2d g(1.0, -10.0);
    Eigen::Matrix2d H;
    H << 1.0, -2.0, -2.0, 5.0;
    const Eigen::Vector2d d = detail::solve_trust_region_box<double, 2>(g, H, Eigen::Vector2d::Zero(), 1.0, Eigen::Vector2d(0.0, -10.0), Eigen::Vector2d::Constant(10.0));
    CHECK(model_value(g, H, d) <= half_disk_minimum(g, H) + 1e-3);
}

TEST_CASE("the BOBYQA alternative-move line search finds the largest Lagrange value along each admissible segment", "[known-defect][!shouldfail]")
{
    const system2 sys = bootstrapped_unit_box();
    const Eigen::Vector2d lower = -sys.xbase;
    const Eigen::Vector2d upper = Eigen::Vector2d::Ones() - sys.xbase;
    const auto [point, score] = detail::altmov_line_search<double, 2>(sys, 4, 0.25, lower, upper);
    CHECK(score >= dense_line_maximum(sys, 4, 0.25, lower, upper) * (1.0 - 1e-12));
}
