#include "bound_ls_al_problems.h"

#include "argmin/detail/cauchy_point.h"
#include "argmin/detail/compact_lbfgs.h"

#include "argmin/line_search/options.h"
#include "argmin/line_search/strong_wolfe.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using defects::unbounded;

namespace
{

// x = (0, 1) with l = (0, -inf) and g = (1, 1): coordinate 0 sits on its bound with an
// inward-pointing gradient and no other coordinate has a finite breakpoint.
detail::cauchy_result<double, 2> cauchy_point_on_bound()
{
    const detail::compact_lbfgs<double, 2, 10> B{};
    detail::cauchy_point_solver<double, 2> solver{2};
    return solver.solve(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(0.0, -unbounded), Eigen::Vector2d::Constant(unbounded), B);
}

}

TEST_CASE("the generalized Cauchy point keeps a coordinate with an inward gradient on its bound", "[known-defect][!shouldfail]")
{
    CHECK(cauchy_point_on_bound().x_cauchy[0] == 0.0);
}

TEST_CASE("the generalized Cauchy point marks a coordinate held on its bound as active", "[known-defect][!shouldfail]")
{
    const std::vector<int> active = cauchy_point_on_bound().active_indices;
    CHECK(std::find(active.begin(), active.end(), 0) != active.end());
}

TEST_CASE("compact L-BFGS keeps the secant equation for a pair whose curvature it safeguards", "[known-defect][!shouldfail]")
{
    detail::compact_lbfgs<double, 2, 10> B{};
    const Eigen::Vector2d s(1.0, 0.0);
    const Eigen::Vector2d y(1e-17, 0.0);
    B.push(s, y);
    const bool pair_skipped = B.size() == 0;
    CHECK((pair_skipped || (B.multiply(s) - y).norm() <= 1e-12 * y.norm()));
}

TEST_CASE("strong Wolfe makes no more objective and slope evaluations than its budget", "[known-defect][!shouldfail]")
{
    std::int32_t calls = 0;
    const auto phi = [&calls](double t)
    {
        ++calls;
        return (t - 2.0) * (t - 2.0);
    };
    const auto dphi = [&calls](double t)
    {
        ++calls;
        return 2.0 * (t - 2.0);
    };
    line_search_options<double> opts;
    opts.max_iterations = 1;
    static_cast<void>(strong_wolfe(phi, dphi, 4.0, -4.0, opts));
    CHECK(calls <= 1);
}
