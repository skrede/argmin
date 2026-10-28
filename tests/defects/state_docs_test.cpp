#include "argmin/detail/steihaug_cg.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

// The cold pass starts at p = 0 on the displaced lower face of coordinate 0, and the first direction's component
// there is roundoff against a unit component in coordinate 1.
std::int32_t cold_pass_blocking_coordinate()
{
    const Eigen::Vector2d g{1e-20, -1.0};
    const Eigen::Vector2d lower{0.0, -10.0};
    const Eigen::Vector2d upper{10.0, 10.0};
    Eigen::Vector2d p, r, d, Bd;
    std::int32_t blocking = -1;
    const auto identity = [](const auto& v, auto& out) { out = v; };
    detail::steihaug_cg<double, 2>(g, identity, 10.0, 1e-8, lower, upper, 10, p, r, d, Bd, detail::identity_null_projector{}, &blocking, false);
    return blocking;
}

}

TEST_CASE("a roundoff-scale direction component does not block the cold Steihaug pass at a box face", "[known-defect][!shouldfail]")
{
    CHECK(cold_pass_blocking_coordinate() != 0);
}
