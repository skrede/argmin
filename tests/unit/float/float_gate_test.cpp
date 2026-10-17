#include "float_gate.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace argmin;
using namespace argmin::float_gate;

TEST_CASE("the float gate predicate rejects a wrong answer and a non-finite result", "[float]")
{
    vector<double> xd(3);
    xd << 0.25, -1.5, 2.0;
    const double displacement = 2.0 * tolerance(k_qp) * std::max(1.0, xd.norm());

    SECTION("an exact cast passes")
    {
        const vector<float> xf = xd.cast<float>();
        CHECK(tracks(xf, xd, k_qp));
    }

    SECTION("a result displaced by twice the tolerance fails")
    {
        vector<double> shifted = xd;
        shifted[1] += displacement;
        const vector<float> xf = shifted.cast<float>();
        CHECK(relative_gap(xf, xd) > tolerance(k_qp));
        CHECK_FALSE(tracks(xf, xd, k_qp));
    }

    SECTION("a result holding a quiet NaN fails")
    {
        vector<float> xf = xd.cast<float>();
        xf[2] = std::numeric_limits<float>::quiet_NaN();
        CHECK_FALSE(tracks(xf, xd, k_qp));
    }

    SECTION("an objective displaced by twice the tolerance fails")
    {
        const double fd = -3.75;
        const double ff = fd + 2.0 * tolerance(k_qp) * std::abs(fd);
        CHECK(objective_tracks(static_cast<float>(fd), fd, k_qp));
        CHECK_FALSE(objective_tracks(static_cast<float>(ff), fd, k_qp));
    }
}
