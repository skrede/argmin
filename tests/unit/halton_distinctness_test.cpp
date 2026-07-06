// Halton coordinate-distinctness invariant.
//
// Each dimension of a Halton sequence must use its own prime base. An
// earlier implementation reused base d%10, so coordinate pairs (d, d-10)
// shared a base and produced identical van der Corput sequences --
// collapsing multi-start seeding onto diagonals of the search box for any
// problem with more than ten dimensions. This test pins two properties
// for dimensions well past the old wrap point:
//   1. all coordinate bases are distinct, and
//   2. no two coordinate sequences coincide.
//
// Reference: Halton (1960), "On the efficiency of certain quasi-random
//            sequences of points in evaluating multi-dimensional
//            integrals", Numerische Mathematik 2(1).

#include "argmin/detail/halton.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <vector>
#include <cstdint>

using Catch::Approx;
using namespace argmin;

TEST_CASE("halton: coordinate bases are distinct for all dimensions", "[halton]")
{
    // Up to 40 dimensions (well past the old d%10 wrap), every base is a
    // distinct prime.
    constexpr int max_dim = 40;
    std::set<std::uint32_t> seen;
    std::uint32_t prev = 0;
    for(int d = 0; d < max_dim; ++d)
    {
        const std::uint32_t base = detail::halton_base(d);
        INFO("dimension " << d << " base " << base);
        // Distinct across all dimensions seen so far.
        CHECK(seen.insert(base).second);
        // Bases are strictly increasing primes (>= 2).
        CHECK(base >= 2u);
        CHECK(base > prev);
        prev = base;
    }
    CHECK(static_cast<int>(seen.size()) == max_dim);
}

TEST_CASE("halton: no two coordinate sequences coincide for dims > 10", "[halton]")
{
    // Generate the first 50 Halton points in 25 dimensions and check that
    // no two coordinate columns are identical -- in particular the
    // formerly-colliding pairs (d, d-10) and (d, d-20).
    constexpr int n = 25;
    constexpr int n_points = 50;

    std::vector<Eigen::VectorXd> pts;
    pts.reserve(n_points);
    for(int i = 0; i < n_points; ++i)
        pts.push_back(detail::halton_point(static_cast<std::uint32_t>(i), n));

    for(int a = 0; a < n; ++a)
    {
        for(int b = a + 1; b < n; ++b)
        {
            bool all_equal = true;
            for(int i = 0; i < n_points; ++i)
            {
                if(pts[static_cast<std::size_t>(i)][a]
                   != pts[static_cast<std::size_t>(i)][b])
                {
                    all_equal = false;
                    break;
                }
            }
            INFO("coordinate columns " << a << " and " << b
                 << " must not be identical sequences");
            CHECK_FALSE(all_equal);
        }
    }

    // Direct spot-check of the historical collision: dims 10 and 20 (0-indexed)
    // formerly shared base 2 via d%10 and produced identical sequences.
    CHECK(detail::halton_base(10) != detail::halton_base(0));
    CHECK(detail::halton_base(20) != detail::halton_base(0));
    CHECK(detail::halton_base(20) != detail::halton_base(10));
}
