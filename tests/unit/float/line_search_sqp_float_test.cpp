#include "sqp_float_gate.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"

#include "argmin/test_functions/small_dense.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

template <template <scalar, int> typename Policy, template <typename> typename Problem>
void check_tracks_double()
{
    check_sqp_tracks_double<Policy, Problem>(k_line_search_sqp);
}

template <template <scalar, int> typename Policy>
void check_family()
{
    SECTION("hs071")
    {
        check_tracks_double<Policy, hs071>();
    }

    SECTION("hs035")
    {
        check_tracks_double<Policy, hs035>();
    }

    SECTION("hs028")
    {
        check_tracks_double<Policy, hs028>();
    }

    SECTION("sd012")
    {
        check_tracks_double<Policy, sd012>();
    }
}

template <template <scalar, int> typename Policy>
void check_filter_family()
{
    SECTION("hs071")
    {
        check_tracks_double<Policy, hs071>();
    }

    SECTION("hs024")
    {
        check_tracks_double<Policy, hs024>();
    }

    SECTION("hs043")
    {
        check_tracks_double<Policy, hs043>();
    }

    SECTION("sd024")
    {
        check_tracks_double<Policy, sd024>();
    }
}

}

TEST_CASE("nw_sqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_family<nw_sqp_policy>();
}

TEST_CASE("kraft_slsqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_family<kraft_slsqp_policy>();
}

TEST_CASE("filter SQP policies in float track double on constrained Hock-Schittkowski fixtures at their own defaults", "[float]")
{
    SECTION("filter_slsqp")
    {
        check_filter_family<filter_slsqp_policy>();
    }

    SECTION("filter_nw_sqp")
    {
        check_filter_family<filter_nw_sqp_policy>();
    }
}
