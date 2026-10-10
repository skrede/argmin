#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_LINE_SEARCH_SQP_FAMILIES_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_LINE_SEARCH_SQP_FAMILIES_H

#include "sqp_float_gate.h"

#include "argmin/test_functions/small_dense.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

namespace argmin::float_gate
{

template <template <scalar, int> typename Policy, template <typename> typename Problem>
void check_line_search_tracks_double()
{
    check_sqp_tracks_double<Policy, Problem>(k_line_search_sqp);
}

template <template <scalar, int> typename Policy>
void check_line_search_family()
{
    SECTION("hs071")
    {
        check_line_search_tracks_double<Policy, hs071>();
    }

    SECTION("hs035")
    {
        check_line_search_tracks_double<Policy, hs035>();
    }

    SECTION("hs028")
    {
        check_line_search_tracks_double<Policy, hs028>();
    }

    SECTION("sd012")
    {
        check_line_search_tracks_double<Policy, sd012>();
    }
}

template <template <scalar, int> typename Policy>
void check_filter_family()
{
    SECTION("hs071")
    {
        check_line_search_tracks_double<Policy, hs071>();
    }

    SECTION("hs024")
    {
        check_line_search_tracks_double<Policy, hs024>();
    }

    SECTION("hs043")
    {
        check_line_search_tracks_double<Policy, hs043>();
    }

    SECTION("sd024")
    {
        check_line_search_tracks_double<Policy, sd024>();
    }
}

}

#endif
