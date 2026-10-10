#include "line_search_sqp_families.h"

#include "argmin/solver/filter_slsqp_policy.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::float_gate;

TEST_CASE("filter_slsqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_filter_family<filter_slsqp_policy>();
}
