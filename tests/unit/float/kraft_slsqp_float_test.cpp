#include "line_search_sqp_families.h"

#include "argmin/solver/kraft_slsqp_policy.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::float_gate;

TEST_CASE("kraft_slsqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_line_search_family<kraft_slsqp_policy>();
}
