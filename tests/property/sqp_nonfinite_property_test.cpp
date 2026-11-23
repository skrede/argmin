#include "nlp_policies.h"
#include "nonfinite_support.h"
#include "generated_problems.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;
using namespace argmin::property;

TEST_CASE("the trust-region SQP solvers never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_non_finite("tr_sqp_fast", mixed_quadratic(), tr_sqp<sqp_mode::fast>{}, options);
    check_non_finite("tr_sqp_accurate", mixed_quadratic(), tr_sqp<sqp_mode::accurate>{}, options);
    check_non_finite("filter_trsqp_fast", mixed_quadratic(), filter_trsqp<sqp_mode::fast>{}, options);
    check_non_finite("filter_trsqp_accurate", mixed_quadratic(), filter_trsqp<sqp_mode::accurate>{}, options);
}

TEST_CASE("the line-search SQP solvers never report a non-finite value as success or as their point", "[property]")
{
    const solver_options<> options = cases::gradient_options();
    check_non_finite("kraft_slsqp", mixed_quadratic(), kraft_slsqp_policy<>{}, options);
    check_non_finite("filter_slsqp", mixed_quadratic(), filter_slsqp_policy<>{}, options);
    check_non_finite("nw_sqp", mixed_quadratic(), nw_sqp_policy<>{}, options);
    check_non_finite("filter_nw_sqp", mixed_quadratic(), filter_nw_sqp_policy<>{}, options);
}
