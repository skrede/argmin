#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every line-search SQP solver result in the case table", "[oracle]")
{
    cases::margin_table margins("sqp");
    cases::judging_visitor judge(margins);
    cases::visit_sqp_cases(judge);
}

TEST_CASE("every listed line-search SQP known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_sqp_cases(visitor); });
}
