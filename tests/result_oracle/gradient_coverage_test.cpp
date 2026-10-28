#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/gradient_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every gradient-based bound solver result in the case table", "[oracle]")
{
    cases::margin_table margins("gradient");
    cases::judging_visitor judge(margins);
    cases::visit_gradient_cases(judge);
}

TEST_CASE("every listed gradient-based known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_gradient_cases(visitor); });
}

TEST_CASE("the gradient case table visits exactly its keys", "[oracle]")
{
    cases::check_visits_exactly([](auto& visitor) { cases::visit_gradient_cases(visitor); }, cases::gradient_keys);
}
