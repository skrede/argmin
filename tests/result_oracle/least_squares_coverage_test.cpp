#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/least_squares_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every least-squares solver result in the case table", "[oracle]")
{
    cases::margin_table margins("least_squares");
    cases::judging_visitor judge(margins);
    cases::visit_least_squares_cases(judge);
}

TEST_CASE("every listed least-squares known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_least_squares_cases(visitor); });
}
