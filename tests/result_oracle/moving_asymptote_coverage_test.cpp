#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/moving_asymptote_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every moving-asymptote solver result in the case table", "[oracle]")
{
    cases::margin_table margins("moving_asymptote");
    cases::judging_visitor judge(margins);
    cases::visit_moving_asymptote_cases(judge);
}

TEST_CASE("every listed moving-asymptote known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_moving_asymptote_cases(visitor); });
}

TEST_CASE("the moving-asymptote case table visits exactly its keys", "[oracle]")
{
    cases::check_visits_exactly([](auto& visitor) { cases::visit_moving_asymptote_cases(visitor); }, cases::moving_asymptote_keys);
}
