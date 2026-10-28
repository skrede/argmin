#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/stochastic_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every stochastic solver result in the case table", "[oracle]")
{
    cases::margin_table margins("stochastic");
    cases::judging_visitor judge(margins);
    cases::visit_stochastic_cases(judge);
}

TEST_CASE("every listed stochastic known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_stochastic_cases(visitor); });
}

TEST_CASE("the stochastic case table visits exactly its keys", "[oracle]")
{
    cases::check_visits_exactly([](auto& visitor) { cases::visit_stochastic_cases(visitor); }, cases::stochastic_keys);
}
