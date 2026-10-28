#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/derivative_free_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every derivative-free solver result in the case table", "[oracle]")
{
    cases::margin_table margins("derivative_free");
    cases::judging_visitor judge(margins);
    cases::visit_derivative_free_cases(judge);
}

TEST_CASE("every listed derivative-free known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_derivative_free_cases(visitor); });
}

TEST_CASE("the derivative-free case table visits exactly its keys", "[oracle]")
{
    cases::check_visits_exactly([](auto& visitor) { cases::visit_derivative_free_cases(visitor); }, cases::derivative_free_keys);
}
