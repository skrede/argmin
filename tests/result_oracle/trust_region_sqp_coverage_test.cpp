#include "coverage_visitors.h"

#include "margin_table.h"

#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("oracle accepts every trust-region SQP solver result in the case table", "[oracle]")
{
    cases::margin_table margins("trust_region_sqp");
    cases::judging_visitor judge(margins);
    cases::visit_trust_region_sqp_cases(judge);
}

TEST_CASE("every listed trust-region SQP known defect matches a case in the table", "[oracle]")
{
    cases::check_known_defects_match([](auto& visitor) { cases::visit_trust_region_sqp_cases(visitor); });
}
