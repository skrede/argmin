#include "cases/case_verdict.h"
#include "cases/derivative_free_cases.h"

#include "oracle/result_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <optional>

using namespace argmin;

namespace
{

const auto visit_family = [](auto& visitor) { cases::visit_derivative_free_cases(visitor); };

}

TEST_CASE("cobyla stays inside the box on the bound-only Wood problem", "[derivative_free]")
{
    const std::optional<oracle::verdict<double>> v = cases::judge_case(visit_family, "cobyla", "bound");
    REQUIRE(v.has_value());
    CHECK(v->legs.bound_violation == 0.0);
    CHECK(v->ok);
}

TEST_CASE("cobyla makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "cobyla", "bound");
}
