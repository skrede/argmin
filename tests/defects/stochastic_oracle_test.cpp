#include "cases/case_verdict.h"
#include "cases/stochastic_cases.h"

#include "oracle/result_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string_view>

using namespace argmin;

namespace
{

const auto visit_family = [](auto& visitor) { cases::visit_stochastic_cases(visitor); };

void check_inside_box(std::string_view key, std::string_view problem_class)
{
    const std::optional<oracle::verdict<double>> v = cases::judge_case(visit_family, key, problem_class);
    INFO(key << ' ' << problem_class);
    REQUIRE(v.has_value());
    CHECK(v->legs.bound_violation == 0.0);
    CHECK(v->failing != "finite");
}

}

TEST_CASE("cmaes stays inside the hs005 box under every boundary handling", "[stochastic]")
{
    check_inside_box("cmaes_pwq_reparameterization", "bound");
    check_inside_box("cmaes_no_repair_adaptive_penalty", "bound");
}

TEST_CASE("cmaes pwq_reparameterization makes an optimality claim the oracle accepts on hs005", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "cmaes_pwq_reparameterization", "bound");
}

TEST_CASE("cmaes no_repair_adaptive_penalty makes an optimality claim the oracle accepts on hs005", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "cmaes_no_repair_adaptive_penalty", "bound");
}

TEST_CASE("isres stays inside the box on hs076", "[stochastic]")
{
    check_inside_box("isres_nlopt_faithful", "inequality");
    check_inside_box("isres_original_argmin", "inequality");
}

TEST_CASE("isres nlopt_faithful makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_nlopt_faithful", "bound");
}

TEST_CASE("isres nlopt_faithful makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_nlopt_faithful", "inequality");
}

TEST_CASE("isres nlopt_faithful makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_nlopt_faithful", "mixed");
}

TEST_CASE("isres original_argmin makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_original_argmin", "bound");
}

TEST_CASE("isres original_argmin makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_original_argmin", "inequality");
}

TEST_CASE("isres original_argmin makes an optimality claim the oracle accepts on hs028", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_original_argmin", "equality");
}

TEST_CASE("isres original_argmin makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_original_argmin", "mixed");
}

TEST_CASE("isres runarsson_yao_paper makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_runarsson_yao_paper", "bound");
}

TEST_CASE("isres runarsson_yao_paper makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_runarsson_yao_paper", "inequality");
}

TEST_CASE("isres runarsson_yao_paper makes an optimality claim the oracle accepts on hs028", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_runarsson_yao_paper", "equality");
}

TEST_CASE("isres runarsson_yao_paper makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    cases::check_case_claim(visit_family, "isres_runarsson_yao_paper", "mixed");
}
