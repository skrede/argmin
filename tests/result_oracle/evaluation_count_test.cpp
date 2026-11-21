#include "count_visitor.h"

#include "cases/sqp_cases.h"
#include "cases/gradient_cases.h"
#include "cases/stochastic_cases.h"
#include "cases/least_squares_cases.h"
#include "cases/derivative_free_cases.h"
#include "cases/moving_asymptote_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string_view>

using namespace argmin;

TEST_CASE("lbfgsb reports every objective evaluation it makes", "[oracle][evaluations]")
{
    constexpr std::array<std::string_view, 1> keys{"lbfgsb"};
    cases::count_visitor count{keys, 0};
    cases::visit_gradient_cases(count);
    CHECK(count.judged == 2);
}

TEST_CASE("byrd_lbfgsb reports every objective evaluation it makes", "[oracle][evaluations]")
{
    constexpr std::array<std::string_view, 1> keys{"byrd_lbfgsb"};
    cases::count_visitor count{keys, 0};
    cases::visit_gradient_cases(count);
    CHECK(count.judged == 2);
}

TEST_CASE("the augmented Lagrangian reports every objective evaluation it and its inner solver make", "[oracle][evaluations]")
{
    constexpr std::array<std::string_view, 1> keys{"augmented_lagrangian_lbfgsb"};
    cases::count_visitor count{keys, 0};
    cases::visit_gradient_cases(count);
    CHECK(count.judged == 3);
}

TEST_CASE("every least-squares solver reports every residual and objective evaluation it makes", "[oracle][evaluations]")
{
    cases::count_visitor count{cases::least_squares_keys, 0};
    cases::visit_least_squares_cases(count);
    CHECK(count.judged == 8);
}

TEST_CASE("every line-search SQP solver reports every objective evaluation it makes", "[oracle][evaluations]")
{
    cases::count_visitor count{cases::sqp_keys, 0};
    cases::visit_sqp_cases(count);
    CHECK(count.judged == 18);
}

TEST_CASE("every trust-region SQP solver reports every objective evaluation it makes", "[oracle][evaluations]")
{
    cases::count_visitor count{cases::trust_region_sqp_keys, 0};
    cases::visit_trust_region_sqp_cases(count);
    CHECK(count.judged == 16);
}

TEST_CASE("every moving-asymptote solver reports every objective evaluation it makes", "[oracle][evaluations]")
{
    cases::count_visitor count{cases::moving_asymptote_keys, 0};
    cases::visit_moving_asymptote_cases(count);
    CHECK(count.judged == 10);
}

TEST_CASE("every derivative-free solver reports every objective evaluation it makes", "[oracle][evaluations]")
{
    cases::count_visitor count{cases::derivative_free_keys, 0};
    cases::visit_derivative_free_cases(count);
    CHECK(count.judged == 5);
}

TEST_CASE("every CMA-ES and ISRES solver reports every fitness evaluation of every generation", "[oracle][evaluations]")
{
    constexpr std::array<std::string_view, 6> keys{
        "cmaes_repair_l2_penalty", "cmaes_pwq_reparameterization", "cmaes_no_repair_adaptive_penalty", "isres_nlopt_faithful", "isres_original_argmin", "isres_runarsson_yao_paper"};
    cases::count_visitor count{keys, 0};
    cases::visit_stochastic_cases(count);
    CHECK(count.judged == 18);
}

TEST_CASE("the multistart and restarting solvers report every evaluation of every start", "[oracle][evaluations]")
{
    constexpr std::array<std::string_view, 2> keys{"multistart_lbfgsb", "restarting_cmaes"};
    cases::count_visitor count{keys, 0};
    cases::visit_gradient_cases(count);
    cases::visit_stochastic_cases(count);
    CHECK(count.judged == 4);
}
