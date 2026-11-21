#include "count_visitor.h"

#include "cases/sqp_cases.h"
#include "cases/gradient_cases.h"
#include "cases/least_squares_cases.h"
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
