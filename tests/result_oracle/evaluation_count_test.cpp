#include "count_visitor.h"

#include "cases/gradient_cases.h"
#include "cases/least_squares_cases.h"

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
