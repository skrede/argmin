#include "membership_visitor.h"

#include "cases/sqp_cases.h"
#include "cases/gradient_cases.h"
#include "cases/stochastic_cases.h"
#include "cases/least_squares_cases.h"
#include "cases/derivative_free_cases.h"
#include "cases/moving_asymptote_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

// The QP solvers have no step(), so they have no trajectory to be a member of.

TEST_CASE("trajectory membership holds for every gradient-based bound solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_gradient_cases(membership);
}

TEST_CASE("trajectory membership holds for every least-squares solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_least_squares_cases(membership);
}

TEST_CASE("trajectory membership holds for every line-search SQP solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_sqp_cases(membership);
}

TEST_CASE("trajectory membership holds for every trust-region SQP solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_trust_region_sqp_cases(membership);
}

TEST_CASE("trajectory membership holds for every moving-asymptote solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_moving_asymptote_cases(membership);
}

TEST_CASE("trajectory membership holds for every derivative-free solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_derivative_free_cases(membership);
}

TEST_CASE("trajectory membership holds for every stochastic solver", "[oracle][membership]")
{
    cases::membership_visitor membership;
    cases::visit_stochastic_cases(membership);
}
