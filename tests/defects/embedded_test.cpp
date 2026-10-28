#include "embedded_problems.h"

#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

using namespace argmin;

TEST_CASE("a time budget solve with default options takes at least one iteration", "[known-defect][!shouldfail]")
{
    const defects::coupled_quadratic<4, 0, 0> problem;
    time_budget_solver solver{lbfgsb_policy<double, 4>{}, problem, problem.initial_point(), time_budget_options<>{}};
    CHECK(solver.solve().iterations > 0);
}

TEST_CASE("the default convergence object is smaller than a kilobyte", "[known-defect][!shouldfail]")
{
    constexpr std::size_t size = sizeof(default_convergence<double>);
    CHECK(size < 1024);
}
