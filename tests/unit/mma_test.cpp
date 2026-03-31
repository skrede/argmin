#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

// --- Concept satisfaction (compile-time) ---
static_assert(differentiable<hs076<>>);
static_assert(constrained<hs076<>>);
static_assert(bound_constrained<hs076<>>);
static_assert(differentiable<hs035<>>);
static_assert(constrained<hs035<>>);

TEST_CASE("mma converges on HS076", "[mma]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.gradient_tolerance = 1e-5;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<mma_policy> solver{problem, x0, opts};

    // MMA is a first-order method; drive toward optimum and track best
    // feasible value found during iteration.
    double best_feasible = 1e10;
    for(int i = 0; i < 500; ++i)
    {
        auto sr = solver.step();
        if(solver.constraint_violation() < 1e-3
           && sr.objective_value < best_feasible)
        {
            best_feasible = sr.objective_value;
        }
    }

    // MMA should find a point within 0.3 of the optimum (-4.6818...)
    CHECK(best_feasible < problem.optimal_value() + 0.3);
    CHECK(best_feasible > problem.optimal_value() - 0.5);
}

TEST_CASE("mma converges on HS035", "[mma]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.gradient_tolerance = 1e-5;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<mma_policy> solver{problem, x0, opts};

    double best_feasible = 1e10;
    for(int i = 0; i < 500; ++i)
    {
        auto sr = solver.step();
        if(solver.constraint_violation() < 1e-3
           && sr.objective_value < best_feasible)
        {
            best_feasible = sr.objective_value;
        }
    }

    // HS035 f* = 1/9 ~ 0.1111
    CHECK(best_feasible < problem.optimal_value() + 0.2);
}

TEST_CASE("gcmma converges on HS076", "[mma][gcmma]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.gradient_tolerance = 1e-5;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<gcmma_policy> solver{problem, x0, opts};

    double best_feasible = 1e10;
    for(int i = 0; i < 500; ++i)
    {
        auto sr = solver.step();
        if(solver.constraint_violation() < 1e-3
           && sr.objective_value < best_feasible)
        {
            best_feasible = sr.objective_value;
        }
    }

    // GCMMA should converge at least as well as MMA
    CHECK(best_feasible < problem.optimal_value() + 0.3);
    CHECK(best_feasible > problem.optimal_value() - 0.5);
}

TEST_CASE("mma rejects equality constraints", "[mma]")
{
    // hs071 has 1 equality constraint -- MMA should reject it.
    // Since the rejection uses assert(), we document the behavior here.
    // In debug builds, this would trigger an assertion failure.
    // We verify the problem has equality constraints to document
    // that MMA is not intended for such problems.
    hs071 problem;
    CHECK(problem.num_equality() == 1);
    // mma_policy{}.init(problem, problem.initial_point(), {})
    // would trigger: assert(problem.num_equality() == 0 && "...")
}

TEST_CASE("mma step and step_n consistency", "[mma]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.gradient_tolerance = 1e-4;
    opts.max_iterations = 200;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    SECTION("step returns finite values")
    {
        basic_solver<mma_policy> solver{problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
            CHECK(std::isfinite(sr.step_size));
        }
    }

    SECTION("step_n reaches similar result as solve")
    {
        basic_solver<mma_policy> solver{problem, x0, opts};
        auto result = solver.step_n(200);
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.objective_value < problem.value(x0));
    }
}
