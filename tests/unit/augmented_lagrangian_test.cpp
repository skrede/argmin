#include "nablapp/solver/augmented_lagrangian_policy.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("augmented lagrangian converges on HS076", "[augmented_lagrangian]")
{
    hs076 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 60;
    opts.gradient_tolerance = 1e-6;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver{
        problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value == Approx(-4.6818).margin(1e-2));
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("augmented lagrangian converges on HS035", "[augmented_lagrangian]")
{
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 50;
    opts.gradient_tolerance = 1e-6;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver{
        problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(1e-3));
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("augmented lagrangian on HS071 (equality + inequality)",
          "[augmented_lagrangian]")
{
    hs071 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 80;
    opts.gradient_tolerance = 1e-4;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver{
        problem, x0, opts};
    auto result = solver.solve();

    // Relaxed tolerance: AugLag on mixed eq/ineq is harder
    CHECK(result.objective_value == Approx(17.0140173).margin(1e-1));
    CHECK(solver.constraint_violation() < 1e-2);
}

TEST_CASE("augmented lagrangian with BOBYQA inner solver",
          "[augmented_lagrangian]")
{
    // Use HS035 (simpler, inequality-only) with BOBYQA inner solver
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 80;
    opts.gradient_tolerance = 1e-4;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    basic_solver<augmented_lagrangian_policy<bobyqa_policy>> solver{
        problem, x0, opts};
    auto result = solver.solve();

    // BOBYQA is derivative-free; relaxed tolerance
    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.5));
}

TEST_CASE("augmented lagrangian step and solve consistency",
          "[augmented_lagrangian]")
{
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 30;
    opts.gradient_tolerance = 1e-5;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    SECTION("step returns finite values")
    {
        basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver{
            problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
        }
    }

    SECTION("solve gives similar result to manual stepping")
    {
        basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver1{
            problem, x0, opts};
        auto result1 = solver1.solve();

        basic_solver<augmented_lagrangian_policy<lbfgsb_policy>> solver2{
            problem, x0, opts};
        step_result<double> last{};
        for(int i = 0; i < opts.max_iterations; ++i)
        {
            last = solver2.step();
            if(last.gradient_norm < opts.gradient_tolerance)
                break;
        }

        // Both should reach a similar objective value
        CHECK(last.objective_value == Approx(result1.objective_value).margin(1e-2));
    }
}
