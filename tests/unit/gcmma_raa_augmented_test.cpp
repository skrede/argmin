// Tests for the strict Svanberg 2002 raa-augmented GCMMA variant
// (nablapp::alternative::gcmma::raa_augmented_policy).

#include "nablapp/solver/alternative/gcmma/raa_augmented_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("gcmma raa-augmented converges on HS024", "[gcmma_raa]")
{
    hs024 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs024<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(-1.0).margin(0.05));
}

TEST_CASE("gcmma raa-augmented converges on HS035", "[gcmma_raa]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs035<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.05));
}

TEST_CASE("gcmma raa-augmented converges on HS076", "[gcmma_raa]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs076<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(-4.6818).margin(0.05));
}
