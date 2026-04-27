// Tests for the rho/wval GCMMA variant
// (nablapp::alternative::gcmma::rho_wval_policy).
//
// NLopt-mma.c-style globalization: rho-augmented MMA approximation
// with wval-based rho-growth on non-conservative trials. Numerical
// per-component primal (Newton) replaces the closed-form analytic
// minimum used by plain MMA / move-limit-shrink.
//
// Expected behavior: should reach optimum on HS024/HS035/HS076 at
// the cost of higher per-iter wall time (Newton per j per dual
// evaluation) compared to move-limit-shrink. The empirical
// comparison in the published benchmark suite captures the
// quantitative trade-off.

#include "nablapp/solver/alternative/gcmma/rho_wval_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("gcmma rho-wval converges on HS024", "[gcmma_rho_wval]")
{
    hs024 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs024<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(0.05));
}

TEST_CASE("gcmma rho-wval converges on HS035", "[gcmma_rho_wval]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs035<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.05));
}

TEST_CASE("gcmma rho-wval converges on HS076", "[gcmma_rho_wval]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs076<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-4.6818).margin(0.05));
}
