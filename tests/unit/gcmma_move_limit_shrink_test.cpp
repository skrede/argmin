// Tests for the move-limit-shrink GCMMA variant
// (argmin::alternative::gcmma::move_limit_shrink_policy).
//
// This is one of three GCMMA conservativity-globalization variants kept
// in solver/alternative/gcmma/ for empirical comparison. Production
// gcmma_policy will alias the winning variant after benchmarking.
//
// Tests verify "makes substantial progress" rather than "converges to
// optimum" since this variant is known to stall early when the
// conservativity loop exhausts its budget on shrunk trust regions
// (each shrink is wasted dual-solver work). The empirical comparison
// in the published benchmark suite captures the actual convergence
// behavior across all three variants on the full problem set.

#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

// HS024 (n=2, m=3 inequality + bound, f* = -1). Move-limit-shrink
// converges fully on this problem.
TEST_CASE("gcmma move-limit-shrink converges on HS024",
          "[gcmma_move_limit_shrink]")
{
    hs024 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::move_limit_shrink_policy<
            hs024<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(0.05));
}

// HS035 (n=3, m=1 inequality + bound, f* = 1/9). Move-limit-shrink is
// expected to make substantial progress but may stall short of f*. Its
// variables are half-bounded (lower 0, upper +inf), so the faithful move
// bound engages once the moving asymptote inflates and caps the per-step
// travel; combined with this variant's own window shrinkage that makes it
// converge markedly slower than the canonical rho-growth GCMMA (it is a
// non-production research variant). The step budget is sized to let it
// still demonstrate substantial progress under that faithful cap.
TEST_CASE("gcmma move-limit-shrink makes progress on HS035",
          "[gcmma_move_limit_shrink]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    const double f_x0 = problem.value(x0);

    solver_options opts;
    opts.max_iterations = 6000;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::move_limit_shrink_policy<
            hs035<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    // Substantial progress from x0 toward f* = 1/9 ~= 0.111.
    CHECK(result.objective_value < f_x0);
    CHECK(result.objective_value < 0.6);
}

// HS076 (n=4, m=3 inequality + bound, f* = -4.6818). Plain MMA fails
// here from infeasibility drift; move-limit-shrink at least makes
// monotone progress without diverging.
TEST_CASE("gcmma move-limit-shrink makes progress on HS076",
          "[gcmma_move_limit_shrink]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    const double f_x0 = problem.value(x0);

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        alternative::gcmma::move_limit_shrink_policy<
            hs076<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    // Substantial progress from f(x0) ~= -1.25 toward f* = -4.6818.
    CHECK(result.objective_value < f_x0);
    CHECK(result.objective_value < -3.0);
}
