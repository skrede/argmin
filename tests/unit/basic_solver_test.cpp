#include "mock_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/result/status.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

// Dummy problem type -- mock_policy ignores it, but basic_solver needs one.
struct quadratic
{
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
};

TEST_CASE("basic_solver step", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    SECTION("single step returns valid step_result")
    {
        auto sr = solver.step();
        CHECK(sr.objective_value < 0.5 * x0.squaredNorm());
        CHECK(sr.gradient_norm > 0.0);
        CHECK(sr.improved);
    }

    SECTION("step modifies state")
    {
        solver.step();
        CHECK(solver.state().x.norm() < x0.norm());
    }
}

TEST_CASE("basic_solver solve converges", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations > 0);
    CHECK(result.gradient_norm < opts.gradient_tolerance);
    CHECK(result.x.norm() < 1e-3);
}

TEST_CASE("basic_solver solve max_iterations", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 10;
    opts.gradient_tolerance = 1e-100;
    opts.objective_tolerance = 1e-100;
    opts.step_tolerance = 1e-100;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.status == solver_status::max_iterations);
    CHECK(result.iterations == 10);
}

TEST_CASE("basic_solver step_n budget", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{10.0, 10.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-100;
    opts.objective_tolerance = 1e-100;
    opts.step_tolerance = 1e-100;
    opts.max_iterations = 1000;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.step_n(3);

    CHECK(result.status == solver_status::budget_exhausted);
    CHECK(result.iterations == 3);
}

TEST_CASE("basic_solver state access", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{5.0, -3.0}};
    solver_options opts;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    CHECK(solver.state().x.isApprox(x0));
    CHECK(solver.state().objective_value == Approx(0.5 * x0.squaredNorm()));
}

TEST_CASE("basic_solver convergence on objective_change", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{0.001, 0.001}};
    solver_options opts;
    opts.gradient_tolerance = 1e-100;
    opts.objective_tolerance = 1e-8;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.status == solver_status::converged);
}
