#include "mock_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/result/status.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

// Dummy problem type -- mock_policy ignores it, but basic_solver needs one.
struct quadratic
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
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
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations > 0);
    CHECK(result.gradient_norm < 1e-4);
    CHECK(result.x.norm() < 1e-3);
}

TEST_CASE("basic_solver solve max_iterations", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options<> opts;
    opts.max_iterations = 10;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::max_iterations);
    CHECK(result.iterations == 10);
}

TEST_CASE("basic_solver step_n budget", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{10.0, 10.0}};
    solver_options<> opts;
    opts.max_iterations = 1000;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.step_n(3, opts);

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
    solver_options<> opts;
    std::get<objective_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-8;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::ftol_reached);
}

TEST_CASE("basic_solver reset preserves convergence ability", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{10.0, 10.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    SECTION("reset to new x0 and solve again")
    {
        auto result1 = solver.solve(opts);
        CHECK(result1.status == solver_status::converged);

        Eigen::VectorXd new_x0{{5.0, 5.0}};
        solver.reset(new_x0);

        CHECK(solver.state().x.isApprox(new_x0));

        auto result2 = solver.solve(opts);
        CHECK(result2.status == solver_status::converged);
    }

    SECTION("reset_clear to new x0 and solve again")
    {
        auto result1 = solver.solve(opts);
        CHECK(result1.status == solver_status::converged);

        Eigen::VectorXd new_x0{{5.0, 5.0}};
        solver.reset_clear(new_x0);

        CHECK(solver.state().x.isApprox(new_x0));

        auto result2 = solver.solve(opts);
        CHECK(result2.status == solver_status::converged);
    }
}
