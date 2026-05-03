#include "mock_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/convergence.h"
#include "argmin/result/status.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

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

TEST_CASE("basic_solver solve_uses_stored_convergence", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations < 500);
    CHECK(result.gradient_norm < 1e-4);
}

TEST_CASE("basic_solver step_n_no_opts_uses_stored_convergence", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.step_n(1000);

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations < 500);
    CHECK(result.gradient_norm < 1e-4);
}

TEST_CASE("basic_solver step_n(budget, opts) back-copies last_check_results",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    // Before any step: all entries are nullopt.
    const auto& before = solver.convergence().last_check_results();
    CHECK(!before[0].has_value());
    CHECK(!before[1].has_value());
    CHECK(!before[2].has_value());
    CHECK(!before[3].has_value());

    // Explicit-opts step_n: this is the cartan-side usage pattern that
    // previously did not back-copy into stored_convergence_.
    auto result = solver.step_n(1000, opts);
    REQUIRE(result.status == solver_status::converged);

    // After: solver.convergence().last_check_results() mirrors what was
    // written into opts.convergence through the const-ref check() calls.
    const auto& after = solver.convergence().last_check_results();
    REQUIRE(after[0].has_value());
    CHECK(*after[0] == solver_status::converged);

    // Same array on both sides -- this is what cartan wanted for the
    // accessor symmetry between the no-opts and explicit-opts paths.
    CHECK(after[0] == opts.convergence.last_check_results()[0]);
}

TEST_CASE("basic_solver step_n(budget, opts) back-copy is gated on Convergence type",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<slsqp_compatible_convergence> alias_opts;
    std::get<0>(alias_opts.convergence.criteria).threshold = 1e-10;
    std::get<0>(alias_opts.convergence.criteria).stationarity_threshold = 1.0;
    std::get<1>(alias_opts.convergence.criteria).threshold = 1e-10;

    basic_solver<test::non_converging_policy> solver{prob, x0, solver_options<>{}};

    // step_n with non-default Convergence compiles and runs through the
    // same code path, but the if-constexpr guard skips the back-copy into
    // stored_convergence_ because default_convergence has four criteria
    // and slsqp_compatible_convergence has three. Consumers read from
    // their own alias_opts.convergence in this case.
    auto result = solver.step_n(5, alias_opts);
    CHECK(result.iterations == 5);

    // stored_convergence_ is untouched -- still all nullopt.
    const auto& stored = solver.convergence().last_check_results();
    CHECK(!stored[0].has_value());
    CHECK(!stored[1].has_value());
    CHECK(!stored[2].has_value());
    CHECK(!stored[3].has_value());
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
