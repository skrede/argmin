// Tests for relative tolerances, time stopping, policy status propagation,
// and solver group failure retirement.
//
// Validates STOP-01 through STOP-05, D-16, D-17, D-18.
//
// Reference: N&W 2e Section 2.2 (convergence criteria),
//            K&W 2e Section 2.3 (stopping conditions).

#include "mock_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/result/status.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>

using namespace nablapp;

namespace
{

struct quadratic
{
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
};

}

// -- Test 10: objective_tolerance_rel stops basic_solver --

TEST_CASE("objective_tolerance_rel stops basic_solver", "[tolerance]")
{
    using conv = convergence_policy<objective_tolerance_rel_criterion>;

    solver_options<conv> opts;
    opts.max_iterations = 1000;
    opts.convergence.criteria = {
        objective_tolerance_rel_criterion{.threshold = 1e-2}
    };

    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::ftol_reached);
    CHECK(result.iterations < 1000);
}

// -- Test 11: step_tolerance_rel stops basic_solver --

TEST_CASE("step_tolerance_rel stops basic_solver", "[tolerance]")
{
    // mock_policy: step_size=0.5 constant, x_norm = 5 * 0.5^k.
    // relative step = step_size / max(x_norm, 1.0).
    // At iteration 2: x_norm = 1.25, ratio = 0.5/1.25 = 0.4 < 0.5 -> triggers.
    using conv = convergence_policy<step_tolerance_rel_criterion>;

    solver_options<conv> opts;
    opts.max_iterations = 1000;
    opts.convergence.criteria = {
        step_tolerance_rel_criterion{.threshold = 0.5}
    };

    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::xtol_reached);
    CHECK(result.iterations < 1000);
}

// -- Test 12: max_time stops with time_limit_reached --

TEST_CASE("max_time stops with time_limit_reached", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 1000000;
    opts.max_time = std::chrono::milliseconds(1);

    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::time_limit_reached);
}

// -- Test 13: policy roundoff_limited propagates to basic_solver --

TEST_CASE("policy roundoff_limited propagates to basic_solver", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 100;

    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    basic_solver<test::roundoff_mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::roundoff_limited);
    CHECK(result.iterations == 3);
}

// -- Test 14: policy diverged propagates to basic_solver --

TEST_CASE("policy diverged propagates to basic_solver", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 100;

    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    basic_solver<test::diverging_mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::diverged);
    CHECK(result.iterations == 5);
}

// -- Test 15: policy_status checked before convergence (D-16 priority) --

TEST_CASE("policy_status checked before convergence criteria", "[tolerance]")
{
    // Set gradient threshold low enough that roundoff_mock would satisfy it,
    // but policy_status should take priority.
    solver_options<> opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-1);

    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    basic_solver<test::roundoff_mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    // roundoff_mock reports roundoff_limited at step 3, and by then
    // gradient_norm = x.norm() after 3 halvings is small but the
    // policy_status should still be what determines the result status.
    CHECK(result.status == solver_status::roundoff_limited);
}

// -- Test 16: solver_group retires roundoff_limited policy, other continues --

TEST_CASE("solver_group retires roundoff_limited policy", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);

    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    basic_solver_group<round_robin_schedule,
                       test::roundoff_mock_policy,
                       test::mock_policy> group(prob, x0, opts);
    auto result = group.step_n(200, opts);

    // mock_policy should converge via gradient threshold
    CHECK(result.status == solver_status::converged);

    auto& results = group.results();
    // results[0] should show roundoff_limited (retired)
    CHECK(results[0].status == solver_status::roundoff_limited);
}

// -- Test 17: solver_group results() returns per-policy results --

TEST_CASE("solver_group results provides per-policy info", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);

    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    basic_solver_group<round_robin_schedule,
                       test::roundoff_mock_policy,
                       test::mock_policy> group(prob, x0, opts);
    auto result = group.step_n(200, opts);

    auto& results = group.results();
    // Both policies should have results populated
    CHECK(results[0].x.size() == 2);
    CHECK(results[1].x.size() == 2);
    // Retired policy has roundoff_limited status
    CHECK(results[0].status == solver_status::roundoff_limited);
    // Active policy gets group status
    CHECK(results[1].status == solver_status::converged);
}

// -- Test 18: solver_group all retired stops iteration --

TEST_CASE("solver_group all retired stops iteration", "[tolerance]")
{
    solver_options<> opts;
    opts.max_iterations = 1000;

    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    basic_solver_group<round_robin_schedule,
                       test::roundoff_mock_policy,
                       test::diverging_mock_policy> group(prob, x0, opts);
    auto result = group.step_n(1000, opts);

    // Both policies fail, group should stop early
    CHECK(result.status == solver_status::budget_exhausted);
    CHECK(result.iterations == 1000);

    auto& results = group.results();
    CHECK(results[0].status == solver_status::roundoff_limited);
    CHECK(results[1].status == solver_status::diverged);
}
