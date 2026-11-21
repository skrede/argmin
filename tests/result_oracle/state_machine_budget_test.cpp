#include "oracle/state_machine.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdint>

using namespace argmin;

namespace
{

using rosenbrock2 = rosenbrock<double, 2>;
using step_solver = step_budget_solver<lbfgsb_policy<double, 2>, 2, rosenbrock2>;
using time_solver = time_budget_solver<lbfgsb_policy<double, 2>, 2, rosenbrock2>;

step_solver step_solver_at_start(const rosenbrock2& problem, std::uint32_t max_iterations)
{
    solver_options<> opts;
    opts.max_iterations = max_iterations;
    return {lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), opts, lbfgsb_policy<double, 2>::options_type{}};
}

time_budget_options<> timed(std::chrono::nanoseconds max_time, std::uint32_t max_iterations)
{
    time_budget_options<> opts;
    opts.core.max_iterations = max_iterations;
    opts.max_time = max_time;
    opts.time_poll_stride = 1;
    return opts;
}

constexpr std::chrono::nanoseconds no_time{0};
constexpr std::chrono::nanoseconds ample_time = std::chrono::hours(1);

template <typename Solver>
std::string sticky_step_failure(Solver& solver)
{
    const oracle::snapshot before = oracle::take_snapshot(solver);
    const step_result<double> r = solver.step();
    return oracle::sticky_failure(before, oracle::take_snapshot(solver), r);
}

template <typename Solver, typename Call>
std::string after_call_failure(Solver& solver, const oracle::snapshot& before, Call call)
{
    const auto after = call(solver);
    return oracle::continuation_failure(before, oracle::take_snapshot(solver, after));
}

}

TEST_CASE("step_budget_solver step() after a solve keeps the solve's status", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    step_solver solver = step_solver_at_start(problem, 1000);
    REQUIRE(oracle::is_claim(solver.solve().status));
    CHECK(sticky_step_failure(solver).empty());
}

TEST_CASE("step_budget_solver step() does not advance from budget_exhausted", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    step_solver solver = step_solver_at_start(problem, 1000);
    REQUIRE(solver.step_n(3).status == solver_status::budget_exhausted);
    CHECK(sticky_step_failure(solver).empty());
}

TEST_CASE("step_budget_solver step_n continues after budget_exhausted", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    step_solver solver = step_solver_at_start(problem, 1000);
    const solve_result<double, 2> first = solver.step_n(3);
    REQUIRE(first.status == solver_status::budget_exhausted);
    CHECK(after_call_failure(solver, oracle::take_snapshot(solver, first), [](step_solver& s) { return s.step_n(3); }).empty());
}

TEST_CASE("step_budget_solver step_n takes no step after max_iterations or a claim", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    step_solver capped = step_solver_at_start(problem, 3);
    const solve_result<double, 2> capped_first = capped.solve();
    REQUIRE(capped_first.status == solver_status::max_iterations);
    CHECK(after_call_failure(capped, oracle::take_snapshot(capped, capped_first), [](step_solver& s) { return s.step_n(5); }).empty());
    step_solver claimed = step_solver_at_start(problem, 1000);
    const solve_result<double, 2> claimed_first = claimed.solve();
    REQUIRE(oracle::is_claim(claimed_first.status));
    CHECK(after_call_failure(claimed, oracle::take_snapshot(claimed, claimed_first), [](step_solver& s) { return s.step_n(5); }).empty());
}

TEST_CASE("time_budget_solver ends at time_limit_reached before a step and resumes with time to spare", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    time_solver solver{lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), timed(no_time, 1000)};
    const auto first = solver.step_n(5, timed(no_time, 1000));
    REQUIRE(first.status == solver_status::time_limit_reached);
    REQUIRE(first.iterations == 0);
    CHECK(sticky_step_failure(solver).empty());
    CHECK(after_call_failure(solver, oracle::take_snapshot(solver, first), [](time_solver& s) { return s.step_n(5, timed(ample_time, 1000)); }).empty());
}

TEST_CASE("time_budget_solver step_n takes no step after max_iterations", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    time_solver solver{lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), timed(ample_time, 3)};
    const auto first = solver.solve(timed(ample_time, 3));
    REQUIRE(first.status == solver_status::max_iterations);
    CHECK(after_call_failure(solver, oracle::take_snapshot(solver, first), [](time_solver& s) { return s.step_n(5, timed(ample_time, 3)); }).empty());
}

// The inner solver resumes after each chunk's budget_exhausted, so the chunk size must not change the outcome.
TEST_CASE("augmented Lagrangian over lbfgsb gives the same result for every inner chunk", "[oracle][state-machine]")
{
    using policy_t = augmented_lagrangian_policy<lbfgsb_policy<double, hs076<>::problem_dimension>, hs076<>::problem_dimension>;
    const hs076 problem{};
    solver_options<> opts;
    opts.max_iterations = 200000;
    auto solve_with_chunk = [&](std::uint32_t chunk)
    {
        policy_t::options_type popts;
        popts.inner_chunk = chunk;
        step_budget_solver solver{policy_t{}, problem, problem.initial_point(), opts, popts};
        return solver.solve();
    };
    const auto single = solve_with_chunk(1);
    const auto chunked = solve_with_chunk(policy_t::options_type{}.inner_chunk);
    REQUIRE(oracle::is_claim(chunked.status));
    CHECK(single.status == chunked.status);
    CHECK(oracle::same_bits(Eigen::VectorXd(single.x), Eigen::VectorXd(chunked.x)));
    CHECK(oracle::same_bits(single.objective_value, chunked.objective_value));
    CHECK(oracle::same_bits(single.constraint_violation, chunked.constraint_violation));
    CHECK(oracle::same_bits(single.gradient_norm, chunked.gradient_norm));
}
