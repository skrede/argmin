#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"

#include "argmin/result/status.h"

#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>

using namespace argmin;

namespace
{

using rosenbrock2 = rosenbrock<double, 2>;
using lbfgsb2 = lbfgsb_policy<double, 2>;
using lbfgsb2_solver = step_budget_solver<lbfgsb2, 2, rosenbrock2>;

lbfgsb2_solver with_policy_window(const rosenbrock2& problem, std::uint16_t window)
{
    return {lbfgsb2{}, problem, problem.initial_point(), solver_options<>{}, lbfgsb2::options_type{.stall_window = window}};
}

solver_options<> caller_window(std::uint16_t window)
{
    solver_options<> opts;
    opts.set_stall_threshold(1e-12);
    opts.set_stall_window(window);
    return opts;
}

template <typename Solver>
void check_timed_rejection(Solver solver)
{
    const time_budget_options<> opts{.core = caller_window(513), .max_time = std::chrono::hours(1)};
    const auto started = solver.step_n(2);
    REQUIRE(started.status == solver_status::budget_exhausted);
    const auto solved = solver.solve(opts);
    const auto stepped = solver.step_n(5, opts);
    CHECK(solved.status == solver_status::invalid_problem);
    CHECK(stepped.status == solver_status::invalid_problem);
    CHECK(stepped.iterations == started.iterations);
    CHECK(stepped.function_evaluations == started.function_evaluations);
    CHECK(solver.status() == solver_status::budget_exhausted);
}

}

TEST_CASE("a stall window above the buffer capacity latches invalid_problem at construction", "[release-safety][stall-capacity]")
{
    const rosenbrock2 problem;
    lbfgsb2_solver solver = with_policy_window(problem, 513);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    const auto first = solver.solve();
    const auto second = solver.step_n(5);
    CHECK(first.status == solver_status::invalid_problem);
    CHECK(second.status == solver_status::invalid_problem);
    CHECK(second.iterations == 0u);
    CHECK(second.function_evaluations == first.function_evaluations);
    CHECK(solver.state().x == problem.initial_point());
}

TEST_CASE("a stall window at the buffer capacity runs", "[release-safety][stall-capacity]")
{
    const rosenbrock2 problem;
    lbfgsb2_solver solver = with_policy_window(problem, 512);
    CHECK(solver.status() == solver_status::running);
    const auto result = solver.solve(caller_window(512));
    CHECK(result.status != solver_status::invalid_problem);
    CHECK(result.iterations > 0u);
}

TEST_CASE("a reset keeps an over-capacity stall window rejected until the stored window fits", "[release-safety][stall-capacity]")
{
    const rosenbrock2 problem;
    lbfgsb2_solver solver = with_policy_window(problem, 513);
    CHECK(solver.reset(problem.initial_point()) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(problem.initial_point()) == solver_status::invalid_problem);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.solve().iterations == 0u);

    solver = with_policy_window(problem, 50);
    CHECK(solver.reset(problem.initial_point()) == solver_status::running);
    CHECK(solver.reset_clear(problem.initial_point()) == solver_status::running);
    CHECK(solver.solve().status != solver_status::invalid_problem);
}

TEST_CASE("an explicit solve with a stall window above the buffer capacity returns invalid_problem without stepping", "[release-safety][stall-capacity]")
{
    const rosenbrock2 problem;
    lbfgsb2_solver solver = with_policy_window(problem, 50);
    const solver_options<> opts = caller_window(513);
    const auto started = solver.step_n(3);
    REQUIRE(started.status == solver_status::budget_exhausted);
    const auto solved = solver.solve(opts);
    const auto stepped = solver.step_n(5, opts);
    CHECK(solved.status == solver_status::invalid_problem);
    CHECK(stepped.status == solver_status::invalid_problem);
    CHECK(stepped.iterations == started.iterations);
    CHECK(stepped.function_evaluations == started.function_evaluations);
    CHECK(solver.status() == solver_status::budget_exhausted);
    CHECK(solver.step_n(5, caller_window(50)).iterations > started.iterations);
}

TEST_CASE("an explicit time-budget solve with a stall window above the buffer capacity returns invalid_problem without stepping", "[release-safety][stall-capacity]")
{
    const rosenbrock2 problem;
    const time_budget_options<> opts{.max_time = std::chrono::hours(1)};
    check_timed_rejection(time_budget_solver<lbfgsb2, 2, rosenbrock2>{lbfgsb2{}, problem, problem.initial_point(), opts, lbfgsb2::options_type{}});
    check_timed_rejection(step_and_time_budget_solver<lbfgsb2, 2, rosenbrock2>{lbfgsb2{}, problem, problem.initial_point(), opts, lbfgsb2::options_type{}});
}
