#include "oracle/state_machine.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"

#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdint>

using namespace argmin;

namespace
{

using rosenbrock2 = rosenbrock<double, 2>;
using lbfgsb2 = lbfgsb_policy<double, 2>;

const Eigen::Vector2d other_start{0.5, -0.5};

time_budget_options<> ample_time()
{
    return {.core = solver_options<>{}, .max_time = std::chrono::hours(1)};
}

template <typename Solver>
Solver at_start(const rosenbrock2& problem, const Eigen::Vector2d& start)
{
    if constexpr(requires { Solver{lbfgsb2{}, problem, start, solver_options<>{}}; })
        return Solver{lbfgsb2{}, problem, start, solver_options<>{}};
    else
        return Solver{lbfgsb2{}, problem, start, ample_time()};
}

template <typename Solver>
void step_until_finished(Solver& solver)
{
    for(std::uint32_t i = 0; i < 1000 && solver.status() == solver_status::running; ++i)
        solver.step();
}

// A zero budget counts what the reset or the construction evaluated, and steps nothing.
template <typename Solver>
std::string reset_count_failure(Solver& used, const rosenbrock2& problem)
{
    Solver fresh = at_start<Solver>(problem, other_start);
    const auto used_result = used.step_n(0);
    const auto fresh_result = fresh.step_n(0);
    if(used_result.iterations != 0 || fresh_result.iterations != 0)
        return "iterations";
    return used_result.function_evaluations == fresh_result.function_evaluations ? std::string{} : std::string{"evaluations"};
}

template <typename Solver>
std::string reset_failure_after_solve(const rosenbrock2& problem)
{
    Solver used = at_start<Solver>(problem, problem.initial_point());
    step_until_finished(used);
    if(used.status() == solver_status::running)
        return "the first run did not end";
    used.reset(other_start);
    if(used.status() != solver_status::running)
        return "status";
    if(std::string field = oracle::reset_failure(oracle::take_snapshot(used), other_start); !field.empty())
        return field;
    if constexpr(requires { used.step_n(0); })
        return reset_count_failure(used, problem);
    return {};
}

}

TEST_CASE("a reset step_budget_solver reads as a fresh one: no iterations, its construction's evaluations, running", "[oracle][state-machine]")
{
    CHECK(reset_failure_after_solve<step_budget_solver<lbfgsb2, 2, rosenbrock2>>(rosenbrock2{}).empty());
}

TEST_CASE("a reset time_budget_solver reads as a fresh one: no iterations, its construction's evaluations, running", "[oracle][state-machine]")
{
    CHECK(reset_failure_after_solve<time_budget_solver<lbfgsb2, 2, rosenbrock2>>(rosenbrock2{}).empty());
}

TEST_CASE("a reset step_and_time_budget_solver reads as a fresh one: no iterations, its construction's evaluations, running", "[oracle][state-machine]")
{
    CHECK(reset_failure_after_solve<step_and_time_budget_solver<lbfgsb2, 2, rosenbrock2>>(rosenbrock2{}).empty());
}

TEST_CASE("a reset stepper reads as a fresh one: running, at its new start, no gradient norm yet", "[oracle][state-machine]")
{
    CHECK(reset_failure_after_solve<stepper<lbfgsb2, 2, rosenbrock2>>(rosenbrock2{}).empty());
}
