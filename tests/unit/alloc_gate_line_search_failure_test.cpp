#include "alloc_gate_support.h"
#include "line_search_failure_problems.h"

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace
{

using argmin::solver_status;
using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::window_counts;
using argmin::line_search_failure::vector2;
using argmin::line_search_failure::poisoned_everywhere;
using argmin::line_search_failure::poisoned_off_steepest_descent;

constexpr std::size_t passes = 2;

const vector2 start{3.0, 2.0};

template <typename Solver>
void take_clean_steps(Solver& solver)
{
    solver.reset_clear(start);
    solver.step();
    solver.step();
}

template <typename Solver>
bool retry_succeeds(Solver& solver, poisoned_off_steepest_descent& problem)
{
    problem.anchor = solver.state().x;
    problem.anchor_gradient = solver.state().g;
    problem.poisoned = true;
    const argmin::step_result<double> r = solver.step();
    problem.poisoned = false;
    return r.diagnostics.bfgs_reset_count == std::size_t{1} && !r.policy_status.has_value();
}

template <typename Solver>
bool search_terminates(Solver& solver, poisoned_everywhere& problem)
{
    problem.anchor = solver.state().x;
    problem.poisoned = true;
    const argmin::step_result<double> r = solver.step();
    problem.poisoned = false;
    return r.diagnostics.bfgs_reset_count == std::size_t{1} && r.policy_status == solver_status::line_search_failed;
}

template <typename Policy>
void run_line_search_failure_gate()
{
    poisoned_off_steepest_descent selective;
    poisoned_everywhere everywhere;
    argmin::step_budget_solver retrying{Policy{}, selective, start};
    argmin::step_budget_solver failing{Policy{}, everywhere, start};
    // The warm-up steps size the curvature memory's rows before the window is armed.
    take_clean_steps(retrying);
    take_clean_steps(failing);

    std::size_t retried = 0;
    std::size_t terminated = 0;
    const window_counts counts = run_armed([&] {
        for(std::size_t pass = 0; pass < passes; ++pass)
        {
            if(retry_succeeds(retrying, selective))
                ++retried;
            take_clean_steps(retrying);
            if(search_terminates(failing, everywhere))
                ++terminated;
            take_clean_steps(failing);
        }
    });

    CHECK(retried == passes);
    CHECK(terminated == passes);
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

}

TEST_CASE("alloc_gate_lbfgsb_line_search_failure", "[alloc-gate]")
{
    run_line_search_failure_gate<argmin::lbfgsb_policy<double, 2>>();
}

TEST_CASE("alloc_gate_byrd_lbfgsb_line_search_failure", "[alloc-gate]")
{
    run_line_search_failure_gate<argmin::byrd_lbfgsb_policy<double, 2>>();
}
