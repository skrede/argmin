// Robustness: the cooperative-abort guard.
//
// A real-time caller must be able to stop a driver deterministically. abort()
// sets an atomic flag the solve loop samples at the top of every iteration
// (detail/solve_loop.h): once set, the loop stops immediately with a terminal
// solver_status::aborted before running another policy step. reset()/reset_clear()
// clear the flag so a driver is reusable after an abort.
//
// These are behavioral tests -- each drives the abort path and asserts the
// observable outcome (status, iteration count, reusability) -- across a policy
// from each search-strategy archetype, since the guard lives in the shared loop
// every driver composes.
//
// Reference: N&W 2e Section 3.1 (iterative drivers).

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/result/status.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace argmin;

namespace
{

// abort() before the first step stops the loop with `aborted` and no iterations.
template <typename Policy>
void abort_before_solve_yields_aborted()
{
    const hs001<double> problem;
    const Eigen::Vector<double, 2> x0 = problem.initial_point();

    step_budget_solver solver{Policy{}, problem, x0};
    solver.abort();
    const auto result = solver.solve();

    CHECK(result.status == solver_status::aborted);
    CHECK(result.iterations == 0);
    CHECK(std::isfinite(result.objective_value));
}

// reset_clear() clears the abort flag: the same driver runs normally afterward.
template <typename Policy>
void reset_clear_reenables_after_abort()
{
    const hs001<double> problem;
    const Eigen::Vector<double, 2> x0 = problem.initial_point();

    step_budget_solver solver{Policy{}, problem, x0};
    solver.abort();
    REQUIRE(solver.solve().status == solver_status::aborted);

    solver.reset_clear(x0);
    const auto result = solver.step_n(20);
    CHECK(result.status != solver_status::aborted);
    CHECK(std::isfinite(result.objective_value));
}

}

TEST_CASE("abort before solve yields aborted -- gradient policy", "[robustness][abort]")
{
    abort_before_solve_yields_aborted<lbfgsb_policy<>>();
}

TEST_CASE("abort before solve yields aborted -- derivative-free policy",
          "[robustness][abort]")
{
    abort_before_solve_yields_aborted<bobyqa_policy<>>();
}

TEST_CASE("abort before solve yields aborted -- stochastic policy",
          "[robustness][abort]")
{
    abort_before_solve_yields_aborted<cmaes_policy<>>();
}

TEST_CASE("reset_clear re-enables a driver after abort -- gradient policy",
          "[robustness][abort]")
{
    reset_clear_reenables_after_abort<lbfgsb_policy<>>();
}

TEST_CASE("reset_clear re-enables a driver after abort -- derivative-free policy",
          "[robustness][abort]")
{
    reset_clear_reenables_after_abort<bobyqa_policy<>>();
}
