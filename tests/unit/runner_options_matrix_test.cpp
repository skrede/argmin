// Options-matrix coverage for the budget-driver split.
//
// The runners compose two independent option axes at construction:
//   axis 1 -- the per-runner budget options (a step budget for the step driver,
//             a wall-clock deadline for the time driver, and both for the
//             combined driver);
//   axis 2 -- the per-policy options_type carried by the wrapped policy.
//
// This exercises the full axis-1 x axis-2 grid over a representative policy from
// each search-strategy archetype -- so the structurally distinct composition
// paths through the runner are all instantiated and driven, not just one:
//   - lbfgsb_policy   : gradient-based (quasi-Newton) descent;
//   - bobyqa_policy   : derivative-free trust-region;
//   - cmaes_policy    : stochastic (evolution strategy).
// Each is a distinct options_type flowing through the same three runners.
//
// The assertions are deliberately about the composition contract, not solver
// accuracy: every composed runner constructs, honors its budget, and reports a
// finite objective. Accuracy is pinned in each policy's own test.
//
// Reference: N&W 2e Section 3.1 (iterative drivers); K&W 2e Section 4.4.

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"
#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/result/status.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>

using namespace argmin;

namespace
{

constexpr int k_step_budget = 40;

// Drive the three budget runners over one policy archetype and check the
// composition contract on each: budget honored, finite objective reported.
template <typename Policy>
void exercise_runner_matrix()
{
    // A bound-constrained, differentiable 2-D problem -- the common surface all
    // three archetype policies (gradient / derivative-free / stochastic) accept.
    const hs001<double> problem;
    const Eigen::Vector<double, 2> x0 = problem.initial_point();

    // axis 1 = step budget, composed with the policy's options_type at ctor.
    {
        step_budget_solver runner{Policy{}, problem, x0};
        const auto result = runner.step_n(k_step_budget);
        CHECK(result.iterations <= k_step_budget);
        CHECK(std::isfinite(result.objective_value));
    }

    // axis 1 = wall-clock deadline (effectively infinite so the iteration cap
    // governs), composed with the policy's options_type at ctor.
    {
        time_budget_options<> opts;
        opts.max_time = std::chrono::hours(1);
        opts.core.max_iterations = k_step_budget;
        time_budget_solver runner{Policy{}, problem, x0, opts};
        const auto result = runner.solve(opts);
        CHECK(result.iterations <= k_step_budget);
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.status != solver_status::running);
    }

    // axis 1 = combined step + time budget, composed with the options_type.
    {
        time_budget_options<> opts;
        opts.max_time = std::chrono::hours(1);
        opts.core.max_iterations = k_step_budget;
        step_and_time_budget_solver runner{Policy{}, problem, x0, opts};
        const auto result = runner.step_n(k_step_budget, opts);
        CHECK(result.iterations <= k_step_budget);
        CHECK(std::isfinite(result.objective_value));
    }
}

}

TEST_CASE("runner options matrix: gradient-based policy across all budget drivers",
          "[solver][runner]")
{
    exercise_runner_matrix<lbfgsb_policy<>>();
}

TEST_CASE("runner options matrix: derivative-free policy across all budget drivers",
          "[solver][runner]")
{
    exercise_runner_matrix<bobyqa_policy<>>();
}

TEST_CASE("runner options matrix: stochastic policy across all budget drivers",
          "[solver][runner]")
{
    exercise_runner_matrix<cmaes_policy<>>();
}
