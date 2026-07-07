// CMA-ES population-cap rejection instruments.
//
// The three boundary-handling CMA-ES variants pre-allocate their lambda-
// sized per-step buffers as Eigen matrices with a static maximum-rows
// bound (MaxPop, derived from the policy's MaxPopulation template
// parameter). An auto-computed or user-requested population above that cap
// would resize those buffers past their compile-time bound -- undefined
// behavior. Because the library is exception-free, the over-cap condition
// is surfaced as a terminal solver_status::invalid_problem: init()/reset()
// flag the state and skip the unsafe sizing, and step() returns the status
// on its first call before touching any buffer.
//
// These tests pin that contract for all three variants on both entry
// paths (init() and reset()), and pin that the valid, under-cap path is
// unchanged (converges on a convex sphere as before).

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

// Bounded convex sphere: objective && bound_constrained, no gradient.
struct bounded_sphere
{
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return static_cast<int>(lb.size()); }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// A small explicit population cap so an over-cap lambda is cheap to
// construct: MaxPop resolves to this value, and a requested lambda of
// OverCapLambda > TestMaxPop trips the guard.
constexpr int TestMaxPop = 8;
constexpr std::uint32_t OverCapLambda = 16u;  // > TestMaxPop
constexpr std::uint32_t UnderCapLambda = 6u;  // < TestMaxPop

using pwq_capped =
    alternative::cmaes::pwq_reparameterization_policy<dynamic_dimension, TestMaxPop>;
using l2_capped =
    alternative::cmaes::repair_l2_penalty_policy<dynamic_dimension, TestMaxPop>;
using no_repair_capped =
    alternative::cmaes::no_repair_adaptive_penalty_policy<dynamic_dimension, TestMaxPop>;

bounded_sphere make_problem()
{
    return bounded_sphere{
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0)};
}

}

TEMPLATE_TEST_CASE("cmaes: over-cap population rejected as invalid_problem",
                   "[cmaes][invalid_problem]",
                   pwq_capped, l2_capped, no_repair_capped)
{
    using Policy = TestType;

    bounded_sphere problem = make_problem();
    Eigen::VectorXd x0{{2.0, 2.0}};

    solver_options opts;
    opts.max_iterations = 100;

    SECTION("init path: first step returns invalid_problem, no crash")
    {
        Policy policy;
        policy.options.lambda = OverCapLambda;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 7u;

        // Construction / init() must not crash on the over-cap request.
        step_budget_solver solver{policy, problem, x0, opts};

        auto first = solver.step();
        REQUIRE(first.policy_status.has_value());
        CHECK(*first.policy_status == solver_status::invalid_problem);
        CHECK(first.is_null_step);
    }

    SECTION("init path: solve terminates with invalid_problem, does not spin")
    {
        Policy policy;
        policy.options.lambda = OverCapLambda;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 7u;

        step_budget_solver solver{policy, problem, x0, opts};
        auto result = solver.solve();
        CHECK(result.status == solver_status::invalid_problem);
    }

    SECTION("reset path: reconfiguring a valid solver over the cap is rejected")
    {
        Policy policy;
        policy.options.lambda = UnderCapLambda;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 11u;

        // A valid under-cap init: no flag, and a real generation steps.
        auto state = policy.init(problem, x0, opts);
        REQUIRE_FALSE(state.invalid_problem);
        auto valid = policy.step(state);
        const bool valid_is_rejected =
            valid.policy_status.has_value()
            && *valid.policy_status == solver_status::invalid_problem;
        CHECK_FALSE(valid_is_rejected);

        // Reconfigure the population over the cap and reset: the reset()
        // site must flag the state so the next step() rejects.
        policy.options.lambda = OverCapLambda;
        policy.reset(state, x0);
        REQUIRE(state.invalid_problem);

        auto rejected = policy.step(state);
        REQUIRE(rejected.policy_status.has_value());
        CHECK(*rejected.policy_status == solver_status::invalid_problem);
        CHECK(rejected.is_null_step);
    }

    SECTION("valid path unchanged: under-cap lambda converges on the sphere")
    {
        Policy policy;
        policy.options.lambda = UnderCapLambda;
        policy.options.initial_sigma = 1.0;
        policy.options.seed = 42u;

        solver_options run_opts;
        run_opts.max_iterations = 500;
        run_opts.set_objective_threshold(1e-14);
        run_opts.set_step_threshold(1e-14);

        step_budget_solver solver{policy, problem, x0, run_opts};
        auto result = solver.solve(run_opts);

        CHECK(result.status != solver_status::invalid_problem);
        CHECK(result.objective_value < 1e-6);
    }
}
