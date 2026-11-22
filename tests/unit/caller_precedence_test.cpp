#include "precedence_visitor.h"

#include "cases/sqp_cases.h"
#include "cases/gradient_cases.h"
#include "cases/stochastic_cases.h"
#include "cases/least_squares_cases.h"
#include "cases/derivative_free_cases.h"
#include "cases/moving_asymptote_cases.h"
#include "cases/trust_region_sqp_cases.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/result/status.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>

using namespace argmin;
using argmin::precedence::stall_of;

namespace
{
using lbfgsb2 = lbfgsb_policy<double, 2>;
using lbfgsb2_solver = step_budget_solver<lbfgsb2, 2, rosenbrock<double, 2>>;

lbfgsb2_solver lbfgsb_hinting(const rosenbrock<double, 2>& problem, std::uint16_t window, const solver_options<>& opts)
{
    return {lbfgsb2{}, problem, problem.initial_point(), opts, lbfgsb2::options_type{.stall_window = window}};
}

using mma4 = mma_policy<double, 4>;
using ccsa4 = ccsa_quadratic_policy<double, 4>;

// The moving-asymptote policies hold no options of their own, so their hints come only from options given here.
template <typename Policy>
step_budget_solver<Policy, 4, hs076<>> with_policy_options(const hs076<>& problem, const Eigen::Vector4d& x0, const solver_options<>& opts)
{
    return {Policy{}, problem, x0, opts, typename Policy::options_type{}};
}

// A threshold no metric change can reach fires the stall test at the first iteration the window allows.
solver_options<> stalling_at(std::uint32_t max_iterations)
{
    solver_options<> opts;
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(1e-30);
    opts.set_stall_threshold(1e100);
    return opts;
}
}

TEST_CASE("a caller's value wins over every gradient-family hint on both paths", "[precedence]")
{
    precedence::visitor visitor;
    cases::visit_gradient_cases(visitor);
}

TEST_CASE("a caller's value wins over every sequential quadratic programming hint on both paths", "[precedence]")
{
    precedence::visitor visitor;
    cases::visit_sqp_cases(visitor);
    cases::visit_trust_region_sqp_cases(visitor);
}

TEST_CASE("a caller's value wins over every least-squares and derivative-free hint on both paths", "[precedence]")
{
    precedence::visitor visitor;
    cases::visit_least_squares_cases(visitor);
    cases::visit_derivative_free_cases(visitor);
}

TEST_CASE("a caller's value wins over every moving-asymptote and stochastic hint on both paths", "[precedence]")
{
    precedence::visitor visitor;
    cases::visit_moving_asymptote_cases(visitor);
    cases::visit_stochastic_cases(visitor);
    CHECK(visitor.asymptote_cells() == cases::moving_asymptote_keys.size());
}

TEST_CASE("an options object reused across policies has the current policy's window in force", "[precedence]")
{
    const rosenbrock<double, 2> problem{};
    solver_options<> opts;
    opts.max_iterations = 1;
    lbfgsb_hinting(problem, 100, opts).solve(opts);
    CHECK(stall_of(opts.convergence).window_in_force() == 100);
    lbfgsb_hinting(problem, 200, opts).solve(opts);
    CHECK(stall_of(opts.convergence).window_in_force() == 200);
    step_budget_solver{multistart_policy<lbfgsb_policy<>>{}, problem, problem.initial_point(), opts}.solve(opts);
    CHECK(stall_of(opts.convergence).window_in_force() == convergence_defaults<double>::stall_window);
    CHECK_FALSE(stall_of(opts.convergence).window.has_value());
}

TEST_CASE("the stall window in force decides the stalling iteration on both paths", "[precedence]")
{
    const rosenbrock<double, 2> problem{};
    solver_options<> caller = stalling_at(100);
    caller.set_stall_window(7);
    lbfgsb2_solver stored = lbfgsb_hinting(problem, 9, caller);
    CHECK(stored.solve().iterations == 7);
    CHECK(lbfgsb_hinting(problem, 9, caller).solve(caller).iterations == 7);
    const solver_options<> unset = stalling_at(100);
    CHECK(lbfgsb_hinting(problem, 9, unset).solve().iterations == 9);
    CHECK(lbfgsb_hinting(problem, 9, unset).solve(unset).iterations == 9);
    CHECK(stored.status() == solver_status::stalled);
}

TEST_CASE("the stall threshold in force decides whether a moving-asymptote solve stalls on both paths", "[precedence]")
{
    const hs076<> problem{};
    solver_options<> caller = stalling_at(100);
    caller.set_stall_window(3);
    CHECK(with_policy_options<mma4>(problem, problem.initial_point(), caller).solve().iterations == 3);
    CHECK(with_policy_options<mma4>(problem, problem.initial_point(), caller).solve(caller).iterations == 3);
    solver_options<> hinted = caller;
    stall_of(hinted.convergence).threshold.reset();
    CHECK(with_policy_options<mma4>(problem, problem.initial_point(), hinted).solve().iterations > 3);
    CHECK(with_policy_options<mma4>(problem, problem.initial_point(), hinted).solve(hinted).iterations > 3);
}

// With no iterate counted as feasible the best-seen metric never becomes finite, so it cannot stall where the combined
// objective-plus-violation metric does.
TEST_CASE("the best-seen flag in force decides the stall metric on both paths", "[precedence]")
{
    const hs076<> problem{};
    solver_options<> caller = stalling_at(100);
    caller.set_stall_window(3);
    stall_of(caller.convergence).feasibility_tolerance = -1.0;
    stall_of(caller.convergence).track_best_seen_feasible = false;
    CHECK(with_policy_options<ccsa4>(problem, problem.initial_point(), caller).solve().iterations == 3);
    CHECK(with_policy_options<ccsa4>(problem, problem.initial_point(), caller).solve(caller).iterations == 3);
    solver_options<> hinted = caller;
    stall_of(hinted.convergence).track_best_seen_feasible.reset();
    CHECK(with_policy_options<ccsa4>(problem, problem.initial_point(), hinted).solve().iterations > 3);
    CHECK(with_policy_options<ccsa4>(problem, problem.initial_point(), hinted).solve(hinted).iterations > 3);
}

TEST_CASE("the augmented Lagrangian tests feasibility against the solver's tolerance, then its own, then the default", "[precedence]")
{
    const hs071<> problem{};
    using al = augmented_lagrangian_policy<lbfgsb_policy<>>;
    const auto in_force = [&](std::optional<double> solver_level, std::optional<double> policy_level)
    {
        solver_options<> opts;
        opts.constraint_tolerance = solver_level;
        return step_budget_solver<al, dynamic_dimension, hs071<>>{al{}, problem, problem.initial_point(), opts, al::options_type{.constraint_tolerance = policy_level}}.state().constraint_tolerance_in_force;
    };
    CHECK(in_force(1e-3, 1e-9) == 1e-3);
    CHECK(in_force(std::nullopt, 1e-9) == 1e-9);
    CHECK(in_force(std::nullopt, std::nullopt) == augmented_lagrangian_defaults<double>::constraint_tolerance);
}
