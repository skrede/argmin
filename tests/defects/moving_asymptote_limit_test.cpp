#include "moving_asymptote_problems.h"
#include "non_finite_limit_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/ccsa_quadratic_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

using ccsa = ccsa_quadratic_policy<double, 1>;
using rho_wval = alternative::gcmma::rho_wval_policy<double, 1>;
using raa_augmented = alternative::gcmma::raa_augmented_policy<double, 1>;
using move_limit_shrink = alternative::gcmma::move_limit_shrink_policy<double, 1>;

template <typename Policy>
typename Policy::options_type inner_iterations(std::optional<std::uint16_t> count)
{
    typename Policy::options_type options;
    options.max_inner_iterations = count;
    return options;
}

// Whether the step that meets the glitch is a null step that leaves the solve running, and a later step commits a trial.
template <typename Policy>
bool steps_on_after_glitch(std::optional<std::uint16_t> inner, std::int32_t glitch_call)
{
    const defects::glitch_bowl problem{{1.0, 2.0, 1, 10.0, 0.0, 3.0, 10.0}, glitch_call, 0};
    step_budget_solver<Policy, 1, defects::glitch_bowl> solver{Policy{}, problem, defects::scalar_bowl::vector::Zero(), solver_options<>{}, inner_iterations<Policy>(inner)};
    step_result<double> met{};
    for(std::int32_t k = 0; k < 40 && problem.gradient_calls < glitch_call && solver.status() == solver_status::running; ++k)
        met = solver.step();
    if(!met.is_null_step || solver.status() != solver_status::running)
        return false;
    for(std::int32_t k = 0; k < 40 && solver.status() == solver_status::running; ++k)
    {
        if(!solver.step().is_null_step)
            return true;
    }
    return false;
}

template <typename Policy>
solve_result<double, 1> solve_glitched(std::uint16_t inner, std::int32_t glitch_call)
{
    const defects::glitch_bowl problem{{1.0, 2.0, 1, 10.0, 0.0, 3.0, 10.0}, glitch_call, 0};
    step_budget_solver<Policy, 1, defects::glitch_bowl> solver{Policy{}, problem, defects::scalar_bowl::vector::Zero(), solver_options<>{}, inner_iterations<Policy>(inner)};
    return solver.solve();
}

template <typename Policy, typename Problem>
bool ends_at_start_within(const Problem& problem, std::optional<std::uint16_t> inner, std::uint32_t evaluations, std::optional<std::uint16_t> window)
{
    solver_options<> options;
    if(window)
        options.set_stall_window(*window);
    step_budget_solver<Policy, 1, Problem> solver{Policy{}, problem, Problem::vector::Constant(1.0), options, inner_iterations<Policy>(inner)};
    const solve_result<double, 1> r = solver.solve();
    return r.status == solver_status::non_finite_evaluation && r.x[0] == 1.0 && r.objective_value == 1.0 && solver.state().evaluations <= evaluations;
}

// Every trial from x = 1 toward the minimizer at 2 is NaN.
template <typename Policy>
bool ends_at_wall_within(std::optional<std::uint16_t> inner, std::uint32_t evaluations, std::optional<std::uint16_t> window = std::nullopt)
{
    return ends_at_start_within<Policy>(defects::scalar_bowl{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.0}, inner, evaluations, window);
}

// A trial within x's resolution is finite here, so the trial that ends the solve at the wall is a finite one.
template <typename Policy>
bool ends_at_finite_trial_within(std::uint32_t evaluations)
{
    return ends_at_start_within<Policy>(defects::scalar_bowl{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.0 + 5e-16}, std::nullopt, evaluations, std::nullopt);
}

// The solver reads the problem through a reference, so lifting the wall after the first solve changes what it solves.
template <typename Policy>
bool reset_after_wall_solves_as_constructed()
{
    using vector = defects::scalar_bowl::vector;
    defects::scalar_bowl problem{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.0};
    step_budget_solver<Policy, 1, defects::scalar_bowl> solver{Policy{}, problem, vector::Constant(1.0), solver_options<>{}};
    if(solver.solve().status != solver_status::non_finite_evaluation)
        return false;
    problem.nan_beyond = 10.0;
    solver.reset(vector::Constant(2.0));
    const solve_result<double, 1> after_reset = solver.solve();
    step_budget_solver<Policy, 1, defects::scalar_bowl> constructed{Policy{}, problem, vector::Constant(2.0), solver_options<>{}};
    const solve_result<double, 1> fresh = constructed.solve();
    return after_reset.status == solver_status::converged && fresh.status == after_reset.status && fresh.iterations == after_reset.iterations && fresh.x == after_reset.x;
}

}

TEST_CASE("mma tries a trial with its asymptotes at their minimum distance before ending non_finite_evaluation", "[moving_asymptote]")
{
    const defects::scalar_bowl problem{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.0};
    step_budget_solver solver{mma_policy<double, 1>{}, problem, defects::scalar_bowl::vector::Constant(1.0), solver_options<>{}};
    double distance_before_last = 0.0;
    for(std::int32_t k = 0; k < 100 && solver.status() == solver_status::running; ++k)
    {
        distance_before_last = solver.state().U[0] - solver.state().x[0];
        solver.step();
    }
    CHECK(solver.status() == solver_status::non_finite_evaluation);
    CHECK(distance_before_last <= 3e-4 * (1.0 + 1e-12));
}

TEST_CASE("a conservative moving-asymptote variant takes a null step on a NaN gradient and steps on", "[moving_asymptote]")
{
    CHECK(steps_on_after_glitch<ccsa>(1, 2));
    CHECK(steps_on_after_glitch<rho_wval>(1, 2));
    CHECK(steps_on_after_glitch<raa_augmented>(1, 2));
    CHECK(steps_on_after_glitch<move_limit_shrink>(1, 2));
    CHECK(steps_on_after_glitch<move_limit_shrink>(std::nullopt, 3));
}

TEST_CASE("a conservative moving-asymptote variant with one inner iteration converges after a NaN gradient", "[moving_asymptote]")
{
    const solve_result<double, 1> results[] = {solve_glitched<ccsa>(1, 2), solve_glitched<rho_wval>(1, 2), solve_glitched<raa_augmented>(1, 2)};
    for(const solve_result<double, 1>& r : results)
    {
        CHECK(r.status == solver_status::converged);
        CHECK(std::abs(r.x[0] - 2.0) <= 1e-4);
    }
}

TEST_CASE("the conservative moving-asymptote variants end non_finite_evaluation at their limit when every trial is non-finite", "[moving_asymptote]")
{
    CHECK(ends_at_wall_within<ccsa>(std::nullopt, 40));
    CHECK(ends_at_wall_within<rho_wval>(std::nullopt, 40));
    CHECK(ends_at_wall_within<raa_augmented>(std::nullopt, 40));
    CHECK(ends_at_wall_within<move_limit_shrink>(std::nullopt, 80));
    CHECK(ends_at_wall_within<ccsa>(1, 40));
    CHECK(ends_at_wall_within<rho_wval>(1, 40));
    CHECK(ends_at_wall_within<raa_augmented>(1, 40));
    CHECK(ends_at_wall_within<move_limit_shrink>(2, 80));
}

TEST_CASE("a run of null steps against a non-finite region ends non_finite_evaluation whatever the stall window", "[moving_asymptote]")
{
    CHECK(ends_at_wall_within<move_limit_shrink>(1, 80));
    CHECK(ends_at_wall_within<ccsa>(1, 40, 10));
    CHECK(ends_at_wall_within<rho_wval>(1, 40, 10));
    CHECK(ends_at_wall_within<raa_augmented>(1, 40, 10));
    CHECK(ends_at_wall_within<move_limit_shrink>(2, 80, 10));
    CHECK(ends_at_wall_within<move_limit_shrink>(std::nullopt, 80, 3));
}

TEST_CASE("a finite trial at the iterate after a non-finite rejection ends non_finite_evaluation", "[moving_asymptote]")
{
    CHECK(ends_at_finite_trial_within<ccsa>(40));
    CHECK(ends_at_finite_trial_within<rho_wval>(40));
    CHECK(ends_at_finite_trial_within<raa_augmented>(40));
    CHECK(ends_at_finite_trial_within<move_limit_shrink>(80));
}

TEST_CASE("rho_wval and raa_augmented grow the penalty on a merit-committed last trial with a non-finite gradient", "[moving_asymptote]")
{
    const defects::gradient_cliff_bowl problem{{{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.0}, 0.9}};
    CHECK(ends_at_start_within<rho_wval>(problem, 1, 40, std::nullopt));
    CHECK(ends_at_start_within<raa_augmented>(problem, 1, 40, std::nullopt));
}

TEST_CASE("a reset after a non-finite end clears the moving-asymptote non-finite record", "[moving_asymptote]")
{
    CHECK(reset_after_wall_solves_as_constructed<ccsa>());
    CHECK(reset_after_wall_solves_as_constructed<rho_wval>());
    CHECK(reset_after_wall_solves_as_constructed<raa_augmented>());
    CHECK(reset_after_wall_solves_as_constructed<move_limit_shrink>());
}
