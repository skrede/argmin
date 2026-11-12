#ifndef HPP_GUARD_ARGMIN_DETAIL_SOLVE_LOOP_H
#define HPP_GUARD_ARGMIN_DETAIL_SOLVE_LOOP_H

#include "argmin/detail/iterate_rank.h"
#include "argmin/detail/status_claim.h"

#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/options.h"

#include <limits>
#include <cstdint>

namespace argmin
{
namespace detail
{

constexpr solver_status budget_status(std::uint32_t iterations, std::uint32_t max_iterations) noexcept
{
    return iterations >= max_iterations ? solver_status::max_iterations : solver_status::budget_exhausted;
}

// A rejected problem is never evaluated, not even at x_0, so its solve takes no step and its result agrees with the
// solver's accessors.
template <typename Core>
typename Core::solve_result_type rejected_solve(const Core& core)
{
    return typename Core::solve_result_type{
        .status = solver_status::invalid_problem,
        .iterations = core.iteration_count(),
        .function_evaluations = core.evaluation_count(),
        .objective_value = std::numeric_limits<typename Core::scalar_type>::quiet_NaN(),
        .gradient_norm = core.gradient_norm(),
        .constraint_violation = core.constraint_violation(),
        .x = core.state().x,
    };
}

// The single solve loop body shared by every driver facade.
//
// A surface type (the step-budget driver, the time-budget drivers, the
// passive stepper) supplies its already-configured solver_core and a budget
// predicate; the loop owns the abort -> budget -> step_impl -> policy-status ->
// convergence -> best-seen -> terminal-resolution anatomy exactly once so
// distinct surfaces never duplicate it.
//
// Convergence loop structure:
//   1. Check abort flag
//   2. Check the budget predicate between steps (not inside the policy)
//   3. Execute the raw policy step
//   4. Check policy-reported failure (policy failure is final)
//   5. Check the convergence policy
//
// The loop is deliberately wall-clock-free: it reads no steady_clock and
// includes no <chrono>, so the step-budget driver (which passes an always-
// false predicate) is chrono-free by construction. The budget predicate is
// invoked with the running loop index i; a time-budget driver builds a
// predicate that captures its own deadline (and poll cadence) and reads the
// clock itself, keeping every chrono symbol confined to the time path. A
// true verdict maps to time_limit_reached. Wall_time, when reported, is
// measured by the time driver around this call -- the base result the loop
// returns carries no wall_time.
template <typename Core, typename OptsConvergence, typename BudgetExhausted>
typename Core::solve_result_type
run_solve_loop(Core& core, std::uint32_t budget,
               const solver_options<OptsConvergence>& opts,
               BudgetExhausted&& budget_exhausted)
{
    if(core.status() == solver_status::invalid_problem)
        return rejected_solve(core);

    using scalar_type = typename Core::scalar_type;
    solver_status status = solver_status::running;
    step_result<scalar_type> last{};

    // Best-seen tracking: the result reports the best point encountered, not the terminal trial, seeded from x_0 and
    // the policy's cached (f, cv) there. A policy without a problem binding (projected_gn family, Problem == void)
    // seeds f at +inf, so its first accepted step becomes the baseline.
    //
    // Reference: nlopt/src/api/nlopt.c nlopt_optimize (best-solution-returned semantics); N&W 2e Def 12.1 (primal
    //            feasibility, L-infinity composition).
    typename Core::vector_type best_x = core.state().x;
    iterate_rank<scalar_type> best{solver_status::running, core.seed_best_f(), core.seed_best_cv(), core.seed_best_bv()};
    const scalar_type feas_tol = static_cast<scalar_type>(opts.feasibility_tolerance_in_force());

    for(std::uint32_t i = 0; i < budget; ++i)
    {
        if(core.abort_requested())
        {
            status = solver_status::aborted;
            break;
        }

        // Budget predicate check (between steps, not inside the policy).
        // The predicate is handed the loop index i so a time-budget driver
        // can poll its deadline only every K iterations.
        if(budget_exhausted(i))
        {
            status = solver_status::time_limit_reached;
            break;
        }

        // step_impl() (not the public step()) runs the raw iteration: this loop owns the stopping
        // decision, so a second per-step convergence consult would double-advance windowed criteria.
        last = core.step_impl();
        status = resolve_step_status(last, opts.convergence, core.iteration_count(), feas_tol);

        const iterate_rank<scalar_type> cand{status, last.objective_value, last.constraint_violation, last.bound_violation};
        if(is_better(cand, best, feas_tol))
        {
            best_x = core.state().x;
            best = cand;
        }
        if(status != solver_status::running)
            break;
    }

    if(status == solver_status::running)
        status = budget_status(core.iteration_count(), opts.max_iterations);

    core.set_status(status);

    // Back-copy per-criterion telemetry from the caller-owned opts into the
    // core's stored convergence when types match, so convergence()
    // .last_check_results() reflects the most recent run. For an
    // OptsConvergence differing from the core's own stored Convergence type,
    // consumers read last_check_results directly from their own instance.
    core.back_copy_convergence(opts.convergence);

    return typename Core::solve_result_type{
        .status = status,
        .iterations = core.iteration_count(),
        // Genuine accumulated evaluation count (summed from each step's
        // reported step_result::evaluations), not an alias of the iteration
        // count. Policies that evaluate several times per iteration (line
        // searches) report > iterations here.
        .function_evaluations = core.evaluation_count(),
        .objective_value = best.objective_value,
        .gradient_norm = last.gradient_norm,
        .constraint_violation = best.constraint_violation,
        .x = best_x,
    };
}

}
}

#endif
