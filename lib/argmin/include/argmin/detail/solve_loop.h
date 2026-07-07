#ifndef HPP_GUARD_ARGMIN_DETAIL_SOLVE_LOOP_H
#define HPP_GUARD_ARGMIN_DETAIL_SOLVE_LOOP_H

#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/options.h"

#include <cstdint>

namespace argmin
{
namespace detail
{

// Feasibility-first best-seen comparator used by the solve loop to select
// the returned iterate.
//
// Tiered ordering:
//   1. Feasible beats infeasible unconditionally.
//   2. Both feasible: prefer lower objective.
//   3. Both infeasible: prefer lower constraint violation.
//
// An iterate is "feasible" when its constraint_violation is <= feas_tol;
// unconstrained problems (cv = 0) always take the feasible branch.
//
// Reference: NLopt nlopt_optimize convention (nlopt/src/api/nlopt.c) --
//            the caller receives the best point encountered, not the
//            terminal trial. This monotonically improves the reported
//            solve_result for oscillation-prone policies (MMA, GCMMA,
//            ISRES, CMA-ES) without regressing well-behaved policies
//            whose terminal iterate is already their best.
template <typename Scalar>
constexpr bool is_better(Scalar f_cand, Scalar cv_cand,
                         Scalar f_best, Scalar cv_best,
                         Scalar feas_tol) noexcept
{
    const bool cand_feasible = cv_cand <= feas_tol;
    const bool best_feasible = cv_best <= feas_tol;
    if(cand_feasible && !best_feasible) return true;
    if(!cand_feasible && best_feasible) return false;
    if(cand_feasible && best_feasible)
        return f_cand < f_best;
    return cv_cand < cv_best;
}

// The single solve loop body shared by every driver facade.
//
// A surface type (the step-budget driver, the time-budget drivers, the
// passive stepper) supplies its already-configured solver_core and a budget
// predicate; the loop owns the abort -> budget -> step_impl -> best-seen ->
// policy-status -> convergence -> terminal-resolution anatomy exactly once so
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
    using scalar_type = typename Core::scalar_type;

    solver_status status = solver_status::running;
    step_result<scalar_type> last{};

    // Best-seen tracking (NLopt convention: the returned solve_result
    // reports the best point encountered across the entire loop, not the
    // terminal trial). Seed from the entry iterate x_0 and the policy's
    // cached (f, cv) at x_0 -- no state-field probing beyond the core's
    // seed helpers. Policies without a problem binding (projected_gn family,
    // where Problem==void) fall back to +inf sentinels; the first accepted
    // step becomes the best-seen baseline in that case, which still
    // preserves the entry invariant because state_.x is unchanged until
    // step() runs.
    //
    // Reference: nlopt/src/api/nlopt.c nlopt_optimize (best-solution-
    //            returned semantics); N&W 2e Def 12.1 (primal feasibility,
    //            L-infinity composition).
    typename Core::vector_type best_x = core.state().x;
    scalar_type best_f = core.seed_best_f();
    scalar_type best_cv = core.seed_best_cv();
    // User-supplied constraint_tolerance takes precedence over the baked-in
    // feasibility_tolerance default: a caller who bothered to tighten the KKT
    // residual gate expects the best-seen comparator to honor the same floor.
    const scalar_type feas_tol = static_cast<scalar_type>(
        opts.constraint_tolerance.value_or(opts.feasibility_tolerance));

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

        // step_impl() (not the public step()) runs the raw iteration:
        // this loop owns the stopping decision through opts.convergence, so
        // a second per-step convergence consult inside step() would
        // double-advance windowed criteria.
        last = core.step_impl();

        // Best-seen update: compare after every step including the one that
        // triggers policy_status or convergence, so the terminal iterate
        // itself is eligible to be the winner. state_.x has already been
        // updated by policy.step(); the policy-reported (f, cv) in last are
        // the values at that x.
        if(is_better(last.objective_value, last.constraint_violation,
                     best_f, best_cv, feas_tol))
        {
            best_x = core.state().x;
            best_f = last.objective_value;
            best_cv = last.constraint_violation;
        }

        // Policy-reported failure is final.
        if(last.policy_status)
        {
            status = *last.policy_status;
            break;
        }

        // Convergence policy check
        auto conv = opts.convergence.check(last, core.iteration_count());
        if(conv)
        {
            status = *conv;
            break;
        }
    }

    if(status == solver_status::running)
    {
        status = (core.iteration_count() >= opts.max_iterations)
                     ? solver_status::max_iterations
                     : solver_status::budget_exhausted;
    }

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
        .objective_value = best_f,
        .gradient_norm = last.gradient_norm,
        .constraint_violation = best_cv,
        .x = best_x,
    };
}

}
}

#endif
