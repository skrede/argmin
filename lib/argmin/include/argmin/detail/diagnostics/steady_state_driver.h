#ifndef HPP_GUARD_ARGMIN_DETAIL_DIAGNOSTICS_STEADY_STATE_DRIVER_H
#define HPP_GUARD_ARGMIN_DETAIL_DIAGNOSTICS_STEADY_STATE_DRIVER_H

// Steady-state allocation driver: drives a fixed-N solver through a warmup
// boundary and an armed hot loop, then RETURNS the observed allocation counts.
// It never prints, never exits, and carries no assertion -- the consuming
// gate or benchmark decides pass/fail from the returned struct.
//
// Warmup boundary: a full warmup solve() walks the whole descent trajectory
// unarmed, warming every one-time / lazy allocation (the QP-solver workspace
// built at construction and the per-state buffer resizes). reset(x0) then
// returns to the start OUTSIDE any armed region, a short unarmed transient
// re-enters the steady descent, and only then does the armed window measure
// the pure per-step traffic. reset() never sits inside the armed window, so a
// reset-time allocation can never be miscounted as per-step. terminated_early
// records whether the policy signaled termination inside the armed window: a
// window that is not pre-convergence is not a steady state, and a zero read
// from it would be vacuous.

#include "argmin/detail/diagnostics/alloc_counter.h"

#include "argmin/solver/step_budget_solver.h"

#include <cstddef>

namespace argmin::detail::bench
{

struct steady_state_result
{
    std::size_t eigen_malloc;
    std::size_t c_alloc;
    std::size_t armed_steps;
    bool terminated_early;
};

template <typename Policy, typename Problem>
steady_state_result measure_steady(Policy policy, const Problem& problem,
                                   const argmin::solver_options<>& opts,
                                   std::size_t hot_steps)
{
    auto x0 = problem.initial_point();
    argmin::step_budget_solver solver{policy, problem, x0, opts};

    solver.solve();
    solver.reset(x0);
    solver.step();
    solver.step();

    reset_alloc_count();
    arm_alloc_trace();
    std::size_t armed_steps = 0;
    bool terminated_early = false;
    for(std::size_t i = 0; i < hot_steps; ++i)
    {
        const auto r = solver.step();
        ++armed_steps;
        if(r.policy_status.has_value())
            terminated_early = true;
    }
    disarm_alloc_trace();

    return steady_state_result{
        read_eigen_malloc_count(),
        read_c_alloc_count(),
        armed_steps,
        terminated_early,
    };
}

}

#endif
