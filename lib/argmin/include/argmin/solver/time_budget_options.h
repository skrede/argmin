#ifndef HPP_GUARD_ARGMIN_SOLVER_TIME_BUDGET_OPTIONS_H
#define HPP_GUARD_ARGMIN_SOLVER_TIME_BUDGET_OPTIONS_H

#include "argmin/solver/options.h"

#include <chrono>
#include <cstdint>

namespace argmin
{

// Options for the time-budget drivers: the chrono-free shared core embedded
// by value, plus the two wall-clock knobs.
//
// The core is EMBEDDED (a member, not a base): there is no shared options
// inheritance hierarchy. A caller configures max_iterations / tolerances /
// convergence on `core` exactly as for the step-budget driver, then adds a
// wall-clock deadline and a poll cadence.
//
//   max_time          -- the wall-clock deadline. The drivers stop with
//                        solver_status::time_limit_reached once the elapsed
//                        time since the solve loop began reaches this.
//   time_poll_stride  -- the deadline is polled every K iterations rather
//                        than every iteration, amortizing the steady_clock
//                        read over K cheap policy steps. A stride of 1 polls
//                        every iteration; 0 is treated as "poll every
//                        iteration". The default is measured, not invented:
//                        see the poll-cadence measurement recorded alongside
//                        this milestone -- K is the smallest cadence keeping
//                        a steady_clock::now() read under ~1% of the cheapest
//                        representative policy step.
template <typename Convergence = default_convergence>
struct time_budget_options
{
    using convergence_type = Convergence;

    solver_options<Convergence> core{};
    std::chrono::nanoseconds max_time{std::chrono::nanoseconds::zero()};
    std::uint32_t time_poll_stride{20};
};

}

#endif
