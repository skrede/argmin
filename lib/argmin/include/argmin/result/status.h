#ifndef HPP_GUARD_ARGMIN_RESULT_STATUS_H
#define HPP_GUARD_ARGMIN_RESULT_STATUS_H

#include <cstdint>

namespace argmin
{

// Solver termination status.
//
// Returned by step_budget_solver to indicate why iteration stopped.
// K&W Section 4.4 (convergence criteria), N&W Section 3.1.

enum class solver_status : std::uint8_t
{
    running,
    converged,
    max_iterations,
    budget_exhausted,
    stalled,
    diverged,
    xtol_reached,
    ftol_reached,
    maxeval_reached,
    roundoff_limited,
    trust_region_step_rejected,
    objective_stalled,
    time_limit_reached,
    aborted,
    // The problem as posed violates a hard precondition of the chosen
    // method and cannot be solved as given -- for example, an
    // equality-constrained problem handed to an inequality-and-box-only
    // method whose constraint buffers have no room for equality rows.
    // Signalled at runtime (the library is exception-free) so the caller
    // observes a terminal status instead of undefined behavior.
    invalid_problem
};

}

#endif
