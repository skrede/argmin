#ifndef HPP_GUARD_NABLAPP_RESULT_STATUS_H
#define HPP_GUARD_NABLAPP_RESULT_STATUS_H

#include <cstdint>

namespace nablapp
{

// Solver termination status.
//
// Returned by basic_solver to indicate why iteration stopped.
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
    objective_stalled,
    time_limit_reached,
    aborted
};

}

#endif
