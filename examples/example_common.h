// Small shared helpers for the examples (not part of the argmin API).

#ifndef HPP_GUARD_ARGMIN_EXAMPLES_EXAMPLE_COMMON_H
#define HPP_GUARD_ARGMIN_EXAMPLES_EXAMPLE_COMMON_H

#include <argmin/result/status.h>

#include <string_view>

namespace example
{

// A human-readable name for a terminal solver_status. Convergence can be
// reported under several codes depending on which tolerance tripped first
// (converged / xtol_reached / ftol_reached are all successful stops).
inline std::string_view status_name(argmin::solver_status s)
{
    using argmin::solver_status;
    switch(s)
    {
        case solver_status::running:                     return "running";
        case solver_status::converged:                   return "converged";
        case solver_status::max_iterations:              return "max_iterations";
        case solver_status::budget_exhausted:            return "budget_exhausted";
        case solver_status::stalled:                     return "stalled";
        case solver_status::diverged:                    return "diverged";
        case solver_status::xtol_reached:                return "xtol_reached";
        case solver_status::ftol_reached:                return "ftol_reached";
        case solver_status::maxeval_reached:             return "maxeval_reached";
        case solver_status::roundoff_limited:            return "roundoff_limited";
        case solver_status::trust_region_step_rejected:  return "trust_region_step_rejected";
        case solver_status::objective_stalled:           return "objective_stalled";
        case solver_status::time_limit_reached:          return "time_limit_reached";
        case solver_status::aborted:                     return "aborted";
        case solver_status::invalid_problem:             return "invalid_problem";
    }
    return "unknown";
}

}

#endif
