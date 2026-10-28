#ifndef HPP_GUARD_ARGMIN_TESTS_GOLDEN_STATUS_TEXT_H
#define HPP_GUARD_ARGMIN_TESTS_GOLDEN_STATUS_TEXT_H

#include "argmin/result/status.h"

#include <string>
#include <cstdint>
#include <string_view>

namespace argmin::golden
{

inline std::string_view status_name(solver_status status)
{
    switch(status)
    {
    case solver_status::running:
        return "running";
    case solver_status::converged:
        return "converged";
    case solver_status::max_iterations:
        return "max_iterations";
    case solver_status::budget_exhausted:
        return "budget_exhausted";
    case solver_status::stalled:
        return "stalled";
    case solver_status::diverged:
        return "diverged";
    case solver_status::xtol_reached:
        return "xtol_reached";
    case solver_status::ftol_reached:
        return "ftol_reached";
    case solver_status::maxeval_reached:
        return "maxeval_reached";
    case solver_status::roundoff_limited:
        return "roundoff_limited";
    case solver_status::trust_region_step_rejected:
        return "trust_region_step_rejected";
    case solver_status::objective_stalled:
        return "objective_stalled";
    case solver_status::time_limit_reached:
        return "time_limit_reached";
    case solver_status::aborted:
        return "aborted";
    case solver_status::invalid_problem:
        return "invalid_problem";
    case solver_status::line_search_failed:
        return "line_search_failed";
    }
    return "unknown";
}

inline std::string status_text(solver_status status)
{
    return std::to_string(static_cast<std::int32_t>(status)) + ":" + std::string(status_name(status));
}

}

#endif
