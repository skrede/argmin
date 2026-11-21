#ifndef HPP_GUARD_ARGMIN_DETAIL_TERMINAL_STEP_H
#define HPP_GUARD_ARGMIN_DETAIL_TERMINAL_STEP_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <limits>

namespace argmin::detail
{

// A step from a terminal status runs no policy, so it reports the latched status and no value: NaN, never a
// fabricated 0 that a caller would read as a stationary or feasible iterate.
template <typename Scalar>
step_result<Scalar> latched_step(Scalar x_norm, solver_status status)
{
    constexpr Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    return step_result<Scalar>{
        .objective_value = nan,
        .gradient_norm = nan,
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .is_null_step = true,
        .constraint_violation = nan,
        .bound_violation = nan,
        .x_norm = x_norm,
        .policy_status = status,
        .evaluations = 0,
    };
}

}

#endif
