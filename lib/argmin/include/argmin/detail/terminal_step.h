#ifndef HPP_GUARD_ARGMIN_DETAIL_TERMINAL_STEP_H
#define HPP_GUARD_ARGMIN_DETAIL_TERMINAL_STEP_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>

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

// A step that leaves the iterate in place and reports no stationarity measure, so no criterion ends a solve on it.
template <typename Scalar>
step_result<Scalar> held_step(Scalar objective_value, Scalar constraint_violation, Scalar x_norm, std::optional<solver_status> status = std::nullopt)
{
    return step_result<Scalar>{
        .objective_value = objective_value,
        .gradient_norm = std::numeric_limits<Scalar>::infinity(),
        .step_size = Scalar(0),
        .objective_change = Scalar(0),
        .improved = false,
        .is_null_step = true,
        .constraint_violation = constraint_violation,
        .x_norm = x_norm,
        .policy_status = status,
    };
}

// A population whose every fitness is non-finite carries no ranking to move a search distribution by.
template <typename Scalar>
std::optional<step_result<Scalar>> unranked_population(Scalar best_fitness, Scalar objective_value, Scalar x_norm, Scalar constraint_violation = Scalar(0))
{
    if(std::isfinite(best_fitness))
        return std::nullopt;
    return step_result<Scalar>{
        .objective_value = objective_value,
        .gradient_norm = Scalar(0.0),
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .is_null_step = true,
        .constraint_violation = constraint_violation,
        .x_norm = x_norm,
        .policy_status = solver_status::non_finite_evaluation,
    };
}

// A search distribution with a non-finite sigma, eigenvalue or covariance entry has diverged. allFinite, not
// isfinite(maxCoeff()): an IEEE 754 max-reduction returns the finite operand of a NaN pair (IEEE Std 754-2019 6.2).
template <typename Scalar, typename State>
std::optional<step_result<Scalar>> diverged_search(const State& s)
{
    if(std::isfinite(s.sigma) && s.D.allFinite() && s.C.allFinite())
        return std::nullopt;
    return step_result<Scalar>{
        .objective_value = s.objective_value,
        .gradient_norm = Scalar(0.0),
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .x_norm = s.x.allFinite() ? s.x.norm() : Scalar(0.0),
        .policy_status = solver_status::diverged,
    };
}

// A decorator's member steps keep the solve running: the solve's criteria never see a member's own history, so they
// must not end the solve on its steps.
template <typename Scalar>
step_result<Scalar> working_step(step_result<Scalar> member)
{
    member.policy_status = solver_status::running;
    return member;
}

// A decorator ends with the member it returns: that member's last step and status, and the evaluations of this step.
template <typename Scalar>
step_result<Scalar> member_end_step(step_result<Scalar> member, solver_status status, std::uint32_t evaluations)
{
    member.policy_status = status;
    member.evaluations = evaluations;
    return member;
}

}

#endif
