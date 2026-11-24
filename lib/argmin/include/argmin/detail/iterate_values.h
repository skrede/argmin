#ifndef HPP_GUARD_ARGMIN_DETAIL_ITERATE_VALUES_H
#define HPP_GUARD_ARGMIN_DETAIL_ITERATE_VALUES_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <cmath>
#include <optional>

namespace argmin::detail
{

// The SQP states keep the objective, its gradient, the constraint values and their Jacobians at the iterate, so a
// non-finite one is seen there whichever callback produced it.
template <typename State>
concept caches_iterate_values = requires(const State& s) {
    s.objective_value;
    s.g.allFinite();
    s.c_eq.allFinite();
    s.c_ineq.allFinite();
    s.J_eq.allFinite();
    s.J_ineq.allFinite();
};

template <typename State>
bool iterate_values_finite(const State& s)
{
    if constexpr(caches_iterate_values<State>)
        return std::isfinite(s.objective_value) && s.g.allFinite() && s.c_eq.allFinite() && s.c_ineq.allFinite() && s.J_eq.allFinite() && s.J_ineq.allFinite();
    else
        return true;
}

// The moving-asymptote states keep f, g and the inequality rows and their Jacobian at the iterate; the least-squares
// states keep the residuals and their Jacobian. Both families reject a non-finite trial, so only their start is checked.
template <typename State>
concept caches_asymptote_values = requires(const State& s) {
    s.f;
    s.x_old1;
    s.g.allFinite();
    s.c_ineq.allFinite();
    s.J_ineq.allFinite();
};

template <typename State>
concept caches_residual_values = requires(const State& s) {
    s.objective_value;
    s.r.allFinite();
    s.J.allFinite();
};

template <typename State>
bool held_values_finite(const State& s)
{
    if constexpr(caches_asymptote_values<State>)
        return std::isfinite(s.f) && s.g.allFinite() && s.c_ineq.allFinite() && s.J_ineq.allFinite();
    else if constexpr(caches_residual_values<State>)
        return std::isfinite(s.objective_value) && s.r.allFinite() && s.J.allFinite();
    else
        return iterate_values_finite(s);
}

// A decorator starts where the state it wraps starts, so a non-finite start latches it as state_rejected rejects what
// that state rejects.
template <typename State>
bool start_values_finite(const State& s)
{
    if constexpr(requires { s.inner; })
        return start_values_finite(s.inner);
    else
        return held_values_finite(s);
}

// A non-finite trial is a failed trial, rejected like any other; the solve ends only once one leaves the family at its
// shrink limit (its largest damping, smallest region, asymptotes at their minimum distance, or a penalty or window that
// leaves the trial within the iterate's resolution).
inline std::optional<solver_status> non_finite_trial_status(bool trial_finite, bool at_limit)
{
    if(trial_finite || !at_limit)
        return std::nullopt;
    return solver_status::non_finite_evaluation;
}

// A family that cannot step back from a non-finite iterate ends there, and the step is never offered as the best point.
template <typename Scalar, typename State>
void flag_non_finite_iterate(step_result<Scalar>& result, const State& s)
{
    if(!iterate_values_finite(s))
        result.policy_status = solver_status::non_finite_evaluation;
}

}

#endif
