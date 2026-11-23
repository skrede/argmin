#ifndef HPP_GUARD_ARGMIN_DETAIL_ITERATE_VALUES_H
#define HPP_GUARD_ARGMIN_DETAIL_ITERATE_VALUES_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <cmath>

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

// A family that cannot step back from a non-finite iterate ends there, and the step is never offered as the best point.
template <typename Scalar, typename State>
void flag_non_finite_iterate(step_result<Scalar>& result, const State& s)
{
    if(!iterate_values_finite(s))
        result.policy_status = solver_status::non_finite_evaluation;
}

}

#endif
