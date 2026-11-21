#ifndef HPP_GUARD_ARGMIN_DETAIL_CRITERIA_STATE_H
#define HPP_GUARD_ARGMIN_DETAIL_CRITERIA_STATE_H

#include "argmin/result/status.h"

#include <tuple>

namespace argmin
{
namespace detail
{

template <typename Criterion>
void reset_criterion(const Criterion& criterion)
{
    if constexpr(requires { criterion.reset(); })
        criterion.reset();
}

template <typename... Criteria>
void reset_criteria(const std::tuple<Criteria...>& criteria)
{
    std::apply([](const auto&... each) { (reset_criterion(each), ...); }, criteria);
}

template <typename Criterion>
bool criterion_within_capacity(const Criterion& criterion)
{
    if constexpr(requires { Criterion::capacity; criterion.window; })
        return criterion.window <= Criterion::capacity;
    else
        return true;
}

// A windowed criterion keeps a fixed buffer, so a window it cannot hold rejects the problem instead of being shortened.
template <typename Convergence>
bool within_capacity(const Convergence& convergence)
{
    if constexpr(requires { convergence.criteria; })
        return std::apply([](const auto&... each) { return (criterion_within_capacity(each) && ...); }, convergence.criteria);
    else
        return true;
}

// A caller's options object holds the history of whatever it last ran, so an explicit solve starts it fresh unless the
// call resumes its own iteration or time budget. The core's stored object is reset only by reset and reset_clear. An
// object over capacity is left untouched: the solve loop rejects it before its first step.
template <typename Convergence>
void reset_unless_resuming(solver_status status, const Convergence& convergence)
{
    if(within_capacity(convergence) && status != solver_status::budget_exhausted && status != solver_status::time_limit_reached)
        reset_criterion(convergence);
}

}
}

#endif
