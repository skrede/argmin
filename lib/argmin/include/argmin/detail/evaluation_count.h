#ifndef HPP_GUARD_ARGMIN_DETAIL_EVALUATION_COUNT_H
#define HPP_GUARD_ARGMIN_DETAIL_EVALUATION_COUNT_H

#include "argmin/result/step_result.h"

#include <cstdint>
#include <concepts>

namespace argmin::detail
{

// A state that counts its own objective (or residual) calls; the core then reports that counter, init and reset calls
// included, instead of summing the evaluations each step claims.
template <typename State>
concept counts_evaluations = requires(const State& s) {
    { s.evaluations } -> std::convertible_to<std::uint32_t>;
};

template <typename State>
std::uint32_t evaluations_of(const State& s)
{
    if constexpr(counts_evaluations<State>)
        return s.evaluations;
    else
        return 0;
}

template <typename Scalar, typename Problem, typename X>
Scalar counted_value(std::uint32_t& evaluations, const Problem& problem, const X& x)
{
    ++evaluations;
    return problem.value(x);
}

template <typename Problem, typename X, typename R>
void counted_residuals(std::uint32_t& evaluations, const Problem& problem, const X& x, R& r)
{
    ++evaluations;
    problem.residuals(x, r);
}

// The counter is unsigned, so the difference is exact across a wrap.
template <typename State, typename Scalar>
void settle_step_evaluations(step_result<Scalar>& result, const State& s, std::uint32_t before)
{
    if constexpr(counts_evaluations<State>)
        result.evaluations = s.evaluations - before;
}

}

#endif
