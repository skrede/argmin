#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_SAME_BITS_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_SAME_BITS_H

#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <bit>
#include <array>
#include <cstdint>
#include <optional>

namespace argmin::property
{

inline bool same_bits(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

inline bool same_bits(const std::optional<double>& a, const std::optional<double>& b)
{
    return a.has_value() == b.has_value() && (!a || same_bits(*a, *b));
}

template <typename A, typename B>
bool same_bits(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b)
{
    if(a.size() != b.size())
        return false;
    for(Eigen::Index i = 0; i < a.size(); ++i)
    {
        if(!same_bits(static_cast<double>(a(i)), static_cast<double>(b(i))))
            return false;
    }
    return true;
}

inline bool same_bits(const solve_result<double>& a, const solve_result<double>& b)
{
    if(a.status != b.status || a.iterations != b.iterations || a.function_evaluations != b.function_evaluations)
        return false;
    return same_bits(a.objective_value, b.objective_value) && same_bits(a.gradient_norm, b.gradient_norm) && same_bits(a.constraint_violation, b.constraint_violation) && same_bits(a.x, b.x);
}

inline bool same_bits(const step_result<double>& a, const step_result<double>& b)
{
    if(a.improved != b.improved || a.is_null_step != b.is_null_step || a.policy_status != b.policy_status || a.evaluations != b.evaluations)
        return false;
    if(!same_bits(a.objective_value, b.objective_value) || !same_bits(a.gradient_norm, b.gradient_norm) || !same_bits(a.step_size, b.step_size))
        return false;
    return same_bits(a.objective_change, b.objective_change) && same_bits(a.constraint_violation, b.constraint_violation) && same_bits(a.x_norm, b.x_norm) && same_bits(a.kkt_residual, b.kkt_residual);
}

// The counters a state keeps under its own name; a reset must restart every one of them.
template <typename State>
std::array<std::uint64_t, 4> counters_of(const State& s)
{
    std::array<std::uint64_t, 4> counters{};
    if constexpr(requires { s.iteration; })
        counters[0] = s.iteration;
    if constexpr(requires { s.outer_iter; })
        counters[1] = s.outer_iter;
    if constexpr(requires { s.restart_count; })
        counters[2] = s.restart_count;
    if constexpr(requires { s.line_search_calls; })
        counters[3] = s.line_search_calls;
    return counters;
}

}

#endif
