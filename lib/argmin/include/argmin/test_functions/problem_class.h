#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_PROBLEM_CLASS_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_PROBLEM_CLASS_H

#include <cstdint>
#include <string_view>

namespace argmin
{

// Bitmask classification for optimization test problems.
//
// Single flags describe atomic constraint types; combine with operator|
// for mixed-constraint problems (e.g., global | bound_constrained).
enum class problem_class : unsigned
{
    unconstrained     = 1u << 0,
    bound_constrained = 1u << 1,
    inequality        = 1u << 2,
    equality          = 1u << 3,
    mixed             = 1u << 4,
    global            = 1u << 5
};

[[nodiscard]] constexpr problem_class operator|(problem_class a,
                                                problem_class b)
{
    return static_cast<problem_class>(
        static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

[[nodiscard]] constexpr problem_class operator&(problem_class a,
                                                problem_class b)
{
    return static_cast<problem_class>(
        static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

[[nodiscard]] constexpr bool has_class(problem_class set, problem_class query)
{
    return (static_cast<unsigned>(set) & static_cast<unsigned>(query)) != 0u;
}

[[nodiscard]] constexpr auto to_string(problem_class pc) -> std::string_view
{
    switch(static_cast<unsigned>(pc))
    {
    case 1u:  return "unconstrained";
    case 2u:  return "bound_constrained";
    case 4u:  return "inequality";
    case 8u:  return "equality";
    case 16u: return "mixed";
    case 32u: return "global";
    case 6u:  return "inequality_bound_constrained";
    case 34u: return "global_bound_constrained";
    case 36u: return "global_inequality";
    case 38u: return "global_inequality_bound_constrained";
    default:  return "mixed";
    }
}

}

#endif
