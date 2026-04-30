#ifndef HPP_GUARD_NABLAPP_DETAIL_TUPLE_CONTAINS_H
#define HPP_GUARD_NABLAPP_DETAIL_TUPLE_CONTAINS_H

// Tuple type membership trait.
//
// std::get<T>(tuple) is not SFINAE-friendly when T is not present: it
// triggers a static_assert ("the type T in std::get<T> must occur exactly
// once in the tuple") rather than substitution failure. if constexpr on
// `requires { std::get<T>(t); }` therefore silently passes for missing
// types and breaks at instantiation. Use this trait to gate the branches.

#include <tuple>
#include <type_traits>

namespace nablapp::detail
{

template <typename T, typename Tuple>
struct tuple_contains;

template <typename T, typename... Us>
struct tuple_contains<T, std::tuple<Us...>>
    : std::bool_constant<(std::is_same_v<T, Us> || ...)> {};

template <typename T, typename Tuple>
inline constexpr bool tuple_contains_v = tuple_contains<T, Tuple>::value;

}

#endif
