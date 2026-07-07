#ifndef HPP_GUARD_ARGMIN_DETAIL_EXPECTED_H
#define HPP_GUARD_ARGMIN_DETAIL_EXPECTED_H

/// @brief Feature-test switch between std::expected and the C++20 fallback.
///
/// When the standard library ships std::expected (__cpp_lib_expected >=
/// 202202L) the detail aliases forward to it; otherwise the hand-rolled
/// C++20 implementation in "argmin/detail/expected_fallback.h" provides the
/// same surface.

#include <version>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L

    #include <expected>

namespace argmin::detail
{

template <typename T, typename E>
using expected = std::expected<T, E>;

template <typename E>
using unexpected = std::unexpected<E>;

using unexpect_t = std::unexpect_t;
inline constexpr unexpect_t unexpect{std::unexpect};

}

#else

    #include "argmin/detail/expected_fallback.h"

#endif

#endif
