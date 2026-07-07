#ifndef HPP_GUARD_ARGMIN_EXPECTED_H
#define HPP_GUARD_ARGMIN_EXPECTED_H

/// @brief Vocabulary result type for every fallible argmin operation.
///
/// `argmin::expected` resolves to `std::expected` when the standard library
/// provides it (C++23 and later) and to an in-library C++20 fallback with an
/// identical call surface otherwise. Callers cannot tell which target is
/// active; both spell construction, `has_value()`, `operator*`, `value()`,
/// and `error()` identically.

#include "argmin/detail/expected.h"

namespace argmin
{

template <typename T, typename E>
using expected = detail::expected<T, E>;

template <typename E>
using unexpected = detail::unexpected<E>;

using detail::unexpect_t;
using detail::unexpect;

}

#endif
