#ifndef HPP_GUARD_ARGMIN_DETAIL_EXPECTED_FALLBACK_H
#define HPP_GUARD_ARGMIN_DETAIL_EXPECTED_FALLBACK_H

/// @brief Hand-rolled C++20 fallback for std::expected.
///
/// Signature-equal to the std::expected subset argmin uses: construction from
/// T, from a convertible U, from unexpected<E>, and in-place unexpect_t tag
/// construction; has_value(), explicit operator bool, ref-qualified operator*,
/// operator->, value(), error(), and value_or(). The expected<void, E>
/// specialization covers fallible operations without a payload.
///
/// Monadic operations (and_then, or_else, transform) are intentionally
/// omitted; no caller in the library uses them. Add them here if a future
/// caller needs them. This fallback never raises an exception.

#include <utility>
#include <variant>
#include <type_traits>

namespace argmin::detail
{

template <typename E>
struct unexpected
{
    E value;

    explicit constexpr unexpected(E e)
        : value(std::move(e))
    {
    }
};

struct unexpect_t
{
    explicit unexpect_t() = default;
};

inline constexpr unexpect_t unexpect{};

template <typename T, typename E>
class expected
{
    std::variant<T, unexpected<E>> m_storage;

public:
    constexpr expected(T val)
        : m_storage(std::move(val))
    {
    }

    template <typename U, std::enable_if_t<std::is_constructible_v<T, U> && !std::is_same_v<std::remove_cvref_t<U>, expected>, int> = 0>
    constexpr expected(U&& val)
        : m_storage(T(std::forward<U>(val)))
    {
    }

    constexpr expected(unexpected<E> err)
        : m_storage(std::move(err))
    {
    }

    template <typename... Args>
    constexpr explicit expected(unexpect_t, Args&&... args)
        : m_storage(unexpected<E>(E(std::forward<Args>(args)...)))
    {
    }

    constexpr explicit operator bool() const noexcept
    {
        return m_storage.index() == 0;
    }
    constexpr bool has_value() const noexcept
    {
        return m_storage.index() == 0;
    }

    constexpr T& operator*() &
    {
        return std::get<0>(m_storage);
    }
    constexpr const T& operator*() const&
    {
        return std::get<0>(m_storage);
    }
    constexpr T&& operator*() &&
    {
        return std::get<0>(std::move(m_storage));
    }
    constexpr const T&& operator*() const&&
    {
        return std::get<0>(std::move(m_storage));
    }

    constexpr T* operator->()
    {
        return &std::get<0>(m_storage);
    }
    constexpr const T* operator->() const
    {
        return &std::get<0>(m_storage);
    }

    constexpr T& value() &
    {
        return std::get<0>(m_storage);
    }
    constexpr const T& value() const&
    {
        return std::get<0>(m_storage);
    }
    constexpr T&& value() &&
    {
        return std::get<0>(std::move(m_storage));
    }
    constexpr const T&& value() const&&
    {
        return std::get<0>(std::move(m_storage));
    }

    constexpr E& error() &
    {
        return std::get<1>(m_storage).value;
    }
    constexpr const E& error() const&
    {
        return std::get<1>(m_storage).value;
    }
    constexpr E&& error() &&
    {
        return std::get<1>(std::move(m_storage)).value;
    }
    constexpr const E&& error() const&&
    {
        return std::get<1>(std::move(m_storage)).value;
    }

    template <typename U>
    constexpr T value_or(U&& fallback) const&
    {
        return has_value() ? **this : static_cast<T>(std::forward<U>(fallback));
    }

    template <typename U>
    constexpr T value_or(U&& fallback) &&
    {
        return has_value() ? std::move(**this) : static_cast<T>(std::forward<U>(fallback));
    }
};

template <typename E>
class expected<void, E>
{
    std::variant<std::monostate, unexpected<E>> m_storage;

public:
    constexpr expected() noexcept
        : m_storage(std::monostate{})
    {
    }

    constexpr expected(unexpected<E> err)
        : m_storage(std::move(err))
    {
    }

    template <typename... Args>
    constexpr explicit expected(unexpect_t, Args&&... args)
        : m_storage(unexpected<E>(E(std::forward<Args>(args)...)))
    {
    }

    constexpr explicit operator bool() const noexcept
    {
        return m_storage.index() == 0;
    }
    constexpr bool has_value() const noexcept
    {
        return m_storage.index() == 0;
    }

    constexpr E& error() &
    {
        return std::get<1>(m_storage).value;
    }
    constexpr const E& error() const&
    {
        return std::get<1>(m_storage).value;
    }
    constexpr E&& error() &&
    {
        return std::get<1>(std::move(m_storage)).value;
    }
    constexpr const E&& error() const&&
    {
        return std::get<1>(std::move(m_storage)).value;
    }
};

}

#endif
