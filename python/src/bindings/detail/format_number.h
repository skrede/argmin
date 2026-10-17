#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_FORMAT_NUMBER_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_FORMAT_NUMBER_H

// The one numeric-to-text path in the binding tree, deliberately clear of the
// C++ formatting facets. A sibling binding rendered its repr and error text
// through those facets and crashed the array library outright the moment a
// module carrying its own statically linked standard-library runtime was loaded
// beside the array library's: two runtimes, two facet tables, one process.
// std::to_chars is specified to be independent of all of that. Where an
// implementation does not offer it for floating point, the fallback converts
// through the C library and then rewrites the decimal separator, so that path
// is independent of the C numeric category as well -- which matters, because
// the interpreter's own setlocale does move that category.
//
// What keeps this the only path is the negative grep over python/src, not a
// runtime test. An interpreter-side setlocale moves the C category only; the
// C++ global formatting state stays classic, so a reintroduced stream would
// still print a decimal point while a locale test watched and passed. Nor can
// any in-process test reproduce the original crash, which needed two coexisting
// standard-library runtimes. The runtime test covers the C category; the grep
// covers the rest.

#include <array>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <version>
#include <charconv>
#include <concepts>
#include <locale.h>

// Left overridable so both branches can be compiled and exercised on one host.
// The second clause exists because Apple's standard library has offered the
// floating-point conversions since its version 14 without advertising the
// feature-test macro; there they carry an availability annotation tied to the
// macOS 13.3 deployment target, which the version check mirrors.
#if !defined(ARGMIN_PYTHON_USE_TO_CHARS)
    #if defined(__cpp_lib_to_chars)
        #define ARGMIN_PYTHON_USE_TO_CHARS 1
    #elif defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 140000                       \
        && (!defined(__APPLE__)                                                       \
            || (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)                \
                && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 130300))
        #define ARGMIN_PYTHON_USE_TO_CHARS 1
    #else
        #define ARGMIN_PYTHON_USE_TO_CHARS 0
    #endif
#endif

namespace argmin::python
{

template <typename Integral>
    requires std::integral<Integral> && (!std::same_as<Integral, bool>)
[[nodiscard]] inline std::string format_number(Integral value)
{
    std::array<char, 24> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string(buffer.data(), converted.ptr);
}

#if ARGMIN_PYTHON_USE_TO_CHARS

[[nodiscard]] inline std::string format_number(double value)
{
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if(converted.ec != std::errc{})
        return std::string("nan");
    return std::string(buffer.data(), converted.ptr);
}

#else

namespace detail
{

inline void rewrite_decimal_separator(std::string& text)
{
    const char* const separator = localeconv()->decimal_point;
    if(separator == nullptr || (separator[0] == '.' && separator[1] == '\0'))
        return;
    const std::size_t width = std::strlen(separator);
    if(width == 0)
        return;
    for(std::size_t at = text.find(separator); at != std::string::npos;
        at = text.find(separator, at + 1))
        text.replace(at, width, 1, '.');
}

}

[[nodiscard]] inline std::string format_number(double value)
{
    std::array<char, 40> buffer{};
    int written = 0;
    for(int precision = 1; precision < 17; ++precision)
    {
        const int candidate =
            std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, value);
        if(candidate > 0 && std::strtod(buffer.data(), nullptr) == value)
        {
            written = candidate;
            break;
        }
    }
    if(written <= 0)
        written = std::snprintf(buffer.data(), buffer.size(), "%.17g", value);
    std::string text(buffer.data(), static_cast<std::size_t>(written > 0 ? written : 0));
    detail::rewrite_decimal_separator(text);
    return text;
}

#endif

[[nodiscard]] inline std::string format_shape(int rows, int cols)
{
    return "(" + format_number(rows) + ", " + format_number(cols) + ")";
}

[[nodiscard]] inline std::string format_length(int length)
{
    return "(" + format_number(length) + ",)";
}

}

#endif
