#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_PRINT_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_PRINT_H

#include <cstdio>
#include <cstddef>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace argmin::bench
{
namespace detail
{

struct print_format_spec
{
    int width{0};
    int precision{-1};
    char type{'\0'};
    bool left_align{false};
    bool show_sign{false};
};

[[nodiscard]] inline auto is_digit(char c) -> bool
{
    return c >= '0' && c <= '9';
}

[[nodiscard]] inline auto parse_format_spec(std::string_view field) -> print_format_spec
{
    print_format_spec spec{};
    if(!field.empty() && field.front() == ':')
        field.remove_prefix(1);

    if(!field.empty() && (field.front() == '<' || field.front() == '>'))
    {
        spec.left_align = field.front() == '<';
        field.remove_prefix(1);
    }

    if(!field.empty() && field.front() == '+')
    {
        spec.show_sign = true;
        field.remove_prefix(1);
    }

    while(!field.empty() && is_digit(field.front()))
    {
        spec.width = spec.width * 10 + (field.front() - '0');
        field.remove_prefix(1);
    }

    if(!field.empty() && field.front() == '.')
    {
        field.remove_prefix(1);
        spec.precision = 0;
        while(!field.empty() && is_digit(field.front()))
        {
            spec.precision = spec.precision * 10 + (field.front() - '0');
            field.remove_prefix(1);
        }
    }

    if(!field.empty())
        spec.type = field.front();

    return spec;
}

inline void write_chars(FILE* out, std::string_view text)
{
    if(!text.empty())
        (void)std::fwrite(text.data(), 1, text.size(), out);
}

inline void write_padding(FILE* out, int count)
{
    for(int i = 0; i < count; ++i)
        (void)std::fputc(' ', out);
}

inline void write_padded_text(FILE* out, std::string_view text, const print_format_spec& spec)
{
    const auto width = static_cast<std::size_t>(spec.width > 0 ? spec.width : 0);
    const auto pad = width > text.size() ? static_cast<int>(width - text.size()) : 0;
    if(!spec.left_align)
        write_padding(out, pad);
    write_chars(out, text);
    if(spec.left_align)
        write_padding(out, pad);
}

template <typename Int>
void write_integer(FILE* out, Int value, const print_format_spec& spec)
{
    char fmt[32]{};
    char buf[128]{};
    int pos = 0;
    fmt[pos++] = '%';
    if(spec.show_sign)
        fmt[pos++] = '+';
    if(spec.width > 0)
        pos += std::snprintf(fmt + pos, sizeof(fmt) - static_cast<std::size_t>(pos),
                             "%d", spec.width);
    fmt[pos++] = 'l';
    fmt[pos++] = 'l';
    fmt[pos++] = 'd';
    fmt[pos] = '\0';
    (void)std::snprintf(buf, sizeof(buf), fmt, static_cast<long long>(value));
    write_chars(out, std::string_view{buf, std::strlen(buf)});
}

template <typename Float>
void write_float(FILE* out, Float value, const print_format_spec& spec)
{
    char fmt[32]{};
    char buf[128]{};
    int pos = 0;
    fmt[pos++] = '%';
    if(spec.show_sign)
        fmt[pos++] = '+';
    if(spec.width > 0)
        pos += std::snprintf(fmt + pos, sizeof(fmt) - static_cast<std::size_t>(pos),
                             "%d", spec.width);
    if(spec.precision >= 0)
        pos += std::snprintf(fmt + pos, sizeof(fmt) - static_cast<std::size_t>(pos),
                             ".%d", spec.precision);
    fmt[pos++] = spec.type != '\0' ? spec.type : 'g';
    fmt[pos] = '\0';
    (void)std::snprintf(buf, sizeof(buf), fmt, static_cast<double>(value));
    write_chars(out, std::string_view{buf, std::strlen(buf)});
}

inline void write_value(FILE* out, const char* value, const print_format_spec& spec)
{
    write_padded_text(out, value == nullptr ? std::string_view{} : std::string_view{value}, spec);
}

inline void write_value(FILE* out, char* value, const print_format_spec& spec)
{
    write_value(out, static_cast<const char*>(value), spec);
}

inline void write_value(FILE* out, std::string_view value, const print_format_spec& spec)
{
    write_padded_text(out, value, spec);
}

template <typename Value>
void write_value(FILE* out, const Value& value, const print_format_spec& spec)
{
    if constexpr(std::is_floating_point_v<Value>)
        write_float(out, value, spec);
    else if constexpr(std::is_same_v<Value, bool>)
        write_integer(out, value ? 1 : 0, spec);
    else if constexpr(std::is_integral_v<Value> || std::is_enum_v<Value>)
        write_integer(out, static_cast<long long>(value), spec);
    else
        write_padded_text(out, std::string_view{value}, spec);
}

inline void write_format(FILE* out, std::string_view fmt)
{
    write_chars(out, fmt);
}

template <typename First, typename... Rest>
void write_format(FILE* out, std::string_view fmt, const First& first, const Rest&... rest)
{
    const auto open = fmt.find('{');
    if(open == std::string_view::npos)
    {
        write_chars(out, fmt);
        return;
    }
    const auto close = fmt.find('}', open);
    if(close == std::string_view::npos)
    {
        write_chars(out, fmt);
        return;
    }

    write_chars(out, fmt.substr(0, open));
    write_value(out, first, parse_format_spec(fmt.substr(open + 1, close - open - 1)));
    write_format(out, fmt.substr(close + 1), rest...);
}

}

template <typename... Args>
void print(FILE* out, std::string_view fmt, const Args&... args)
{
    detail::write_format(out, fmt, args...);
}

template <typename... Args>
void println(FILE* out, std::string_view fmt, const Args&... args)
{
    print(out, fmt, args...);
    (void)std::fputc('\n', out);
}

template <typename... Args>
void println(std::string_view fmt, const Args&... args)
{
    println(stdout, fmt, args...);
}

inline void println()
{
    (void)std::fputc('\n', stdout);
}

}

#endif
