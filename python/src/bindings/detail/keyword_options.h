#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_KEYWORD_OPTIONS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_KEYWORD_OPTIONS_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"

#include "argmin/solver/options.h"

#include <nanobind/nanobind.h>

#include <limits>
#include <string>
#include <cstdint>
#include <utility>
#include <concepts>
#include <optional>
#include <string_view>

namespace argmin::python
{

// The driver keywords every bound method accepts, carried as one aggregate so
// the forwarding logic is written once. Each field stays empty unless the
// caller named it, which is what keeps the library's own defaults the single
// source of truth.
struct driver_keywords
{
    std::optional<int> max_iterations{};
    std::optional<double> feasibility_tolerance{};
    std::optional<double> constraint_tolerance{};
    std::optional<double> gradient_threshold{};
    std::optional<double> objective_threshold{};
    std::optional<double> step_threshold{};
    std::optional<double> stall_threshold{};
    std::optional<int> stall_window{};
};

// A bare cast would wrap an out-of-range Python integer into a different,
// valid-looking count, so every unsigned count field is narrowed through here.
template <std::unsigned_integral Count>
Count positive_count(int value, std::string_view name)
{
    if(value > 0 && std::cmp_less_equal(value, std::numeric_limits<Count>::max()))
        return static_cast<Count>(value);
    raise_argmin_error(error_kind::dimension_mismatch,
                       std::string(name) + " must be in [1, "
                           + format_number(std::numeric_limits<Count>::max()) + "], got "
                           + format_number(value));
}

inline solver_options<> configure(const driver_keywords& keywords)
{
    solver_options<> opts;
    if(keywords.max_iterations)
        opts.max_iterations = positive_count<std::uint32_t>(*keywords.max_iterations,
                                                            "max_iterations");
    if(keywords.feasibility_tolerance)
    {
        check_non_negative(*keywords.feasibility_tolerance, "feasibility_tolerance");
        opts.feasibility_tolerance = *keywords.feasibility_tolerance;
    }
    if(keywords.constraint_tolerance)
    {
        check_non_negative(*keywords.constraint_tolerance, "constraint_tolerance");
        opts.constraint_tolerance = *keywords.constraint_tolerance;
    }
    if(keywords.gradient_threshold)
    {
        check_non_negative(*keywords.gradient_threshold, "gradient_threshold");
        opts.set_gradient_threshold(*keywords.gradient_threshold);
    }
    if(keywords.objective_threshold)
    {
        check_non_negative(*keywords.objective_threshold, "objective_threshold");
        opts.set_objective_threshold(*keywords.objective_threshold);
    }
    if(keywords.step_threshold)
    {
        check_non_negative(*keywords.step_threshold, "step_threshold");
        opts.set_step_threshold(*keywords.step_threshold);
    }
    if(keywords.stall_threshold)
    {
        check_non_negative(*keywords.stall_threshold, "stall_threshold");
        opts.set_stall_threshold(*keywords.stall_threshold);
    }
    if(keywords.stall_window)
        opts.set_stall_window(positive_count<std::uint16_t>(*keywords.stall_window,
                                                            "stall_window"));
    return opts;
}

inline void require_callable(const nanobind::object& value, std::string_view name)
{
    if(value.is_valid() && !value.is_none() && PyCallable_Check(value.ptr()) != 0)
        return;
    raise_argmin_error(error_kind::invalid_callback, std::string(name) + " must be callable");
}

inline void require_optional_callable(const nanobind::object& value, std::string_view name)
{
    if(!value.is_valid() || value.is_none())
        return;
    require_callable(value, name);
}

inline double positive_option(double value, std::string_view name)
{
    check_finite(value, name);
    if(value > 0.0)
        return value;
    raise_argmin_error(error_kind::invalid_problem,
                       std::string(name) + " must be positive");
}

inline int constraint_count(std::optional<int> value, std::string_view name)
{
    if(!value)
        return 0;
    if(*value < 0)
        raise_argmin_error(error_kind::dimension_mismatch,
                           std::string(name) + " must not be negative");
    return *value;
}

}

#endif
