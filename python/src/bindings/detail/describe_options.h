#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_DESCRIBE_OPTIONS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_DESCRIBE_OPTIONS_H

#include "bindings/detail/format_number.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/line_search/options.h"

#include <tuple>
#include <string>
#include <optional>

namespace argmin::python
{

using driver_options = solver_options<>;

template <typename Criterion>
const Criterion& criterion_of(const driver_options& opts)
{
    return std::get<Criterion>(opts.convergence.criteria);
}

inline std::string describe_line_search(const line_search_options<>& opts)
{
    return "LineSearchOptions(c1=" + format_number(opts.c1) + ", c2=" + format_number(opts.c2)
           + ", rho=" + format_number(opts.rho) + ", max_alpha=" + format_number(opts.max_alpha)
           + ", max_iterations=" + format_number(opts.max_iterations) + ")";
}

inline std::string describe_driver_options(const driver_options& opts)
{
    const auto& tolerance = opts.constraint_tolerance;
    return "SolverOptions(max_iterations=" + format_number(opts.max_iterations)
           + ", constraint_tolerance="
           + (tolerance ? format_number(*tolerance) : std::string("None"))
           + ", feasibility_tolerance=" + format_number(opts.feasibility_tolerance)
           + ", gradient_threshold="
           + format_number(criterion_of<gradient_tolerance_criterion<>>(opts).threshold)
           + ", objective_threshold="
           + format_number(criterion_of<objective_tolerance_criterion<>>(opts).threshold)
           + ", stationarity_threshold="
           + format_number(criterion_of<objective_tolerance_criterion<>>(opts).stationarity_threshold)
           + ", step_threshold="
           + format_number(criterion_of<step_tolerance_criterion<>>(opts).threshold) + ")";
}

inline std::string describe_optional(const std::optional<double>& value)
{
    return value ? format_number(*value) : std::string("None");
}

}

#endif
