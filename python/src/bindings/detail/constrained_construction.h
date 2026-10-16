#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CONSTRAINED_CONSTRUCTION_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CONSTRAINED_CONSTRUCTION_H

#include "bindings/detail/errors.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/callback_set.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"

#include <nanobind/nanobind.h>

#include <memory>
#include <utility>
#include <optional>

namespace argmin::python
{

struct constrained_callables
{
    nanobind::object objective;
    nanobind::object gradient;
    nanobind::object constraints;
    nanobind::object constraint_jacobian;
    nanobind::object lower_bounds;
    nanobind::object upper_bounds;
};

// constrained_problem has no bound accessors, so a box passed here would never
// reach the policy; refusing it keeps the keyword from being accepted and ignored.
inline void reject_bounds(const nanobind::object& lower, const nanobind::object& upper)
{
    const auto absent = [](const nanobind::object& bound)
    { return !bound.is_valid() || bound.is_none(); };
    if(absent(lower) && absent(upper))
        return;
    raise_argmin_error(error_kind::invalid_problem, "bounds are not supported by this solver yet");
}

inline void check_constrained_inputs(const vector<double>& x0, const constrained_callables& calls)
{
    check_positive_dimension(static_cast<int>(x0.size()), "x0");
    check_all_finite(x0, "x0");
    require_callable(calls.objective, "objective");
    require_optional_callable(calls.gradient, "gradient");
    require_callable(calls.constraints, "constraints");
    require_callable(calls.constraint_jacobian, "constraint_jacobian");
}

inline callback_set constrained_callback_set(constrained_callables calls,
                                             std::optional<int> num_equality,
                                             std::optional<int> num_inequality)
{
    callback_set set;
    set.objective = std::move(calls.objective);
    set.gradient = std::move(calls.gradient);
    set.constraints = std::move(calls.constraints);
    set.constraint_jacobian = std::move(calls.constraint_jacobian);
    set.lower_bounds = std::move(calls.lower_bounds);
    set.upper_bounds = std::move(calls.upper_bounds);
    set.num_equality = constraint_count(num_equality, "num_equality");
    set.num_inequality = constraint_count(num_inequality, "num_inequality");
    check_positive_dimension(set.num_equality + set.num_inequality,
                             "num_equality + num_inequality");
    return set;
}

inline std::unique_ptr<constrained_problem> make_constrained_problem(const vector<double>& x0,
                                                                     callback_set calls)
{
    return std::make_unique<constrained_problem>(static_cast<int>(x0.size()), std::move(calls));
}

}

#endif
