#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/callback_set.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/wrapper_slots.h"
#include "bindings/detail/solver_methods.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"
#include "bindings/detail/describe_options.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/kraft_slsqp_policy.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

#include <new>
#include <memory>
#include <string>
#include <cstdint>
#include <utility>
#include <optional>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using slsqp_options = kraft_slsqp_policy<>::options_type;
using slsqp_wrapper = solver_wrapper<kraft_slsqp_policy<>, constrained_problem>;

void bind_slsqp_options(nb::module_& m)
{
    nb::class_<slsqp_options>(m, "KraftSlsqpOptions",
                             "Read-only view of the sequential least-squares quadratic "
                             "programming policy's own configuration.")
        .def(nb::init<>())
        .def_ro("initial_penalty", &slsqp_options::initial_penalty)
        .def_ro("line_search", &slsqp_options::line_search)
        .def_ro("qp", &slsqp_options::qp)
        .def_ro("stall_window", &slsqp_options::stall_window)
        .def_prop_ro("bfgs_reset_max",
                     [](const slsqp_options& opts)
                     { return static_cast<int>(opts.bfgs_reset_max); })
        .def_prop_ro("multiplier_reest_every_k",
                     [](const slsqp_options& opts)
                     { return static_cast<int>(opts.multiplier_reest_every_k); })
        .def("__repr__",
             [](const slsqp_options& opts)
             {
                 return "KraftSlsqpOptions(initial_penalty=" + format_number(opts.initial_penalty)
                        + ", line_search=" + describe_line_search(opts.line_search)
                        + ", stall_window=" + format_number(opts.stall_window) + ")";
             });
}

void bind_slsqp_solver(nb::module_& m)
{
    nb::class_<slsqp_wrapper> cls(
        m, "KraftSlsqpSolver", nb::type_slots(wrapper_slots<slsqp_wrapper>),
        "Sequential least-squares quadratic programming for a constrained problem. Needs "
        "the constraint vector and its Jacobian, plus the equality and inequality counts.");
    cls.def(
        "__init__",
        [](slsqp_wrapper* self, nb::object objective, const vector<double>& x0,
           nb::object constraints, nb::object constraint_jacobian, nb::object gradient,
           std::optional<int> num_equality, std::optional<int> num_inequality,
           nb::object lower_bounds, nb::object upper_bounds,
           std::optional<double> initial_penalty, std::optional<int> policy_stall_window,
           std::optional<int> max_iterations, std::optional<double> feasibility_tolerance,
           std::optional<double> constraint_tolerance,
           std::optional<double> gradient_threshold,
           std::optional<double> objective_threshold, std::optional<double> step_threshold,
           std::optional<double> stall_threshold, std::optional<int> stall_window)
        {
            check_positive_dimension(static_cast<int>(x0.size()), "x0");
            check_all_finite(x0, "x0");
            require_callable(objective, "objective");
            require_optional_callable(gradient, "gradient");
            require_callable(constraints, "constraints");
            require_callable(constraint_jacobian, "constraint_jacobian");

            const driver_options opts =
                configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                           gradient_threshold, objective_threshold, step_threshold,
                           stall_threshold, stall_window});

            slsqp_options policy_opts;
            if(initial_penalty)
                policy_opts.initial_penalty =
                    positive_option(*initial_penalty, "initial_penalty");
            if(policy_stall_window)
            {
                check_positive_dimension(*policy_stall_window, "policy_stall_window");
                policy_opts.stall_window = static_cast<std::uint16_t>(*policy_stall_window);
            }

            callback_set calls;
            calls.objective = std::move(objective);
            calls.gradient = std::move(gradient);
            calls.constraints = std::move(constraints);
            calls.constraint_jacobian = std::move(constraint_jacobian);
            calls.lower_bounds = std::move(lower_bounds);
            calls.upper_bounds = std::move(upper_bounds);
            calls.num_equality = constraint_count(num_equality, "num_equality");
            calls.num_inequality = constraint_count(num_inequality, "num_inequality");
            check_positive_dimension(calls.num_equality + calls.num_inequality,
                                     "num_equality + num_inequality");

            auto adapter = std::make_unique<constrained_problem>(
                static_cast<int>(x0.size()), std::move(calls));
            new(self) slsqp_wrapper(std::move(adapter), x0, opts, policy_opts);
        },
        nb::arg("objective"), nb::arg("x0"), nb::arg("constraints"),
        nb::arg("constraint_jacobian"), nb::arg("gradient") = nb::none(),
        nb::arg("num_equality") = nb::none(), nb::arg("num_inequality") = nb::none(),
        nb::arg("lower_bounds") = nb::none(), nb::arg("upper_bounds") = nb::none(),
        nb::arg("initial_penalty") = nb::none(),
        nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
        nb::arg("feasibility_tolerance") = nb::none(),
        nb::arg("constraint_tolerance") = nb::none(),
        nb::arg("gradient_threshold") = nb::none(),
        nb::arg("objective_threshold") = nb::none(),
        nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
        nb::arg("stall_window") = nb::none(),
        "Sequential least-squares quadratic programming: a line-search sequential "
        "quadratic program with a BFGS Lagrangian-Hessian approximation, after Kraft "
        "(1988), DFVLR-FB 88-28. Equalities come first in the constraint vector and are "
        "driven to zero; inequalities follow and are held non-negative.");
    def_solver_methods(cls, "KraftSlsqpSolver");
}

}

void register_kraft_slsqp(nb::module_& m)
{
    bind_slsqp_options(m);
    bind_slsqp_solver(m);
}

}
