#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/wrapper_slots.h"
#include "bindings/detail/solver_methods.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"
#include "bindings/detail/describe_options.h"
#include "bindings/detail/constrained_construction.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/filter_slsqp_policy.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

#include <new>
#include <string>
#include <cstdint>
#include <utility>
#include <optional>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using filter_options = filter_slsqp_policy<>::options_type;
using filter_wrapper = solver_wrapper<filter_slsqp_policy<>, constrained_problem>;

constexpr const char* filter_solver_doc =
    "Filter line-search sequential quadratic programming for a constrained problem: a "
    "trial step is accepted when it improves either the objective or the constraint "
    "violation against a filter of earlier pairs, after Fletcher and Leyffer (2002), "
    "Mathematical Programming 91, with feasibility restoration when no step is acceptable. "
    "Needs the constraint vector and its Jacobian, plus the equality and inequality counts. "
    "Equalities come first in the constraint vector and are driven to zero; inequalities "
    "follow and are held non-negative. The lower_bounds and upper_bounds keywords are "
    "rejected, because this solver does not support bounds yet.";

const char* restoration_name(detail::restoration_strategy strategy)
{
    switch(strategy)
    {
        case detail::restoration_strategy::l1_penalty: return "l1_penalty";
        case detail::restoration_strategy::feasibility_qp: return "feasibility_qp";
        case detail::restoration_strategy::hybrid: return "hybrid";
    }
    return "unknown";
}

std::string describe_filter_options(const filter_options& opts)
{
    return "FilterSlsqpOptions(line_search=" + describe_line_search(opts.line_search)
           + ", restoration=" + restoration_name(opts.restoration)
           + ", max_restoration_steps=" + format_number(opts.max_restoration_steps)
           + ", stall_window=" + format_number(opts.stall_window)
           + ", gamma_f=" + format_number(opts.gamma_f)
           + ", gamma_h=" + format_number(opts.gamma_h) + ")";
}

void bind_restoration_strategy(nb::module_& m)
{
    nb::enum_<detail::restoration_strategy>(
        m, "RestorationStrategy",
        "How the filter method regains feasibility when the filter rejects every trial step: "
        "descent on the l1 constraint violation, a feasibility quadratic program, or the "
        "l1 descent first with the quadratic program as its fallback.")
        .value("l1_penalty", detail::restoration_strategy::l1_penalty)
        .value("feasibility_qp", detail::restoration_strategy::feasibility_qp)
        .value("hybrid", detail::restoration_strategy::hybrid);
}

void bind_filter_options(nb::module_& m)
{
    nb::class_<filter_options>(m, "FilterSlsqpOptions",
                               "Read-only view of the filter line-search sequential quadratic "
                               "programming policy's own configuration.")
        .def(nb::init<>())
        .def_ro("line_search", &filter_options::line_search)
        .def_ro("qp", &filter_options::qp)
        .def_ro("restoration", &filter_options::restoration)
        .def_ro("max_restoration_steps", &filter_options::max_restoration_steps)
        .def_ro("stall_window", &filter_options::stall_window)
        .def_ro("gamma_f", &filter_options::gamma_f)
        .def_ro("gamma_h", &filter_options::gamma_h)
        .def_prop_ro("bfgs_reset_max", [](const filter_options& opts)
                     { return static_cast<int>(opts.bfgs_reset_max); })
        .def_prop_ro("multiplier_reest_every_k", [](const filter_options& opts)
                     { return static_cast<int>(opts.multiplier_reest_every_k); })
        .def("__repr__", &describe_filter_options);
}

filter_options filter_policy_options(std::optional<int> policy_stall_window)
{
    filter_options opts;
    if(policy_stall_window)
        opts.stall_window =
            positive_count<std::uint16_t>(*policy_stall_window, "policy_stall_window");
    return opts;
}

void construct_filter_slsqp(filter_wrapper* self, nb::object objective, const vector<double>& x0,
                            nb::object constraints, nb::object constraint_jacobian,
                            nb::object gradient, std::optional<int> num_equality,
                            std::optional<int> num_inequality, nb::object lower_bounds,
                            nb::object upper_bounds, std::optional<int> policy_stall_window,
                            std::optional<int> max_iterations,
                            std::optional<double> feasibility_tolerance,
                            std::optional<double> constraint_tolerance,
                            std::optional<double> gradient_threshold,
                            std::optional<double> objective_threshold,
                            std::optional<double> step_threshold,
                            std::optional<double> stall_threshold, std::optional<int> stall_window)
{
    reject_bounds(lower_bounds, upper_bounds);
    constrained_callables calls{std::move(objective),    std::move(gradient),
                                std::move(constraints),  std::move(constraint_jacobian),
                                std::move(lower_bounds), std::move(upper_bounds)};
    check_constrained_inputs(x0, calls);
    const driver_options opts =
        configure({max_iterations, feasibility_tolerance, constraint_tolerance, gradient_threshold,
                   objective_threshold, step_threshold, stall_threshold, stall_window});
    const filter_options policy_opts = filter_policy_options(policy_stall_window);
    auto adapter = make_constrained_problem(
        x0, constrained_callback_set(std::move(calls), num_equality, num_inequality));
    new(self) filter_wrapper(std::move(adapter), x0, opts, policy_opts);
}

void bind_filter_solver(nb::module_& m)
{
    nb::class_<filter_wrapper> cls(m, "FilterSlsqpSolver",
                                   nb::type_slots(wrapper_slots<filter_wrapper>),
                                   filter_solver_doc);
    cls.def("__init__", &construct_filter_slsqp, nb::arg("objective"), nb::arg("x0"),
            nb::arg("constraints"), nb::arg("constraint_jacobian"),
            nb::arg("gradient") = nb::none(), nb::arg("num_equality") = nb::none(),
            nb::arg("num_inequality") = nb::none(), nb::arg("lower_bounds") = nb::none(),
            nb::arg("upper_bounds") = nb::none(), nb::arg("policy_stall_window") = nb::none(),
            nb::arg("max_iterations") = nb::none(), nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(), nb::arg("step_threshold") = nb::none(),
            nb::arg("stall_threshold") = nb::none(), nb::arg("stall_window") = nb::none(),
            filter_solver_doc);
    def_solver_methods(cls, "FilterSlsqpSolver");
}

}

void register_filter_slsqp(nb::module_& m)
{
    bind_restoration_strategy(m);
    bind_filter_options(m);
    bind_filter_solver(m);
}

}
