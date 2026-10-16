#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/callback_set.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/wrapper_slots.h"
#include "bindings/detail/solver_methods.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"

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

using cobyla_options = cobyla_policy<>::options_type;
using cobyla_wrapper = solver_wrapper<cobyla_policy<>, constrained_values_problem>;

constexpr const char* cobyla_known_defect =
    "COBYLA returns objective values of 84, 7516 and 7.7 against an optimum of 0 on the "
    "Hock-Schittkowski problems 48, 50 and 51, and its bound repair can evaluate the "
    "objective outside the box under the default initial trust radius.";

void bind_cobyla_options(nb::module_& m)
{
    nb::class_<cobyla_options>(m, "CobylaOptions",
                              "Read-only view of the constrained linear-approximation "
                              "policy's own configuration.")
        .def(nb::init<>())
        .def_ro("initial_trust_radius", &cobyla_options::initial_trust_radius)
        .def_ro("final_trust_radius", &cobyla_options::final_trust_radius)
        .def_ro("stall_window", &cobyla_options::stall_window)
        .def_ro("feasibility_gate", &cobyla_options::feasibility_gate)
        .def("__repr__",
             [](const cobyla_options& opts)
             {
                 return "CobylaOptions(initial_trust_radius="
                        + format_number(opts.initial_trust_radius)
                        + ", final_trust_radius=" + format_number(opts.final_trust_radius)
                        + ", stall_window=" + format_number(opts.stall_window)
                        + ", feasibility_gate=" + format_number(opts.feasibility_gate) + ")";
             });
}

void bind_cobyla_solver(nb::module_& m)
{
    nb::class_<cobyla_wrapper> cls(
        m, "CobylaSolver", nb::type_slots(wrapper_slots<cobyla_wrapper>),
        "Derivative-free constrained optimization by linear approximations. Takes an "
        "objective, a constraint vector and the two constraint counts; box bounds are "
        "rewritten into explicit constraints by the method itself. Experimental, with a "
        "known defect. COBYLA returns objective values of 84, 7516 and 7.7 against an "
        "optimum of 0 on the Hock-Schittkowski problems 48, 50 and 51, and its bound repair "
        "can evaluate the objective outside the box under the default initial trust radius.");
    cls.attr("maturity") = "experimental";
    cls.attr("known_defect") = cobyla_known_defect;
    cls.def(
        "__init__",
        [](cobyla_wrapper* self, nb::object objective, const vector<double>& x0,
           nb::object constraints, std::optional<int> num_equality,
           std::optional<int> num_inequality, nb::object lower_bounds,
           nb::object upper_bounds, std::optional<double> initial_trust_radius,
           std::optional<double> final_trust_radius, std::optional<int> policy_stall_window,
           std::optional<int> max_iterations, std::optional<double> feasibility_tolerance,
           std::optional<double> constraint_tolerance,
           std::optional<double> gradient_threshold,
           std::optional<double> objective_threshold, std::optional<double> step_threshold,
           std::optional<double> stall_threshold, std::optional<int> stall_window)
        {
            check_positive_dimension(static_cast<int>(x0.size()), "x0");
            check_all_finite(x0, "x0");
            require_callable(objective, "objective");
            require_callable(constraints, "constraints");

            const solver_options<> opts =
                configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                           gradient_threshold, objective_threshold, step_threshold,
                           stall_threshold, stall_window});

            cobyla_options policy_opts;
            if(initial_trust_radius)
                policy_opts.initial_trust_radius =
                    positive_option(*initial_trust_radius, "initial_trust_radius");
            if(final_trust_radius)
                policy_opts.final_trust_radius =
                    positive_option(*final_trust_radius, "final_trust_radius");
            if(policy_stall_window)
            {
                check_positive_dimension(*policy_stall_window, "policy_stall_window");
                policy_opts.stall_window =
                    static_cast<std::uint16_t>(*policy_stall_window);
            }

            callback_set calls;
            calls.objective = std::move(objective);
            calls.constraints = std::move(constraints);
            calls.lower_bounds = std::move(lower_bounds);
            calls.upper_bounds = std::move(upper_bounds);
            calls.num_equality = constraint_count(num_equality, "num_equality");
            calls.num_inequality = constraint_count(num_inequality, "num_inequality");
            check_positive_dimension(calls.num_equality + calls.num_inequality,
                                     "num_equality + num_inequality");

            auto adapter = std::make_unique<constrained_values_problem>(
                static_cast<int>(x0.size()), std::move(calls));
            new(self) cobyla_wrapper(std::move(adapter), x0, opts, policy_opts);
        },
        nb::arg("objective"), nb::arg("x0"), nb::arg("constraints"),
        nb::arg("num_equality") = nb::none(), nb::arg("num_inequality") = nb::none(),
        nb::arg("lower_bounds") = nb::none(), nb::arg("upper_bounds") = nb::none(),
        nb::arg("initial_trust_radius") = nb::none(),
        nb::arg("final_trust_radius") = nb::none(),
        nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
        nb::arg("feasibility_tolerance") = nb::none(),
        nb::arg("constraint_tolerance") = nb::none(),
        nb::arg("gradient_threshold") = nb::none(),
        nb::arg("objective_threshold") = nb::none(),
        nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
        nb::arg("stall_window") = nb::none(),
        "Powell's constrained trust-region method over linear interpolation models of the "
        "objective and every constraint, requiring no derivatives (Powell 1994). "
        "Equalities come first in the constraint vector and are driven to zero; "
        "inequalities follow and are held non-negative.");
    def_solver_methods(cls, "CobylaSolver");
}

}

void register_cobyla(nb::module_& m)
{
    bind_cobyla_options(m);
    bind_cobyla_solver(m);
}

}
