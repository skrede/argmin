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
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/options/trust_region_options.h"

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

using bobyqa_options = bobyqa_policy<>::options_type;
using bobyqa_wrapper = solver_wrapper<bobyqa_policy<>, bounded_problem>;

void bind_trust_region_options(nb::module_& m)
{
    nb::class_<trust_region_options<>>(m, "TrustRegionOptions",
                                    "Read-only view of the trust-region acceptance and radius "
                                    "update parameters.")
        .def(nb::init<>())
        .def_ro("eta_good", &trust_region_options<>::eta_good)
        .def_ro("eta_poor", &trust_region_options<>::eta_poor)
        .def_ro("expand_factor", &trust_region_options<>::expand_factor)
        .def_ro("shrink_factor", &trust_region_options<>::shrink_factor)
        .def_ro("step_threshold", &trust_region_options<>::step_threshold)
        .def_ro("geometry_factor", &trust_region_options<>::geometry_factor)
        .def("__repr__",
             [](const trust_region_options<>& opts)
             {
                 return "TrustRegionOptions(eta_good=" + format_number(opts.eta_good)
                        + ", eta_poor=" + format_number(opts.eta_poor)
                        + ", expand_factor=" + format_number(opts.expand_factor)
                        + ", shrink_factor=" + format_number(opts.shrink_factor)
                        + ", step_threshold=" + format_number(opts.step_threshold)
                        + ", geometry_factor=" + format_number(opts.geometry_factor) + ")";
             });
}

void bind_bobyqa_options(nb::module_& m)
{
    nb::class_<bobyqa_options>(m, "BobyqaOptions",
                              "Read-only view of the bound-constrained interpolation "
                              "policy's own configuration.")
        .def(nb::init<>())
        .def_ro("initial_trust_radius", &bobyqa_options::initial_trust_radius)
        .def_ro("final_trust_radius", &bobyqa_options::final_trust_radius)
        .def_ro("trust", &bobyqa_options::trust)
        .def_ro("stall_window", &bobyqa_options::stall_window)
        .def_ro("feasibility_gate", &bobyqa_options::feasibility_gate)
        .def("__repr__",
             [](const bobyqa_options& opts)
             {
                 return "BobyqaOptions(initial_trust_radius="
                        + describe_optional(opts.initial_trust_radius) + ", final_trust_radius="
                        + describe_optional(opts.final_trust_radius)
                        + ", stall_window=" + format_number(opts.stall_window)
                        + ", feasibility_gate=" + format_number(opts.feasibility_gate) + ")";
             });
}

void bind_bobyqa_solver(nb::module_& m)
{
    nb::class_<bobyqa_wrapper> cls(
        m, "BobyqaSolver", nb::type_slots(wrapper_slots<bobyqa_wrapper>),
        "Bound-constrained derivative-free trust-region interpolation method. Takes an "
        "objective only; no gradient is used or requested.");
    cls.def(
        "__init__",
        [](bobyqa_wrapper* self, nb::object objective, const vector<double>& x0,
           nb::object lower_bounds, nb::object upper_bounds,
           std::optional<double> initial_trust_radius,
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

            const solver_options<> opts =
                configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                           gradient_threshold, objective_threshold, step_threshold,
                           stall_threshold, stall_window});

            bobyqa_options policy_opts;
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
            calls.lower_bounds = std::move(lower_bounds);
            calls.upper_bounds = std::move(upper_bounds);

            auto adapter = std::make_unique<bounded_problem>(static_cast<int>(x0.size()),
                                                             std::move(calls));
            new(self) bobyqa_wrapper(std::move(adapter), x0, opts, policy_opts);
        },
        nb::arg("objective"), nb::arg("x0"), nb::arg("lower_bounds") = nb::none(),
        nb::arg("upper_bounds") = nb::none(), nb::arg("initial_trust_radius") = nb::none(),
        nb::arg("final_trust_radius") = nb::none(),
        nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
        nb::arg("feasibility_tolerance") = nb::none(),
        nb::arg("constraint_tolerance") = nb::none(),
        nb::arg("gradient_threshold") = nb::none(),
        nb::arg("objective_threshold") = nb::none(),
        nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
        nb::arg("stall_window") = nb::none(),
        "Powell's bound-constrained trust-region method over a quadratic interpolation "
        "model, requiring objective values only (Powell 2009, DAMTP 2009/NA06).");
    def_solver_methods(cls, "BobyqaSolver");
}

}

void register_bobyqa(nb::module_& m)
{
    bind_trust_region_options(m);
    bind_bobyqa_options(m);
    bind_bobyqa_solver(m);
}

}
