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
#include "argmin/solver/tr_sqp_policy.h"

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

using tr_sqp_options = tr_sqp_policy<>::options_type;
using tr_sqp_wrapper = solver_wrapper<tr_sqp_policy<>, constrained_problem>;

constexpr const char* tr_sqp_solver_doc =
    "Trust-region sequential quadratic programming for a constrained problem, in its "
    "accurate mode: the Byrd-Omojokun composite step of Nocedal and Wright, Numerical "
    "Optimization 2e, Section 18.5, Algorithm 18.4, which splits each step into a normal "
    "step toward feasibility and a tangential step that reduces the objective, and accepts "
    "it by the ratio of actual to predicted reduction instead of a line search. Needs the "
    "constraint vector and its Jacobian, plus the equality and inequality counts. Equalities "
    "come first in the constraint vector and are driven to zero; inequalities follow and "
    "are held non-negative. The lower_bounds and upper_bounds keywords are rejected, "
    "because this solver does not support bounds yet.";

std::string describe_tr_sqp_options(const tr_sqp_options& opts)
{
    return "TrSqpOptions(initial_trust_radius=" + format_number(opts.initial_trust_radius)
           + ", min_trust_radius=" + format_number(opts.min_trust_radius)
           + ", max_cg_iterations_multiplier=" + format_number(opts.max_cg_iterations_multiplier)
           + ", stall_window=" + format_number(opts.stall_window)
           + ", multiplier_reest_every_k=" + format_number(opts.multiplier_reest_every_k)
           + ", initial_penalty=" + format_number(opts.initial_penalty)
           + ", penalty_factor=" + format_number(opts.penalty_factor)
           + ", soc_max_iterations=" + format_number(opts.soc_max_iterations) + ")";
}

void bind_tr_sqp_options(nb::module_& m)
{
    nb::class_<tr_sqp_options>(m, "TrSqpOptions",
                               "Read-only view of the trust-region sequential quadratic "
                               "programming policy's own configuration, in its accurate mode.")
        .def(nb::init<>())
        .def_ro("initial_trust_radius", &tr_sqp_options::initial_trust_radius)
        .def_ro("min_trust_radius", &tr_sqp_options::min_trust_radius)
        .def_ro("stall_window", &tr_sqp_options::stall_window)
        .def_ro("initial_penalty", &tr_sqp_options::initial_penalty)
        .def_ro("penalty_factor", &tr_sqp_options::penalty_factor)
        .def_prop_ro("max_cg_iterations_multiplier", [](const tr_sqp_options& opts)
                     { return static_cast<int>(opts.max_cg_iterations_multiplier); })
        .def_prop_ro("multiplier_reest_every_k", [](const tr_sqp_options& opts)
                     { return static_cast<int>(opts.multiplier_reest_every_k); })
        .def_prop_ro("soc_max_iterations", [](const tr_sqp_options& opts)
                     { return static_cast<int>(opts.soc_max_iterations); })
        .def("__repr__", &describe_tr_sqp_options);
}

tr_sqp_options tr_sqp_policy_options(std::optional<double> initial_trust_radius,
                                     std::optional<double> initial_penalty,
                                     std::optional<int> policy_stall_window)
{
    tr_sqp_options opts;
    if(initial_trust_radius)
        opts.initial_trust_radius = positive_option(*initial_trust_radius, "initial_trust_radius");
    if(initial_penalty)
        opts.initial_penalty = positive_option(*initial_penalty, "initial_penalty");
    if(policy_stall_window)
    {
        check_positive_dimension(*policy_stall_window, "policy_stall_window");
        opts.stall_window = static_cast<std::uint16_t>(*policy_stall_window);
    }
    return opts;
}

void construct_tr_sqp(tr_sqp_wrapper* self, nb::object objective, const vector<double>& x0,
                      nb::object constraints, nb::object constraint_jacobian, nb::object gradient,
                      std::optional<int> num_equality, std::optional<int> num_inequality,
                      nb::object lower_bounds, nb::object upper_bounds,
                      std::optional<double> initial_trust_radius,
                      std::optional<double> initial_penalty,
                      std::optional<int> policy_stall_window, std::optional<int> max_iterations,
                      std::optional<double> feasibility_tolerance,
                      std::optional<double> constraint_tolerance,
                      std::optional<double> gradient_threshold,
                      std::optional<double> objective_threshold,
                      std::optional<double> step_threshold, std::optional<double> stall_threshold,
                      std::optional<int> stall_window)
{
    reject_bounds(lower_bounds, upper_bounds);
    constrained_callables calls{std::move(objective),    std::move(gradient),
                                std::move(constraints),  std::move(constraint_jacobian),
                                std::move(lower_bounds), std::move(upper_bounds)};
    check_constrained_inputs(x0, calls);
    const driver_options opts =
        configure({max_iterations, feasibility_tolerance, constraint_tolerance, gradient_threshold,
                   objective_threshold, step_threshold, stall_threshold, stall_window});
    const tr_sqp_options policy_opts =
        tr_sqp_policy_options(initial_trust_radius, initial_penalty, policy_stall_window);
    auto adapter = make_constrained_problem(
        x0, constrained_callback_set(std::move(calls), num_equality, num_inequality));
    new(self) tr_sqp_wrapper(std::move(adapter), x0, opts, policy_opts);
}

void bind_tr_sqp_solver(nb::module_& m)
{
    nb::class_<tr_sqp_wrapper> cls(m, "TrSqpSolver", nb::type_slots(wrapper_slots<tr_sqp_wrapper>),
                                   tr_sqp_solver_doc);
    cls.def("__init__", &construct_tr_sqp, nb::arg("objective"), nb::arg("x0"),
            nb::arg("constraints"), nb::arg("constraint_jacobian"),
            nb::arg("gradient") = nb::none(), nb::arg("num_equality") = nb::none(),
            nb::arg("num_inequality") = nb::none(), nb::arg("lower_bounds") = nb::none(),
            nb::arg("upper_bounds") = nb::none(), nb::arg("initial_trust_radius") = nb::none(),
            nb::arg("initial_penalty") = nb::none(), nb::arg("policy_stall_window") = nb::none(),
            nb::arg("max_iterations") = nb::none(), nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(), nb::arg("step_threshold") = nb::none(),
            nb::arg("stall_threshold") = nb::none(), nb::arg("stall_window") = nb::none(),
            tr_sqp_solver_doc);
    def_solver_methods(cls, "TrSqpSolver");
}

}

void register_tr_sqp(nb::module_& m)
{
    bind_tr_sqp_options(m);
    bind_tr_sqp_solver(m);
}

}
