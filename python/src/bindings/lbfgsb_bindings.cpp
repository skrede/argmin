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
#include "argmin/solver/lbfgsb_policy.h"

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

using lbfgsb_options = lbfgsb_policy<>::options_type;
using lbfgsb_wrapper = solver_wrapper<lbfgsb_policy<>, bounded_problem>;

void bind_lbfgsb_options(nb::module_& m)
{
    nb::enum_<lbfgsb_line_search>(m, "LbfgsbLineSearch",
                                 "Which line search the quasi-Newton method uses.")
        .value("strong_wolfe", lbfgsb_line_search::strong_wolfe)
        .value("armijo", lbfgsb_line_search::armijo);

    nb::class_<lbfgsb_options>(m, "LbfgsbOptions",
                              "Read-only view of the bound-constrained quasi-Newton policy's "
                              "own configuration.")
        .def(nb::init<>())
        .def_ro("line_search", &lbfgsb_options::line_search)
        .def_ro("line_search_type", &lbfgsb_options::line_search_type)
        .def_ro("stall_window", &lbfgsb_options::stall_window)
        .def("__repr__",
             [](const lbfgsb_options& opts)
             {
                 return "LbfgsbOptions(line_search=" + describe_line_search(opts.line_search)
                        + ", line_search_type="
                        + (opts.line_search_type == lbfgsb_line_search::armijo ? "armijo"
                                                                              : "strong_wolfe")
                        + ", stall_window=" + format_number(opts.stall_window) + ")";
             });
}

void bind_lbfgsb_solver(nb::module_& m)
{
    nb::class_<lbfgsb_wrapper> cls(
        m, "LbfgsbSolver", nb::type_slots(wrapper_slots<lbfgsb_wrapper>),
        "Bound-constrained limited-memory quasi-Newton method. Supply an objective and a "
        "starting point; a gradient is optional and falls back to the library's own "
        "finite differences. The solver owns the callables it was given.");
    cls.def(
        "__init__",
        [](lbfgsb_wrapper* self, nb::object objective, const vector<double>& x0,
           nb::object gradient, nb::object lower_bounds, nb::object upper_bounds,
           std::optional<int> policy_stall_window,
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

            const driver_options opts =
                configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                           gradient_threshold, objective_threshold, step_threshold,
                           stall_threshold, stall_window});

            callback_set calls;
            calls.objective = std::move(objective);
            calls.gradient = std::move(gradient);
            calls.lower_bounds = std::move(lower_bounds);
            calls.upper_bounds = std::move(upper_bounds);

            lbfgsb_options policy_opts;
            if(policy_stall_window)
                policy_opts.stall_window =
                    positive_count<std::uint16_t>(*policy_stall_window, "policy_stall_window");

            auto adapter = std::make_unique<bounded_problem>(static_cast<int>(x0.size()),
                                                             std::move(calls));
            new(self) lbfgsb_wrapper(std::move(adapter), x0, opts, policy_opts);
        },
        nb::arg("objective"), nb::arg("x0"), nb::arg("gradient") = nb::none(),
        nb::arg("lower_bounds") = nb::none(), nb::arg("upper_bounds") = nb::none(),
        nb::arg("policy_stall_window") = nb::none(),
        nb::arg("max_iterations") = nb::none(),
        nb::arg("feasibility_tolerance") = nb::none(),
        nb::arg("constraint_tolerance") = nb::none(),
        nb::arg("gradient_threshold") = nb::none(),
        nb::arg("objective_threshold") = nb::none(),
        nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
        nb::arg("stall_window") = nb::none());
    def_solver_methods(cls, "LbfgsbSolver");
}

}

void register_lbfgsb(nb::module_& m)
{
    bind_lbfgsb_options(m);
    bind_lbfgsb_solver(m);
}

}
