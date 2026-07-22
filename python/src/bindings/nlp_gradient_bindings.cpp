#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/line_search/options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

#include <new>
#include <tuple>
#include <memory>
#include <string>
#include <cstdint>
#include <utility>
#include <optional>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using driver_options = solver_options<>;
using lbfgsb_options = lbfgsb_policy<>::options_type;
using lbfgsb_wrapper = solver_wrapper<lbfgsb_policy<>, bounded_problem>;

template <typename Criterion>
const Criterion& criterion_of(const driver_options& opts)
{
    return std::get<Criterion>(opts.convergence.criteria);
}

void require_callable(const nb::object& value, std::string_view name)
{
    if(value.is_valid() && !value.is_none() && PyCallable_Check(value.ptr()) != 0)
        return;
    raise_argmin_error(error_kind::invalid_callback, std::string(name) + " must be callable");
}

void require_optional_callable(const nb::object& value, std::string_view name)
{
    if(!value.is_valid() || value.is_none())
        return;
    require_callable(value, name);
}

std::string describe_line_search(const line_search_options& opts)
{
    return "LineSearchOptions(c1=" + format_number(opts.c1) + ", c2=" + format_number(opts.c2)
           + ", rho=" + format_number(opts.rho) + ", max_alpha=" + format_number(opts.max_alpha)
           + ", max_iterations=" + format_number(opts.max_iterations) + ")";
}

std::string describe_driver_options(const driver_options& opts)
{
    const auto& tolerance = opts.constraint_tolerance;
    return "SolverOptions(max_iterations=" + format_number(opts.max_iterations)
           + ", constraint_tolerance="
           + (tolerance ? format_number(*tolerance) : std::string("None"))
           + ", feasibility_tolerance=" + format_number(opts.feasibility_tolerance)
           + ", gradient_threshold="
           + format_number(criterion_of<gradient_tolerance_criterion>(opts).threshold)
           + ", objective_threshold="
           + format_number(criterion_of<objective_tolerance_criterion>(opts).threshold)
           + ", stationarity_threshold="
           + format_number(criterion_of<objective_tolerance_criterion>(opts).stationarity_threshold)
           + ", step_threshold="
           + format_number(criterion_of<step_tolerance_criterion>(opts).threshold) + ")";
}

void bind_line_search_options(nb::module_& m)
{
    nb::class_<line_search_options>(m, "LineSearchOptions")
        .def(nb::init<>())
        .def_ro("c1", &line_search_options::c1)
        .def_ro("c2", &line_search_options::c2)
        .def_ro("rho", &line_search_options::rho)
        .def_ro("max_alpha", &line_search_options::max_alpha)
        .def_ro("max_iterations", &line_search_options::max_iterations)
        .def("__repr__", &describe_line_search);
}

void bind_driver_options(nb::module_& m)
{
    nb::class_<driver_options>(m, "SolverOptions")
        .def(nb::init<>())
        .def_ro("max_iterations", &driver_options::max_iterations)
        .def_ro("constraint_tolerance", &driver_options::constraint_tolerance)
        .def_ro("feasibility_tolerance", &driver_options::feasibility_tolerance)
        .def_prop_ro("gradient_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<gradient_tolerance_criterion>(opts).threshold; })
        .def_prop_ro("objective_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<objective_tolerance_criterion>(opts).threshold; })
        .def_prop_ro("stationarity_threshold",
                     [](const driver_options& opts) {
                         return criterion_of<objective_tolerance_criterion>(opts)
                             .stationarity_threshold;
                     })
        .def_prop_ro("step_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<step_tolerance_criterion>(opts).threshold; })
        .def_prop_ro("stall_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<stall_tolerance_criterion>(opts).threshold; })
        .def_prop_ro("stall_window",
                     [](const driver_options& opts)
                     { return criterion_of<stall_tolerance_criterion>(opts).window; })
        .def("__repr__", &describe_driver_options);
}

void bind_lbfgsb_options(nb::module_& m)
{
    nb::enum_<lbfgsb_line_search>(m, "LbfgsbLineSearch")
        .value("strong_wolfe", lbfgsb_line_search::strong_wolfe)
        .value("armijo", lbfgsb_line_search::armijo);

    nb::class_<lbfgsb_options>(m, "LbfgsbOptions")
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

driver_options configure(std::optional<int> max_iterations,
                         std::optional<double> feasibility_tolerance,
                         std::optional<double> constraint_tolerance,
                         std::optional<double> gradient_threshold,
                         std::optional<double> objective_threshold,
                         std::optional<double> step_threshold,
                         std::optional<double> stall_threshold,
                         std::optional<int> stall_window)
{
    driver_options opts;
    if(max_iterations)
    {
        check_positive_dimension(*max_iterations, "max_iterations");
        opts.max_iterations = static_cast<std::uint32_t>(*max_iterations);
    }
    if(feasibility_tolerance)
    {
        check_non_negative(*feasibility_tolerance, "feasibility_tolerance");
        opts.feasibility_tolerance = *feasibility_tolerance;
    }
    if(constraint_tolerance)
    {
        check_non_negative(*constraint_tolerance, "constraint_tolerance");
        opts.constraint_tolerance = *constraint_tolerance;
    }
    if(gradient_threshold)
    {
        check_non_negative(*gradient_threshold, "gradient_threshold");
        opts.set_gradient_threshold(*gradient_threshold);
    }
    if(objective_threshold)
    {
        check_non_negative(*objective_threshold, "objective_threshold");
        opts.set_objective_threshold(*objective_threshold);
    }
    if(step_threshold)
    {
        check_non_negative(*step_threshold, "step_threshold");
        opts.set_step_threshold(*step_threshold);
    }
    if(stall_threshold)
    {
        check_non_negative(*stall_threshold, "stall_threshold");
        opts.set_stall_threshold(*stall_threshold);
    }
    if(stall_window)
    {
        check_positive_dimension(*stall_window, "stall_window");
        opts.set_stall_window(static_cast<std::uint16_t>(*stall_window));
    }
    return opts;
}

void bind_lbfgsb_solver(nb::module_& m)
{
    nb::class_<lbfgsb_wrapper>(m, "LbfgsbSolver", nb::type_slots(wrapper_slots<lbfgsb_wrapper>))
        .def(
            "__init__",
            [](lbfgsb_wrapper* self, nb::object objective, const vector<double>& x0,
               nb::object gradient, nb::object lower, nb::object upper,
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
                    configure(max_iterations, feasibility_tolerance, constraint_tolerance,
                              gradient_threshold, objective_threshold, step_threshold,
                              stall_threshold, stall_window);

                callback_set calls;
                calls.objective = std::move(objective);
                calls.gradient = std::move(gradient);
                calls.lower_bounds = std::move(lower);
                calls.upper_bounds = std::move(upper);

                auto adapter = std::make_unique<bounded_problem>(static_cast<int>(x0.size()),
                                                                 std::move(calls));
                new(self) lbfgsb_wrapper(std::move(adapter), x0, opts, lbfgsb_options{});
            },
            nb::arg("objective"), nb::arg("x0"), nb::arg("gradient") = nb::none(),
            nb::arg("lower") = nb::none(), nb::arg("upper") = nb::none(),
            nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none())
        .def("solve", &lbfgsb_wrapper::solve)
        .def("step", &lbfgsb_wrapper::step)
        .def("step_n", &lbfgsb_wrapper::step_n, nb::arg("budget"))
        .def("reset", &lbfgsb_wrapper::reset, nb::arg("x0"))
        .def("reset_clear", &lbfgsb_wrapper::reset_clear, nb::arg("x0"))
        .def("abort", &lbfgsb_wrapper::abort)
        .def("status", &lbfgsb_wrapper::status)
        .def("options", &lbfgsb_wrapper::options)
        .def("policy_options", &lbfgsb_wrapper::policy_options)
        .def_prop_ro("x", &lbfgsb_wrapper::x)
        .def_prop_ro("n", &lbfgsb_wrapper::dimension)
        .def_prop_ro("gradient_norm", &lbfgsb_wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &lbfgsb_wrapper::constraint_violation)
        .def("__repr__",
             [](const lbfgsb_wrapper& solver)
             {
                 return "LbfgsbSolver(n=" + format_number(solver.dimension()) + ")";
             });
}

}

void register_nlp_gradient(nb::module_& m)
{
    bind_line_search_options(m);
    bind_driver_options(m);
    bind_lbfgsb_options(m);
    bind_lbfgsb_solver(m);
}

}
