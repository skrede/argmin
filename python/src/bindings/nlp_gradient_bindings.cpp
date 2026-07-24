#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/options/qp_options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/line_search/options.h"
#include "argmin/solver/kraft_slsqp_policy.h"

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
using slsqp_options = kraft_slsqp_policy<>::options_type;
using lbfgsb_wrapper = solver_wrapper<lbfgsb_policy<>, bounded_problem>;
using slsqp_wrapper = solver_wrapper<kraft_slsqp_policy<>, constrained_problem>;

template <typename Criterion>
const Criterion& criterion_of(const driver_options& opts)
{
    return std::get<Criterion>(opts.convergence.criteria);
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
    nb::class_<line_search_options>(m, "LineSearchOptions",
                                   "Read-only view of the line-search parameters a policy is "
                                   "configured with: the two Wolfe constants, the backtracking "
                                   "factor, the maximum step and the evaluation cap.")
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
    nb::class_<driver_options>(m, "SolverOptions",
                              "Read-only view of the driver configuration shared by every "
                              "bound nonlinear method: the iteration budget, the feasibility "
                              "tolerances and the convergence thresholds. Construct one to "
                              "read the library defaults; pass values as keyword arguments "
                              "to a solver to change them.")
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

void bind_qp_subproblem_options(nb::module_& m)
{
    nb::class_<qp_options>(m, "QpSubproblemOptions",
                          "Read-only view of the quadratic subproblem parameters carried by "
                          "the sequential quadratic programming policy.")
        .def(nb::init<>())
        .def_ro("max_iterations", &qp_options::max_iterations)
        .def_ro("tolerance", &qp_options::tolerance)
        .def("__repr__",
             [](const qp_options& opts)
             {
                 return "QpSubproblemOptions(max_iterations=" + format_number(opts.max_iterations)
                        + ", tolerance=" + format_number(opts.tolerance) + ")";
             });
}

void bind_slsqp_options(nb::module_& m)
{
    nb::class_<slsqp_options>(m, "SlsqpOptions",
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
                 return "SlsqpOptions(initial_penalty=" + format_number(opts.initial_penalty)
                        + ", line_search=" + describe_line_search(opts.line_search)
                        + ", stall_window=" + format_number(opts.stall_window) + ")";
             });
}

void bind_lbfgsb_solver(nb::module_& m)
{
    nb::class_<lbfgsb_wrapper>(
        m, "LbfgsbSolver", nb::type_slots(wrapper_slots<lbfgsb_wrapper>),
        "Bound-constrained limited-memory quasi-Newton method. Supply an objective and a "
        "starting point; a gradient is optional and falls back to the library's own "
        "finite differences. The solver owns the callables it was given.")
        .def(
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
                {
                    check_positive_dimension(*policy_stall_window, "policy_stall_window");
                    policy_opts.stall_window = static_cast<std::uint16_t>(*policy_stall_window);
                }

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

void bind_slsqp_solver(nb::module_& m)
{
    nb::class_<slsqp_wrapper>(
        m, "SlsqpSolver", nb::type_slots(wrapper_slots<slsqp_wrapper>),
        "Sequential least-squares quadratic programming for a constrained problem. Needs "
        "the constraint vector and its Jacobian, plus the equality and inequality counts.")
        .def(
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
            "driven to zero; inequalities follow and are held non-negative.")
        .def("solve", &slsqp_wrapper::solve)
        .def("step", &slsqp_wrapper::step)
        .def("step_n", &slsqp_wrapper::step_n, nb::arg("budget"))
        .def("reset", &slsqp_wrapper::reset, nb::arg("x0"))
        .def("reset_clear", &slsqp_wrapper::reset_clear, nb::arg("x0"))
        .def("abort", &slsqp_wrapper::abort)
        .def("status", &slsqp_wrapper::status)
        .def("options", &slsqp_wrapper::options)
        .def("policy_options", &slsqp_wrapper::policy_options)
        .def_prop_ro("x", &slsqp_wrapper::x)
        .def_prop_ro("n", &slsqp_wrapper::dimension)
        .def_prop_ro("gradient_norm", &slsqp_wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &slsqp_wrapper::constraint_violation)
        .def("__repr__",
             [](const slsqp_wrapper& solver)
             { return "SlsqpSolver(n=" + format_number(solver.dimension()) + ")"; });
}

}

void register_nlp_gradient(nb::module_& m)
{
    bind_line_search_options(m);
    bind_driver_options(m);
    bind_qp_subproblem_options(m);
    bind_lbfgsb_options(m);
    bind_slsqp_options(m);
    bind_lbfgsb_solver(m);
    bind_slsqp_solver(m);
}

}
