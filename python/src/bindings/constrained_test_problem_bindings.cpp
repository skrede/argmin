#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/native_dispatch.h"
#include "bindings/detail/native_problems.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>

#include <optional>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

result_type solve_constrained_native(const native_constrained& problem, std::string_view method,
                                     const std::optional<vector<double>>& start,
                                     const solver_options<>& opts)
{
    const vector<double> x0 = start ? *start : problem.initial_point();
    check_vector_length(x0, problem.dimension(), "x0");
    check_all_finite(x0, "x0");

    if(method == "slsqp")
        return run_native<kraft_slsqp_policy<>>(problem, x0, opts);
    if(method == "cobyla")
        return run_native<cobyla_policy<>>(problem, x0, opts);
    reject_method(method, "'slsqp', 'cobyla'");
}

void bind_constrained_problem(nb::module_& m)
{
    nb::class_<native_constrained>(
        m, "ConstrainedTestProblem",
        "A two-variable equality-constrained problem compiled into the extension, with the "
        "same callback-free solve_native() entry point as the Rosenbrock valley.")
        .def(nb::init<>(),
             "Hock and Schittkowski problem six: minimize (1 - x0)^2 subject to the equality "
             "10 (x1 - x0^2) = 0, with the solution at the all-ones point and an optimal value "
             "of zero.")
        .def("dimension", &native_constrained::dimension)
        .def("num_equality", &native_constrained::num_equality)
        .def("num_inequality", &native_constrained::num_inequality)
        .def("optimal_value", &native_constrained::optimal_value)
        .def("initial_point", &native_constrained::initial_point)
        .def(
            "value",
            [](const native_constrained& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                return problem.value(x);
            },
            nb::arg("x"))
        .def(
            "gradient",
            [](const native_constrained& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                vector<double> g;
                problem.gradient(x, g);
                return g;
            },
            nb::arg("x"))
        .def(
            "constraints",
            [](const native_constrained& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                vector<double> c;
                problem.constraints(x, c);
                return c;
            },
            nb::arg("x"))
        .def(
            "constraint_jacobian",
            [](const native_constrained& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                matrix<double> J;
                problem.constraint_jacobian(x, J);
                return J;
            },
            nb::arg("x"))
        .def(
            "solve_native",
            [](const native_constrained& problem, std::string_view method,
               const std::optional<vector<double>>& x0, std::optional<int> max_iterations,
               std::optional<double> feasibility_tolerance,
               std::optional<double> constraint_tolerance,
               std::optional<double> gradient_threshold,
               std::optional<double> objective_threshold, std::optional<double> step_threshold,
               std::optional<double> stall_threshold, std::optional<int> stall_window)
            {
                const solver_options<> opts =
                    configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                               gradient_threshold, objective_threshold, step_threshold,
                               stall_threshold, stall_window});
                return solve_constrained_native(problem, method, x0, opts);
            },
            nb::arg("method"), nb::arg("x0") = nb::none(),
            nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none())
        .def("__repr__", [](const native_constrained&) { return "ConstrainedTestProblem()"; });
}

}

void register_constrained_test_problem(nb::module_& m)
{
    bind_constrained_problem(m);
}

}
