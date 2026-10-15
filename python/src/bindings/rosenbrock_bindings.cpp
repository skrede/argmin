#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/native_dispatch.h"
#include "bindings/detail/native_problems.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/lbfgsb_policy.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>

#include <new>
#include <string>
#include <utility>
#include <optional>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

result_type solve_rosenbrock_native(const native_rosenbrock& problem, std::string_view method,
                                    const std::optional<vector<double>>& start,
                                    const solver_options<>& opts)
{
    const vector<double> x0 = start ? *start : problem.initial_point();
    check_vector_length(x0, problem.dimension(), "x0");
    check_all_finite(x0, "x0");

    if(method == "lbfgsb")
        return run_native<lbfgsb_policy<>>(problem, x0, opts);
    if(method == "bobyqa")
        return run_native<bobyqa_policy<>>(problem, x0, opts);
    if(method == "cmaes")
        return run_native<cmaes_policy<>>(problem, x0, opts);
    reject_method(method, "'lbfgsb', 'bobyqa', 'cmaes'");
}

void bind_rosenbrock(nb::module_& m)
{
    nb::class_<native_rosenbrock>(
        m, "Rosenbrock",
        "The Rosenbrock valley compiled into the extension at any dimension of at least "
        "two, with an optional box. Its value and gradient are callable from the "
        "interpreter, and solve_native() runs a solve with no interpreter callback in the "
        "loop, which is what makes a parity comparison against the boundary possible.")
        .def(
            "__init__",
            [](native_rosenbrock* self, int n, const std::optional<vector<double>>& lower_bounds,
               const std::optional<vector<double>>& upper_bounds)
            {
                check_positive_dimension(n, "n");
                if(n < 2)
                    raise_argmin_error(error_kind::dimension_mismatch,
                                       "n must be at least 2, got " + format_number(n));
                native_rosenbrock problem;
                problem.shape.n = n;
                problem.lower = lower_bounds ? *lower_bounds
                                             : vector<double>::Constant(n, -unbounded);
                problem.upper = upper_bounds ? *upper_bounds
                                             : vector<double>::Constant(n, unbounded);
                check_vector_length(problem.lower, n, "lower_bounds");
                check_vector_length(problem.upper, n, "upper_bounds");
                check_no_nan(problem.lower, "lower_bounds");
                check_no_nan(problem.upper, "upper_bounds");
                check_ordered_bounds(problem.lower, problem.upper);
                new(self) native_rosenbrock(std::move(problem));
            },
            nb::arg("n") = 2, nb::arg("lower_bounds") = nb::none(),
            nb::arg("upper_bounds") = nb::none(),
            "The n-dimensional Rosenbrock valley compiled into the extension, with the global "
            "minimum at the all-ones point and an optimal value of zero.")
        .def("dimension", &native_rosenbrock::dimension)
        .def("optimal_value", &native_rosenbrock::optimal_value)
        .def("initial_point", &native_rosenbrock::initial_point)
        .def("lower_bounds", &native_rosenbrock::lower_bounds)
        .def("upper_bounds", &native_rosenbrock::upper_bounds)
        .def(
            "value",
            [](const native_rosenbrock& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                return problem.value(x);
            },
            nb::arg("x"))
        .def(
            "gradient",
            [](const native_rosenbrock& problem, const vector<double>& x)
            {
                check_vector_length(x, problem.dimension(), "x");
                vector<double> g;
                problem.gradient(x, g);
                return g;
            },
            nb::arg("x"))
        // The only independent oracle for the callback boundary is the same
        // objective solved without it: an interpreter-defined objective checked
        // against a hand-written expected value tests arithmetic, whereas
        // checking it against this entry point tests the boundary itself.
        .def(
            "solve_native",
            [](const native_rosenbrock& problem, std::string_view method,
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
                return solve_rosenbrock_native(problem, method, x0, opts);
            },
            nb::arg("method"), nb::arg("x0") = nb::none(),
            nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none())
        .def("__repr__",
             [](const native_rosenbrock& problem)
             { return "Rosenbrock(n=" + format_number(problem.dimension()) + ")"; });
}

}

void register_rosenbrock(nb::module_& m)
{
    bind_rosenbrock(m);
}

}
