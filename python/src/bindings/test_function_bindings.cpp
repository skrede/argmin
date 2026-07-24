#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/keyword_options.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/result/solve_result.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>

#include <new>
#include <string>
#include <limits>
#include <utility>
#include <optional>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using result_type = solve_result<double>;

constexpr double unbounded = std::numeric_limits<double>::infinity();

[[noreturn]] void reject_method(std::string_view method, std::string_view accepted)
{
    raise_argmin_error(error_kind::invalid_problem,
                       "unknown method '" + std::string(method) + "', expected one of "
                           + std::string(accepted));
}

// Compiled, dynamically dimensioned face of the library's own banana-shaped
// valley. The box is exposed so a caller can pose the identical bounded
// problem to the interpreter-side methods, which is the only way the parity
// comparison holds every input fixed but the boundary.
struct native_rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;

    rosenbrock<double> shape{};
    vector<double> lower{};
    vector<double> upper{};

    [[nodiscard]] int dimension() const { return shape.n; }
    [[nodiscard]] double optimal_value() const { return shape.optimal_value(); }
    [[nodiscard]] vector<double> initial_point() const { return shape.initial_point(); }
    [[nodiscard]] double value(const vector<double>& x) const { return shape.value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const { shape.gradient(x, g); }

    [[nodiscard]] vector<double> lower_bounds() const { return lower; }
    [[nodiscard]] vector<double> upper_bounds() const { return upper; }
};

// The equality-constrained problem is published at a fixed size; this face
// re-expresses it at dynamic dimension so a native result carries the same
// vector type the wrapped methods return.
struct native_constrained
{
    static constexpr int problem_dimension = dynamic_dimension;

    hs006<double> shape{};

    [[nodiscard]] int dimension() const { return shape.dimension(); }
    [[nodiscard]] int num_equality() const { return shape.num_equality(); }
    [[nodiscard]] int num_inequality() const { return shape.num_inequality(); }
    [[nodiscard]] double optimal_value() const { return shape.optimal_value(); }

    [[nodiscard]] vector<double> initial_point() const
    {
        return vector<double>(shape.initial_point());
    }

    [[nodiscard]] double value(const vector<double>& x) const
    {
        return shape.value(Eigen::Vector2d(x));
    }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        Eigen::Vector2d fixed = Eigen::Vector2d(x);
        Eigen::Vector2d produced;
        shape.gradient(fixed, produced);
        g = produced;
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        c.resize(1);
        shape.constraints(Eigen::Vector2d(x), c);
    }

    void constraint_jacobian(const vector<double>& x, matrix<double>& J) const
    {
        J.resize(1, 2);
        shape.constraint_jacobian(Eigen::Vector2d(x), J);
    }

    [[nodiscard]] vector<double> lower_bounds() const
    {
        return vector<double>::Constant(2, -unbounded);
    }

    [[nodiscard]] vector<double> upper_bounds() const
    {
        return vector<double>::Constant(2, unbounded);
    }
};

template <typename Policy, typename Problem>
result_type run_native(const Problem& problem, const vector<double>& x0,
                       const solver_options<>& opts)
{
    step_budget_solver<Policy, dynamic_dimension, Problem> driver(problem, x0, opts);
    const nb::gil_scoped_release released;
    return driver.solve();
}

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
        return run_native<cobyla_policy>(problem, x0, opts);
    reject_method(method, "'slsqp', 'cobyla'");
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

void register_test_functions(nb::module_& m)
{
    bind_rosenbrock(m);
    bind_constrained_problem(m);
}

}
