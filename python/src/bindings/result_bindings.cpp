#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/status_names.h"
#include "bindings/detail/format_number.h"

#include "argmin/qp/qp_types.h"
#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include <new>
#include <string>
#include <cstdint>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using qp_result_type = qp_result<double>;
using solve_result_type = solve_result<double>;

void register_qp_result(nb::module_& m)
{
    nb::class_<qp_result_type>(m, "QpResult",
                              "The outcome of a quadratic-program solve: the primal and dual "
                              "iterates, the terminal status, the residuals and the objective "
                              "value. The arrays own their data and outlive the solver that "
                              "produced them.")
        .def(
            "__init__",
            [](qp_result_type* self, const vector<double>& x, const vector<double>& y,
               qp_solve_status status, int iterations, bool polished, bool warm_started,
               double primal_residual, double dual_residual, double objective_value)
            {
                check_all_finite(x, "x");
                check_all_finite(y, "y");
                check_non_negative(static_cast<double>(iterations), "iterations");
                check_non_negative(primal_residual, "primal_residual");
                check_non_negative(dual_residual, "dual_residual");
                check_finite(objective_value, "objective_value");
                new(self) qp_result_type{x,
                                         y,
                                         status,
                                         iterations,
                                         polished,
                                         warm_started,
                                         primal_residual,
                                         dual_residual,
                                         objective_value};
            },
            nb::arg("x") = vector<double>(), nb::arg("y") = vector<double>(),
            nb::arg("status") = qp_solve_status::solved, nb::arg("iterations") = 0,
            nb::arg("polished") = false, nb::arg("warm_started") = false,
            nb::arg("primal_residual") = 0.0,
            nb::arg("dual_residual") = 0.0, nb::arg("objective_value") = 0.0)
        .def_prop_ro("x", [](const qp_result_type& result) { return result.x; })
        .def_prop_ro("y", [](const qp_result_type& result) { return result.y; })
        .def_ro("status", &qp_result_type::status)
        .def_ro("iterations", &qp_result_type::iterations)
        .def_ro("polished", &qp_result_type::polished)
        .def_ro("warm_started", &qp_result_type::warm_started,
                "True iff an operator-splitting solver consumed a primal seed (the warm_start() "
                "vector or the iterate retained from the previous solve). A dual seed whose length "
                "does not match the posed problem is replaced by zeros and does not clear the "
                "flag. The active-set solver always reports False.")
        .def_ro("primal_residual", &qp_result_type::primal_residual)
        .def_ro("dual_residual", &qp_result_type::dual_residual)
        .def_ro("objective_value", &qp_result_type::objective_value)
        .def("__repr__",
             [](const qp_result_type& result)
             {
                 return "QpResult(status=" + std::string(name_of(result.status))
                        + ", iterations=" + format_number(result.iterations)
                        + ", objective_value=" + format_number(result.objective_value)
                        + ", primal_residual=" + format_number(result.primal_residual)
                        + ", dual_residual=" + format_number(result.dual_residual)
                        + ", polished=" + (result.polished ? "True" : "False")
                        + ", warm_started=" + (result.warm_started ? "True" : "False")
                        + ", x=" + format_length(static_cast<int>(result.x.size()))
                        + ", y=" + format_length(static_cast<int>(result.y.size())) + ")";
             });
}

void register_solve_result(nb::module_& m)
{
    nb::class_<solve_result_type>(m, "SolveResult",
                                 "The outcome of a nonlinear solve: the iterate, the terminal "
                                 "status, the iteration and evaluation counts, the objective "
                                 "value, the gradient norm and the constraint violation.")
        .def(
            "__init__",
            [](solve_result_type* self, solver_status status, int iterations,
               int function_evaluations, double objective_value, double gradient_norm,
               double constraint_violation, const vector<double>& x)
            {
                check_non_negative(static_cast<double>(iterations), "iterations");
                check_non_negative(static_cast<double>(function_evaluations),
                                   "function_evaluations");
                check_finite(objective_value, "objective_value");
                check_non_negative(gradient_norm, "gradient_norm");
                check_non_negative(constraint_violation, "constraint_violation");
                check_all_finite(x, "x");
                new(self) solve_result_type{status,
                                            static_cast<std::uint32_t>(iterations),
                                            static_cast<std::uint32_t>(function_evaluations),
                                            objective_value,
                                            gradient_norm,
                                            constraint_violation,
                                            x};
            },
            nb::arg("status") = solver_status::running, nb::arg("iterations") = 0,
            nb::arg("function_evaluations") = 0, nb::arg("objective_value") = 0.0,
            nb::arg("gradient_norm") = 0.0, nb::arg("constraint_violation") = 0.0,
            nb::arg("x") = vector<double>())
        .def_prop_ro("x", [](const solve_result_type& result) { return result.x; })
        .def_ro("status", &solve_result_type::status)
        .def_ro("iterations", &solve_result_type::iterations)
        .def_ro("function_evaluations", &solve_result_type::function_evaluations)
        .def_ro("objective_value", &solve_result_type::objective_value)
        .def_ro("gradient_norm", &solve_result_type::gradient_norm)
        .def_ro("constraint_violation", &solve_result_type::constraint_violation)
        .def("__repr__",
             [](const solve_result_type& result)
             {
                 return "SolveResult(status=" + std::string(name_of(result.status))
                        + ", iterations=" + format_number(result.iterations)
                        + ", function_evaluations=" + format_number(result.function_evaluations)
                        + ", objective_value=" + format_number(result.objective_value)
                        + ", gradient_norm=" + format_number(result.gradient_norm)
                        + ", constraint_violation=" + format_number(result.constraint_violation)
                        + ", x=" + format_length(static_cast<int>(result.x.size())) + ")";
             });
}

}

void register_results(nb::module_& m)
{
    register_qp_result(m);
    register_solve_result(m);
}

}
