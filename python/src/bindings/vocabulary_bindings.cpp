#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
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
#include <exception>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using qp_result_type = qp_result<double>;
using solve_result_type = solve_result<double>;

const char* name_of(qp_solve_status status)
{
    switch(status)
    {
        case qp_solve_status::solved: return "solved";
        case qp_solve_status::solved_inaccurate: return "solved_inaccurate";
        case qp_solve_status::max_iterations: return "max_iterations";
        case qp_solve_status::primal_infeasible: return "primal_infeasible";
        case qp_solve_status::dual_infeasible: return "dual_infeasible";
    }
    return "unknown";
}

const char* name_of(solver_status status)
{
    switch(status)
    {
        case solver_status::running: return "running";
        case solver_status::converged: return "converged";
        case solver_status::max_iterations: return "max_iterations";
        case solver_status::budget_exhausted: return "budget_exhausted";
        case solver_status::stalled: return "stalled";
        case solver_status::diverged: return "diverged";
        case solver_status::xtol_reached: return "xtol_reached";
        case solver_status::ftol_reached: return "ftol_reached";
        case solver_status::maxeval_reached: return "maxeval_reached";
        case solver_status::roundoff_limited: return "roundoff_limited";
        case solver_status::trust_region_step_rejected: return "trust_region_step_rejected";
        case solver_status::objective_stalled: return "objective_stalled";
        case solver_status::time_limit_reached: return "time_limit_reached";
        case solver_status::aborted: return "aborted";
        case solver_status::invalid_problem: return "invalid_problem";
    }
    return "unknown";
}

// Held for the process lifetime rather than in a destructible object: a static
// handle released during interpreter teardown would drop a reference after the
// runtime that owns it is gone.
PyObject* python_error_type = nullptr;

void translate_argmin_error(const std::exception_ptr& raised, void*)
{
    try
    {
        std::rethrow_exception(raised);
    }
    catch(const argmin_error& error)
    {
        nb::object instance = nb::borrow<nb::object>(python_error_type)(error.what());
        instance.attr("kind") = error.kind();
        PyErr_SetObject(python_error_type, instance.ptr());
    }
}

void register_error_channel(nb::module_& m)
{
    nb::enum_<error_kind>(m, "ErrorKind")
        .value("dimension_mismatch", error_kind::dimension_mismatch)
        .value("invalid_bounds", error_kind::invalid_bounds)
        .value("non_finite_input", error_kind::non_finite_input)
        .value("capacity_exceeded", error_kind::capacity_exceeded)
        .value("infeasible_start", error_kind::infeasible_start)
        .value("invalid_problem", error_kind::invalid_problem)
        .value("invalid_array", error_kind::invalid_array)
        .value("invalid_callback", error_kind::invalid_callback)
        .value("invalid_state", error_kind::invalid_state);

    if(python_error_type == nullptr)
        python_error_type = PyErr_NewException("argmin.ArgminError", PyExc_RuntimeError, nullptr);
    m.attr("ArgminError") = nb::borrow<nb::object>(python_error_type);

    nb::register_exception_translator(translate_argmin_error, nullptr);
}

void register_status_enums(nb::module_& m)
{
    nb::enum_<qp_solve_status>(m, "QpStatus")
        .value("solved", qp_solve_status::solved)
        .value("solved_inaccurate", qp_solve_status::solved_inaccurate)
        .value("max_iterations", qp_solve_status::max_iterations)
        .value("primal_infeasible", qp_solve_status::primal_infeasible)
        .value("dual_infeasible", qp_solve_status::dual_infeasible);

    nb::enum_<solver_status>(m, "SolverStatus")
        .value("running", solver_status::running)
        .value("converged", solver_status::converged)
        .value("max_iterations", solver_status::max_iterations)
        .value("budget_exhausted", solver_status::budget_exhausted)
        .value("stalled", solver_status::stalled)
        .value("diverged", solver_status::diverged)
        .value("xtol_reached", solver_status::xtol_reached)
        .value("ftol_reached", solver_status::ftol_reached)
        .value("maxeval_reached", solver_status::maxeval_reached)
        .value("roundoff_limited", solver_status::roundoff_limited)
        .value("trust_region_step_rejected", solver_status::trust_region_step_rejected)
        .value("objective_stalled", solver_status::objective_stalled)
        .value("time_limit_reached", solver_status::time_limit_reached)
        .value("aborted", solver_status::aborted)
        .value("invalid_problem", solver_status::invalid_problem);
}

void register_qp_result(nb::module_& m)
{
    nb::class_<qp_result_type>(m, "QpResult")
        .def(
            "__init__",
            [](qp_result_type* self, const vector<double>& x, const vector<double>& y,
               qp_solve_status status, int iterations, bool polished, double primal_residual,
               double dual_residual, double objective_value)
            {
                check_all_finite(x, "x");
                check_all_finite(y, "y");
                check_non_negative(static_cast<double>(iterations), "iterations");
                check_non_negative(primal_residual, "primal_residual");
                check_non_negative(dual_residual, "dual_residual");
                check_finite(objective_value, "objective_value");
                new(self) qp_result_type{x,          y,
                                         status,     iterations,
                                         polished,   primal_residual,
                                         dual_residual, objective_value};
            },
            nb::arg("x") = vector<double>(), nb::arg("y") = vector<double>(),
            nb::arg("status") = qp_solve_status::solved, nb::arg("iterations") = 0,
            nb::arg("polished") = false, nb::arg("primal_residual") = 0.0,
            nb::arg("dual_residual") = 0.0, nb::arg("objective_value") = 0.0)
        .def_prop_ro("x", [](const qp_result_type& result) { return result.x; })
        .def_prop_ro("y", [](const qp_result_type& result) { return result.y; })
        .def_ro("status", &qp_result_type::status)
        .def_ro("iterations", &qp_result_type::iterations)
        .def_ro("polished", &qp_result_type::polished)
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
                        + ", x=" + format_length(static_cast<int>(result.x.size()))
                        + ", y=" + format_length(static_cast<int>(result.y.size())) + ")";
             });
}

void register_solve_result(nb::module_& m)
{
    nb::class_<solve_result_type>(m, "SolveResult")
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

void register_vocabulary(nb::module_& m)
{
    register_error_channel(m);
    register_status_enums(m);
    register_qp_result(m);
    register_solve_result(m);
}

}
