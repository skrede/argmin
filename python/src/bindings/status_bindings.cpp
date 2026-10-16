#include "bindings/registrations.h"

#include "argmin/qp/qp_types.h"
#include "argmin/result/status.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace argmin::python
{

void register_status_enums(nb::module_& m)
{
    nb::enum_<qp_solve_status>(m, "QpSolveStatus",
                              "How a quadratic-program solve ended. An iteration cap or an "
                              "infeasibility certificate is an answer, not an error, so it "
                              "arrives here rather than as a raised exception.")
        .value("solved", qp_solve_status::solved)
        .value("solved_inaccurate", qp_solve_status::solved_inaccurate)
        .value("max_iterations", qp_solve_status::max_iterations)
        .value("primal_infeasible", qp_solve_status::primal_infeasible)
        .value("dual_infeasible", qp_solve_status::dual_infeasible);

    nb::enum_<solver_status>(m, "SolverStatus",
                            "How a nonlinear solve ended, or that it is still running.")
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
        .value("invalid_problem", solver_status::invalid_problem)
        .value("line_search_failed", solver_status::line_search_failed);
}

}
