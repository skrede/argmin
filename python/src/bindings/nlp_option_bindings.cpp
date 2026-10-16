#include "bindings/registrations.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/describe_options.h"

#include "argmin/solver/options.h"
#include "argmin/options/qp_options.h"
#include "argmin/solver/convergence.h"
#include "argmin/line_search/options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

#include <string>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

void bind_line_search_options(nb::module_& m)
{
    nb::class_<line_search_options<>>(m, "LineSearchOptions",
                                   "Read-only view of the line-search parameters a policy is "
                                   "configured with: the two Wolfe constants, the backtracking "
                                   "factor, the maximum step and the evaluation cap.")
        .def(nb::init<>())
        .def_ro("c1", &line_search_options<>::c1)
        .def_ro("c2", &line_search_options<>::c2)
        .def_ro("rho", &line_search_options<>::rho)
        .def_ro("max_alpha", &line_search_options<>::max_alpha)
        .def_ro("max_iterations", &line_search_options<>::max_iterations)
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
                     { return criterion_of<gradient_tolerance_criterion<>>(opts).threshold; })
        .def_prop_ro("objective_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<objective_tolerance_criterion<>>(opts).threshold; })
        .def_prop_ro("stationarity_threshold",
                     [](const driver_options& opts) {
                         return criterion_of<objective_tolerance_criterion<>>(opts)
                             .stationarity_threshold;
                     })
        .def_prop_ro("step_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<step_tolerance_criterion<>>(opts).threshold; })
        .def_prop_ro("stall_threshold",
                     [](const driver_options& opts)
                     { return criterion_of<stall_tolerance_criterion<>>(opts).threshold; })
        .def_prop_ro("stall_window",
                     [](const driver_options& opts)
                     { return criterion_of<stall_tolerance_criterion<>>(opts).window; })
        .def("__repr__", &describe_driver_options);
}

void bind_qp_subproblem_options(nb::module_& m)
{
    nb::class_<qp_options<>>(m, "QpOptions",
                          "Read-only view of the quadratic subproblem parameters carried by "
                          "the sequential quadratic programming policy.")
        .def(nb::init<>())
        .def_ro("max_iterations", &qp_options<>::max_iterations)
        .def_ro("tolerance", &qp_options<>::tolerance)
        .def("__repr__",
             [](const qp_options<>& opts)
             {
                 return "QpOptions(max_iterations=" + format_number(opts.max_iterations)
                        + ", tolerance=" + format_number(opts.tolerance) + ")";
             });
}

}

void register_nlp_options(nb::module_& m)
{
    bind_line_search_options(m);
    bind_driver_options(m);
    bind_qp_subproblem_options(m);
}

}
