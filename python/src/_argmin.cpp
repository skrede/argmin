#include "bindings/registrations.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_argmin, m)
{
    m.doc() = "argmin C++20 optimization library — Python bindings";
    m.attr("__version__") = ARGMIN_PYTHON_VERSION_STRING;

    argmin::python::register_error_channel(m);
    argmin::python::register_status_enums(m);
    argmin::python::register_results(m);
    argmin::python::register_qp_options(m);
    argmin::python::register_qp_solvers(m);
    argmin::python::register_nlp_options(m);
    argmin::python::register_lbfgsb(m);
    argmin::python::register_kraft_slsqp(m);
    argmin::python::register_filter_slsqp(m);
    argmin::python::register_tr_sqp(m);
    argmin::python::register_rosenbrock(m);
    argmin::python::register_constrained_test_problem(m);
    argmin::python::register_experimental(m);
}
