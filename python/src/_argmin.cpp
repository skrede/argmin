#include "bindings/registrations.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_argmin, m)
{
    m.doc() = "argmin C++20 optimization library — Python bindings";
    m.attr("__version__") = ARGMIN_PYTHON_VERSION_STRING;

    argmin::python::register_vocabulary(m);
    argmin::python::register_qp(m);
    argmin::python::register_nlp_gradient(m);
    argmin::python::register_nlp_derivative_free(m);
    argmin::python::register_test_functions(m);
}
