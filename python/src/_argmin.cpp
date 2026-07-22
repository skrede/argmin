#include "bindings/registrations.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_argmin, m)
{
    m.doc() = "argmin C++20 optimization library — Python bindings";
    m.attr("__version__") = ARGMIN_PYTHON_VERSION_STRING;
}
