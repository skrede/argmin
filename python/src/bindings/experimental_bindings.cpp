#include "bindings/registrations.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace argmin::python
{

void register_experimental(nb::module_& m)
{
    nb::module_ gated = m.def_submodule(
        "experimental",
        "Solvers reachable only by a deliberate import, each carrying a documented known "
        "defect in its docstring and in its known_defect attribute.");
    register_bobyqa(gated);
    register_cobyla(gated);
    register_cmaes(gated);
    register_experimental_native(gated);
}

}
