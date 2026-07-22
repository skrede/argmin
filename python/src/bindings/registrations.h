#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_REGISTRATIONS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_REGISTRATIONS_H

#include <nanobind/nanobind.h>

namespace argmin::python
{

void register_vocabulary(nanobind::module_& m);
void register_qp(nanobind::module_& m);
void register_nlp_gradient(nanobind::module_& m);
void register_problem(nanobind::module_& m);
void register_solver(nanobind::module_& m);
void register_test_functions(nanobind::module_& m);

}

#endif
