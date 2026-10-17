#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_REGISTRATIONS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_REGISTRATIONS_H

#include <nanobind/nanobind.h>

namespace argmin::python
{

void register_error_channel(nanobind::module_& m);
void register_status_enums(nanobind::module_& m);
void register_results(nanobind::module_& m);
void register_qp_options(nanobind::module_& m);
void register_qp_solvers(nanobind::module_& m);
void register_nlp_options(nanobind::module_& m);
void register_lbfgsb(nanobind::module_& m);
void register_kraft_slsqp(nanobind::module_& m);
void register_filter_slsqp(nanobind::module_& m);
void register_tr_sqp(nanobind::module_& m);
void register_bobyqa(nanobind::module_& m);
void register_cobyla(nanobind::module_& m);
void register_cmaes(nanobind::module_& m);
void register_rosenbrock(nanobind::module_& m);
void register_constrained_test_problem(nanobind::module_& m);
void register_experimental(nanobind::module_& m);
void register_experimental_native(nanobind::module_& gated);

}

#endif
