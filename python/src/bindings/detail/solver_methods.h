#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SOLVER_METHODS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_SOLVER_METHODS_H

#include "bindings/detail/format_number.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include <string>

namespace argmin::python
{

template <typename Wrapper>
void def_solver_methods(nanobind::class_<Wrapper>& cls, const char* name)
{
    cls.def("solve", &Wrapper::solve)
        .def("step", &Wrapper::step)
        .def("step_n", &Wrapper::step_n, nanobind::arg("budget"))
        .def("reset", &Wrapper::reset, nanobind::arg("x0"))
        .def("reset_clear", &Wrapper::reset_clear, nanobind::arg("x0"))
        .def("abort", &Wrapper::abort)
        .def("status", &Wrapper::status)
        .def("options", &Wrapper::options)
        .def("policy_options", &Wrapper::policy_options)
        .def_prop_ro("x", &Wrapper::x)
        .def_prop_ro("n", &Wrapper::dimension)
        .def_prop_ro("gradient_norm", &Wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &Wrapper::constraint_violation)
        .def("__repr__",
             [name](const Wrapper& solver)
             { return std::string(name) + "(n=" + format_number(solver.dimension()) + ")"; });
}

}

#endif
