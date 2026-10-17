#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_SET_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_SET_H

#include <nanobind/nanobind.h>

namespace argmin::python
{

struct callback_set
{
    nanobind::object objective{};
    nanobind::object gradient{};
    nanobind::object lower_bounds{};
    nanobind::object upper_bounds{};
    nanobind::object constraints{};
    nanobind::object constraint_jacobian{};
    int num_equality{0};
    int num_inequality{0};
};

}

#endif
