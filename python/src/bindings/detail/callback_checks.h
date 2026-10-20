#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_CHECKS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_CALLBACK_CHECKS_H

#include "bindings/detail/errors.h"
#include "bindings/detail/format_number.h"

#include "argmin/types.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Core>

#include <string>
#include <string_view>

namespace argmin::python
{

inline nanobind::object copy_out(const vector<double>& x)
{
    return nanobind::cast(x, nanobind::rv_policy::copy);
}

[[noreturn]] inline void reject(const std::string& message)
{
    raise_argmin_error(error_kind::invalid_callback, message);
}

inline vector<double> as_vector(const nanobind::object& produced, std::string_view name)
{
    vector<double> values;
    if(!nanobind::try_cast<vector<double>>(produced, values))
        reject(std::string(name) + " did not convert to a real vector");
    return values;
}

inline void require_length(int actual, int expected, std::string_view name)
{
    if(actual == expected)
        return;
    reject(std::string(name) + " has length " + format_number(actual) + ", expected "
           + format_number(expected));
}

template <typename Derived>
inline void require_finite(const Eigen::DenseBase<Derived>& values, std::string_view name)
{
    if(values.allFinite())
        return;
    reject(std::string(name) + " has a non-finite entry");
}

inline void require_no_nan(const vector<double>& values, std::string_view name)
{
    if(!values.hasNaN())
        return;
    reject(std::string(name) + " has an entry that is not a number");
}

}

#endif
