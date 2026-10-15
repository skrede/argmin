#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_NATIVE_DISPATCH_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_NATIVE_DISPATCH_H

#include "bindings/detail/errors.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/result/solve_result.h"
#include "argmin/solver/step_budget_solver.h"

#include <nanobind/nanobind.h>

#include <string>
#include <string_view>

namespace argmin::python
{

using result_type = solve_result<double>;

[[noreturn]] inline void reject_method(std::string_view method, std::string_view accepted)
{
    raise_argmin_error(error_kind::invalid_problem,
                       "unknown method '" + std::string(method) + "', expected one of "
                           + std::string(accepted));
}

template <typename Policy, typename Problem>
result_type run_native(const Problem& problem, const vector<double>& x0,
                       const solver_options<>& opts)
{
    step_budget_solver<Policy, dynamic_dimension, Problem> driver(problem, x0, opts);
    const nanobind::gil_scoped_release released;
    return driver.solve();
}

}

#endif
