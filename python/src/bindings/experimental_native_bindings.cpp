#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/native_dispatch.h"
#include "bindings/detail/native_problems.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>

#include <optional>
#include <string_view>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

template <typename Problem>
vector<double> start_point(const Problem& problem, const std::optional<vector<double>>& start)
{
    vector<double> x0 = start ? *start : problem.initial_point();
    check_vector_length(x0, problem.dimension(), "x0");
    check_all_finite(x0, "x0");
    return x0;
}

result_type dispatch(const rosenbrock& problem, std::string_view method,
                     const vector<double>& x0, const solver_options<>& opts)
{
    if(method == "bobyqa")
        return run_native<bobyqa_policy<>>(problem, x0, opts);
    if(method == "cmaes")
        return run_native<cmaes_policy<>>(problem, x0, opts);
    reject_method(method, "'bobyqa', 'cmaes'");
}

template <typename Problem>
result_type solve_gated(const Problem& problem, std::string_view method,
                        const std::optional<vector<double>>& x0,
                        std::optional<int> max_iterations,
                        std::optional<double> feasibility_tolerance,
                        std::optional<double> constraint_tolerance,
                        std::optional<double> gradient_threshold,
                        std::optional<double> objective_threshold,
                        std::optional<double> step_threshold,
                        std::optional<double> stall_threshold, std::optional<int> stall_window)
{
    const solver_options<> opts =
        configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                   gradient_threshold, objective_threshold, step_threshold, stall_threshold,
                   stall_window});
    return dispatch(problem, method, start_point(problem, x0), opts);
}

template <typename Problem>
void def_solve_native(nb::module_& gated, const char* doc)
{
    gated.def("solve_native", &solve_gated<Problem>, nb::arg("problem"), nb::arg("method"),
              nb::arg("x0") = nb::none(), nb::arg("max_iterations") = nb::none(),
              nb::arg("feasibility_tolerance") = nb::none(),
              nb::arg("constraint_tolerance") = nb::none(),
              nb::arg("gradient_threshold") = nb::none(),
              nb::arg("objective_threshold") = nb::none(),
              nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
              nb::arg("stall_window") = nb::none(), doc);
}

}

void register_experimental_native(nb::module_& gated)
{
    def_solve_native<rosenbrock>(
        gated, "Runs 'bobyqa' or 'cmaes' against the compiled Rosenbrock valley with no "
               "interpreter callback in the loop.");
}

}
