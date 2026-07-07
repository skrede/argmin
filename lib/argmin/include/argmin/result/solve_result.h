#ifndef HPP_GUARD_ARGMIN_RESULT_SOLVE_RESULT_H
#define HPP_GUARD_ARGMIN_RESULT_SOLVE_RESULT_H

#include "argmin/types.h"
#include "argmin/result/status.h"

#include <cstdint>

namespace argmin
{

// Chrono-free convergence diagnostics returned by the step-budget driver and
// the base of the time-budget drivers' result.
//
// Captures the final iterate, objective value, gradient norm, constraint
// violation, and iteration counts. Wall-clock time is deliberately absent:
// only the time-budget drivers report it, via timed_solve_result
// (argmin/result/timed_solve_result.h), which publicly derives from this
// type. Keeping the clock off the base result is what lets the step-budget
// path stay structurally wall-clock-free.

template <typename Scalar = double, int N = dynamic_dimension>
struct solve_result
{
    solver_status status{solver_status::running};
    std::uint32_t iterations{0};
    std::uint32_t function_evaluations{0};
    Scalar objective_value{};
    Scalar gradient_norm{};
    Scalar constraint_violation{};
    vector<Scalar, N> x;
};

}

#endif
