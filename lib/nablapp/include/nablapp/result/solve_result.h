#ifndef HPP_GUARD_NABLAPP_RESULT_SOLVE_RESULT_H
#define HPP_GUARD_NABLAPP_RESULT_SOLVE_RESULT_H

#include "nablapp/types.h"
#include "nablapp/result/status.h"

#include <chrono>

namespace nablapp
{

// Full convergence diagnostics returned by basic_solver::solve() (per CORE-05).
//
// Captures the final iterate, objective value, gradient norm, constraint
// violation, iteration counts, and wall-clock time.

template <typename Scalar = double, int N = dynamic_dimension>
struct solve_result
{
    solver_status status{solver_status::running};
    int iterations{0};
    int function_evaluations{0};
    Scalar objective_value{};
    Scalar gradient_norm{};
    Scalar constraint_violation{};
    vector<Scalar, N> x;
    std::chrono::steady_clock::duration wall_time{};
};

}

#endif
