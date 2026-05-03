#ifndef HPP_GUARD_ARGMIN_RESULT_SOLVE_RESULT_H
#define HPP_GUARD_ARGMIN_RESULT_SOLVE_RESULT_H

#include "argmin/types.h"
#include "argmin/result/status.h"

#include <chrono>
#include <cstdint>

namespace argmin
{

// Full convergence diagnostics returned by basic_solver::solve().
//
// Captures the final iterate, objective value, gradient norm, constraint
// violation, iteration counts, and wall-clock time.

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
    std::chrono::steady_clock::duration wall_time{};
};

}

#endif
