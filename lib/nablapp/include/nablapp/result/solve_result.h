#ifndef HPP_GUARD_NABLAPP_RESULT_SOLVE_RESULT_H
#define HPP_GUARD_NABLAPP_RESULT_SOLVE_RESULT_H

#include "nablapp/result/status.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdint>

namespace nablapp
{

// Full convergence diagnostics returned by basic_solver::solve().
//
// Captures the final iterate, objective value, gradient norm, constraint
// violation, iteration counts, and wall-clock time.

template <typename Scalar = double>
struct solve_result
{
    solver_status status{solver_status::running};
    std::uint32_t iterations{0};
    std::uint32_t function_evaluations{0};
    Scalar objective_value{};
    Scalar gradient_norm{};
    Scalar constraint_violation{};
    Eigen::VectorX<Scalar> x;
    std::chrono::steady_clock::duration wall_time{};
};

}

#endif
