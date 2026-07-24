#ifndef HPP_GUARD_ARGMIN_QP_QP_TYPES_H
#define HPP_GUARD_ARGMIN_QP_QP_TYPES_H

// Public vocabulary for the dense operator-splitting QP solver.
//
// Problem form: min 0.5 x^T P x + q^T x  s.t.  l <= A x <= u.
//
// Outcomes are reported as result values (exception-free posture, mirroring
// result/status.h); argument and precondition violations travel on the
// expected<> error channel as qp_error.
//
// Reference: Stellato, Banjac, Goulart, Bemporad, Boyd (2020), "OSQP: An
//            operator splitting solver for quadratic programs."

#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

enum class qp_solve_status : std::uint8_t
{
    solved,
    solved_inaccurate,
    max_iterations,
    primal_infeasible,
    dual_infeasible
};

enum class qp_error : std::uint8_t
{
    dimension_mismatch,
    invalid_bounds,
    non_finite_input,
    capacity_exceeded,
    infeasible_start,
    invalid_problem
};

template <typename Scalar = double, int N = argmin::dynamic_dimension>
struct qp_result
{
    vector<Scalar, N> x;
    vector<Scalar> y;
    qp_solve_status status{qp_solve_status::solved};
    int iterations{0};
    bool polished{false};
    Scalar primal_residual{0};
    Scalar dual_residual{0};
    Scalar objective_value{0};
};

}

#endif
