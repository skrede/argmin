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

#include <cmath>
#include <cstdint>
#include <algorithm>

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

// invalid_problem is problem data the method cannot handle, invalid_options a tuning value
// outside its domain, and factorization_failed a numerical decomposition that reported failure.
enum class qp_error : std::uint8_t
{
    dimension_mismatch,
    invalid_bounds,
    non_finite_input,
    capacity_exceeded,
    infeasible_start,
    invalid_problem,
    invalid_options,
    factorization_failed
};

template <scalar Scalar = double, int N = argmin::dynamic_dimension>
struct qp_result
{
    vector<Scalar, N> x;
    vector<Scalar> y;
    qp_solve_status status{qp_solve_status::solved};
    int iterations{0};
    bool polished{false};
    // True iff a primal seed (the caller's warm_start() vector or the iterate retained from
    // the previous solve) was consumed; a dual seed replaced by zeros does not clear it.
    bool warm_started{false};
    Scalar primal_residual{0};
    Scalar dual_residual{0};
    Scalar objective_value{0};
};

namespace detail
{

// The max(1, ...) floor keeps the previous absolute rule exactly at unit scale, so no row of
// unit magnitude reclassifies. It is one shared definition because the dense and sparse solvers
// must classify identically, or comparing their solutions stops being meaningful.
template <scalar Scalar>
bool is_equality_bound(Scalar l, Scalar u, Scalar eq_tol)
{
    if(!std::isfinite(l) || !std::isfinite(u))
        return false;
    return (u - l) <= eq_tol * std::max(Scalar(1), std::max(std::abs(l), std::abs(u)));
}

}

}

#endif
