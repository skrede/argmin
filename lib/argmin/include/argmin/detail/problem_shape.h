#ifndef HPP_GUARD_ARGMIN_DETAIL_PROBLEM_SHAPE_H
#define HPP_GUARD_ARGMIN_DETAIL_PROBLEM_SHAPE_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <limits>

namespace argmin::detail
{

template <typename State>
bool state_rejected(const State& s)
{
    if constexpr(requires { s.invalid_problem; })
        return s.invalid_problem;
    else
        return false;
}

template <typename State>
solver_status latch_status(const State& s)
{
    return state_rejected(s) ? solver_status::invalid_problem : solver_status::running;
}

template <typename Scalar, typename State>
Scalar rejected_objective(const State& s)
{
    if constexpr(requires { s.f; })
        return s.f;
    else
        return std::numeric_limits<Scalar>::quiet_NaN();
}

template <typename Scalar, typename State>
Scalar rejected_gradient_norm(const State& s)
{
    if constexpr(requires { s.g.norm(); })
        return s.g.norm();
    else
        return std::numeric_limits<Scalar>::quiet_NaN();
}

// A static row capacity is a hard storage bound; dynamic_dimension stays a runtime capacity with no bound to exceed.
template <int Capacity>
bool exceeds_capacity(Eigen::Index rows)
{
    return Capacity != dynamic_dimension && rows > Capacity;
}

// A static row count sized exact fixed-size storage, so any other runtime count over-reads or overflows it.
template <int Rows>
bool mismatches_fixed_count(Eigen::Index rows)
{
    return Rows != dynamic_dimension && rows != Rows;
}

inline bool more_equalities_than_variables(Eigen::Index n_eq, Eigen::Index n)
{
    return n_eq > n;
}

template <typename Problem>
bool constraint_rows_changed(const Problem& problem, Eigen::Index n_eq, Eigen::Index n_ineq)
{
    return problem.num_equality() != n_eq || problem.num_inequality() != n_ineq;
}

// The result of a step the policy refuses to take on a problem it rejected.
template <typename Scalar, typename State>
step_result<Scalar> rejected_step(const State& s)
{
    return step_result<Scalar>{
        .objective_value = rejected_objective<Scalar>(s),
        .gradient_norm = rejected_gradient_norm<Scalar>(s),
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .is_null_step = true,
        .x_norm = s.x.norm(),
        .policy_status = solver_status::invalid_problem,
    };
}

}

#endif
