#ifndef HPP_GUARD_ARGMIN_DETAIL_LEAST_SQUARES_TRIAL_H
#define HPP_GUARD_ARGMIN_DETAIL_LEAST_SQUARES_TRIAL_H

#include "argmin/detail/evaluation_count.h"

#include "argmin/derivative/finite_difference.h"

#include <Eigen/Core>

#include <cmath>

namespace argmin::detail
{

// Analytic when the problem provides jacobian(), central differences of the residuals otherwise.
template <int N, typename Scalar, typename State>
void residual_jacobian(State& s, const Eigen::Vector<Scalar, N>& x, Eigen::MatrixX<Scalar>& sink)
{
    if constexpr(requires { s.problem->jacobian(x, sink); })
        s.problem->jacobian(x, sink);
    else
    {
        auto residual_fn = [&s](const Eigen::Vector<Scalar, N>& xx, Eigen::VectorX<Scalar>& rr) { counted_residuals(s.evaluations, *s.problem, xx, rr); };
        fd_jacobian<N>(residual_fn, x, sink, s.num_residuals);
    }
}

// Whether the trial's values are finite. An accepted trial keeps its acceptance only with a finite Jacobian, evaluated
// into sink before anything is committed, so a non-finite one is rejected like a non-finite residual.
template <int N, typename Scalar, typename State>
bool finite_trial(State& s, const Eigen::Vector<Scalar, N>& x_trial, Scalar f_trial, Eigen::MatrixX<Scalar>& sink, bool& accepted)
{
    if(!accepted)
        return std::isfinite(f_trial);
    residual_jacobian<N, Scalar>(s, x_trial, sink);
    accepted = sink.allFinite();
    return accepted;
}

// Whether the trial failed. After a non-finite rejection a trial the damped step no longer moves off x fails too: every
// trial that moved was non-finite, so the shrink has run out below x's resolution as surely as at the damping ceiling.
template <typename State, typename X>
bool failed_trial(State& s, const X& x_trial, bool trial_finite)
{
    const bool failed = !trial_finite || (s.non_finite_rejected && x_trial == s.x);
    s.non_finite_rejected = failed;
    return failed;
}

}

#endif
