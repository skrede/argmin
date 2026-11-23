#ifndef HPP_GUARD_ARGMIN_DETAIL_ASYMPTOTE_TRIAL_H
#define HPP_GUARD_ARGMIN_DETAIL_ASYMPTOTE_TRIAL_H

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/iterate_values.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/options/mma_defaults.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

// No conservative approximation bounds a trial ranked worst (an infinite objective, every row violated without
// bound), so the conservativity loop grows its penalty at the formula's cap and never accepts the trial.
template <typename Scalar, typename Values>
void rank_worst(Scalar& f, Values& c)
{
    f = std::numeric_limits<Scalar>::infinity();
    c.setConstant(-std::numeric_limits<Scalar>::infinity());
}

template <typename Scalar, typename Values>
void rank_non_finite_worst(Scalar& f, Values& c)
{
    if(!(std::isfinite(f) && c.allFinite()))
        rank_worst(f, c);
}

// Svanberg 2002 Section 4.2 concheck against the raw approximation values; in the MMA convention g_i = -c_i, and the
// artificial slacks that keep the subproblem feasible do not enter it.
template <typename Scalar, typename Dual, typename Values>
bool is_conservative(const Dual& dual, Scalar f, const Values& c)
{
    bool conservative = dual.gval >= f;
    for(Eigen::Index i = 0; i < c.size() && conservative; ++i)
        conservative = dual.gcval[i] >= -c[i];
    return conservative;
}

template <typename Problem, typename X, typename G, typename Jac>
bool derivatives_finite_at(const Problem& problem, const X& x, G& g, Jac& J)
{
    problem.gradient(x, g);
    if(J.rows() > 0)
        problem.constraint_jacobian(x, J);
    return g.allFinite() && J.allFinite();
}

// A conservative trial is accepted only with a finite gradient and constraint Jacobian there, evaluated into g and J;
// otherwise it fails and ranks worst.
template <typename Scalar, typename Problem, typename X, typename Values, typename G, typename Jac>
bool accepts_trial(bool conservative, const Problem& problem, const X& x, Scalar& f, Values& c, G& g, Jac& J)
{
    if(!conservative)
        return false;
    if(derivatives_finite_at(problem, x, g, J))
        return true;
    rank_worst(f, c);
    return false;
}

// The asymptotes contract around x by the oscillation rule's factor, no closer than its minimum distance, and the
// history restarts at x so the next step keeps them; false once every one already sits at that distance.
template <typename Scalar, typename State>
bool contract_asymptotes(State& s, Scalar factor, Scalar min_fraction)
{
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
    bool moved = false;
    for(Eigen::Index j = 0; j < s.x.size(); ++j)
    {
        const Scalar range = s.lower[j] > -inf && s.upper[j] < inf ? s.upper[j] - s.lower[j] : std::max(std::abs(s.x[j]), Scalar(1));
        const Scalar min_d = std::max(min_fraction * range, mma_defaults<Scalar>::min_asymptote_distance);
        const Scalar to_lower = factor * (s.x[j] - s.L[j]);
        const Scalar to_upper = factor * (s.U[j] - s.x[j]);
        moved = moved || to_lower > min_d || to_upper > min_d;
        s.L[j] = s.x[j] - std::max(to_lower, min_d);
        s.U[j] = s.x[j] + std::max(to_upper, min_d);
    }
    s.x_old1 = s.x;
    s.x_old2 = s.x;
    return moved;
}

// The step reported at the iterate the state holds; the KKT residual is N&W 2e Definition 12.1 with eq. 12.34, the
// dual multipliers serving as the inequality multipliers with no sign flip.
template <typename Scalar, typename State>
step_result<Scalar> asymptote_step(const State& s, Scalar f_old, Scalar step_size)
{
    constexpr int N = decltype(s.x)::RowsAtCompileTime;
    const Eigen::Matrix<Scalar, 0, N> J_eq_empty(0, s.x.size());
    const Eigen::Vector<Scalar, 0> lambda_eq_empty;
    const Eigen::Vector<Scalar, 0> c_eq_empty;
    const Scalar kkt = kkt_residual<Scalar, N, 0, State::M>(s.g, J_eq_empty, s.J_ineq, lambda_eq_empty, s.y_dual, c_eq_empty, s.c_ineq);
    return step_result<Scalar>{
        .objective_value = s.f,
        .gradient_norm = s.g.norm(),
        .step_size = step_size,
        .objective_change = s.f - f_old,
        .improved = s.f < f_old,
        .constraint_violation = primal_feasibility_inf(c_eq_empty, s.c_ineq),
        .x_norm = s.x.norm(),
        .kkt_residual = kkt,
    };
}

template <typename Scalar, typename State>
step_result<Scalar> asymptote_null_step(const State& s, std::optional<solver_status> status = std::nullopt)
{
    step_result<Scalar> result = asymptote_step<Scalar>(s, s.f, Scalar(0));
    result.is_null_step = true;
    result.policy_status = status;
    return result;
}

// A trial failed by a non-finite value contracts the asymptotes; the solve ends once a trial made with them already at
// their minimum distance fails too.
template <typename Scalar, typename State>
step_result<Scalar> contracted_null_step(State& s, Scalar factor, Scalar min_fraction)
{
    const bool at_floor = s.asymptotes_at_floor;
    s.asymptotes_at_floor = !contract_asymptotes(s, factor, min_fraction);
    return asymptote_null_step<Scalar>(s, non_finite_trial_status(false, at_floor));
}

// A trial within the iterate's resolution is the conservativity loop's fixed point: no larger penalty or smaller window
// moves the trial any closer to x.
template <typename X>
bool at_iterate(const X& x_trial, const X& x)
{
    using Scalar = typename X::Scalar;
    return (x_trial - x).norm() <= std::numeric_limits<Scalar>::epsilon() * (Scalar(1.0) + x.norm());
}

// Whether the trial ends the solve: a trial at the iterate that is non-finite, or follows a non-finite rejection, leaves
// the penalty growth or window shrink nothing to try. The record of a non-finite trial carries across a null step, so
// the next outer step continues from the grown penalty or the shrunk window.
template <typename State, typename X, typename Scalar>
bool non_finite_limit(State& s, const X& x_trial, Scalar f_trial)
{
    const bool non_finite = !std::isfinite(f_trial);
    const bool limit = (non_finite || s.non_finite_rejected) && at_iterate(x_trial, s.x);
    s.non_finite_rejected = non_finite;
    return limit;
}

template <typename Scalar, typename State, typename X>
step_result<Scalar> non_finite_null_step(State& s, const X& x_trial)
{
    return asymptote_null_step<Scalar>(s, non_finite_trial_status(false, non_finite_limit(s, x_trial, std::numeric_limits<Scalar>::infinity())));
}

}

#endif
