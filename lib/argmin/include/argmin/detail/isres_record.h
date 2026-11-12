#ifndef HPP_GUARD_ARGMIN_DETAIL_ISRES_RECORD_H
#define HPP_GUARD_ARGMIN_DETAIL_ISRES_RECORD_H

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/iterate_rank.h"
#include "argmin/detail/bound_projection.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <optional>

namespace argmin::detail
{

template <typename State>
using held_vector = decltype(State::x);

template <typename State, typename Point>
typename held_vector<State>::Scalar outside_box(const State& s, const Point& x)
{
    using vector_type = held_vector<State>;
    return box_violation_inf<typename vector_type::Scalar, vector_type::RowsAtCompileTime>(x, s.lower, s.upper);
}

template <typename State, typename Points, typename Values>
void hold_individual(State& s, const Points& points, const Values& values, Eigen::Index k)
{
    s.x = points.col(k);
    s.objective_value = values[k];
    s.c_eq = s.all_constraints_buf.col(k).head(s.n_eq);
    s.c_ineq = s.all_constraints_buf.col(k).tail(s.n_ineq);
}

// Each evaluated individual is measured on its rows and the box, as a returned iterate is, and the policy holds the
// record. The rows come from the evaluation buffer, so no individual is evaluated twice.
template <typename State, typename Points, typename Values>
bool keep_best_evaluated(State& s, const Points& points, const Values& values)
{
    Eigen::Index kept = -1;
    for(Eigen::Index k = 0; k < points.cols(); ++k)
    {
        const auto rows = s.all_constraints_buf.col(k);
        if(s.record.offer(values[k], primal_feasibility_inf(rows.head(s.n_eq), rows.tail(s.n_ineq)), outside_box(s, points.col(k))))
            kept = k;
    }
    if(kept >= 0)
        hold_individual(s, points, values, kept);
    return kept >= 0;
}

// The first individual is held until one ranks, so a population with no comparable point still holds a point.
template <typename State, typename Points, typename Values, typename Scalar>
void start_best_evaluated(State& s, const Points& points, const Values& values, Scalar feasibility_tolerance)
{
    s.record = best_evaluated<Scalar>{feasibility_tolerance};
    hold_individual(s, points, values, 0);
    keep_best_evaluated(s, points, values);
}

// A reset evaluates only its start, so the start is the held point and the record's one candidate.
template <typename State, typename Start>
void restart_best_evaluated(State& s, const Start& x0)
{
    s.x = x0;
    s.objective_value = s.problem->value(s.x);
    if(s.n_eq + s.n_ineq > 0)
    {
        Eigen::VectorX<typename held_vector<State>::Scalar> c(s.n_eq + s.n_ineq);
        s.problem->constraints(s.x, c);
        s.c_eq = c.head(s.n_eq);
        s.c_ineq = c.tail(s.n_ineq);
    }
    s.record.clear();
    s.record.offer(s.objective_value, primal_feasibility_inf(s.c_eq, s.c_ineq), outside_box(s, s.x));
}

// A collapsed population has stopped moving, which certifies the record only as a point within the feasibility
// tolerance: an evolution strategy holds no gradient or multiplier estimate to certify optimality with.
template <typename Scalar>
std::optional<solver_status> collapse_status(bool collapsed, const best_evaluated<Scalar>& record)
{
    if(!collapsed)
        return std::nullopt;
    return record.within_tolerance() ? solver_status::xtol_reached : solver_status::objective_stalled;
}

}

#endif
