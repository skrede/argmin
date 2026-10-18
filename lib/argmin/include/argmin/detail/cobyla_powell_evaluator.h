#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_POWELL_EVALUATOR_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_POWELL_EVALUATOR_H

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <type_traits>

namespace argmin::detail
{

// Count finite box bounds (added as explicit Powell constraints).
template <typename Scalar>
inline std::int32_t cobyla_finite_bound_count(const Eigen::VectorX<Scalar>& lower, const Eigen::VectorX<Scalar>& upper)
{
    int n_bound = 0;
    for(int i = 0; i < static_cast<int>(lower.size()); ++i)
    {
        if(std::isfinite(lower[i])) ++n_bound;
        if(std::isfinite(upper[i])) ++n_bound;
    }
    return n_bound;
}

template <typename Scalar, typename P>
struct cobyla_powell_evaluator
{
    const P& problem;
    std::int32_t n;
    std::int32_t n_eq;
    std::int32_t n_ineq;
    Eigen::VectorX<Scalar> lower;
    Eigen::VectorX<Scalar> upper;

    Scalar operator()(const Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& con) const
    {
        const int m_total = n_eq + n_ineq;
        Eigen::VectorX<Scalar> c_full(std::max(m_total, 0));
        if(m_total > 0)
            problem.constraints(x, c_full);

        int idx = 0;
        for(int i = 0; i < n_ineq; ++i)
            con(idx++) = c_full[n_eq + i];
        for(int j = 0; j < n_eq; ++j)
        {
            con(idx++) = c_full[j];
            con(idx++) = -c_full[j];
        }
        for(int i = 0; i < n; ++i)
        {
            if(std::isfinite(lower[i]))
                con(idx++) = x[i] - lower[i];
            if(std::isfinite(upper[i]))
                con(idx++) = upper[i] - x[i];
        }
        return problem.value(x);
    }
};

// Build the Powell-convention evaluator (holds the problem by reference)
// and initialize the engine. The evaluator also refreshes s.c_eq / s.c_ineq
// so step_budget_solver's constraint_violation reporting sees the latest point.
template <typename State, typename Scalar>
inline void cobyla_init_engine(State& s, Scalar rhobeg, Scalar rhoend)
{
    using problem_type = std::remove_cvref_t<decltype(*s.problem)>;
    const int n = static_cast<int>(s.x.size());
    cobyla_powell_evaluator<Scalar, problem_type> eval{*s.problem, n, s.n_eq, s.n_ineq, s.lower, s.upper};
    s.engine->init(n, s.m_powell, s.x, s.lower, s.upper, rhobeg, rhoend,
                   std::move(eval));
}

}

#endif
