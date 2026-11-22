#ifndef HPP_GUARD_ARGMIN_DETAIL_DECORATOR_VERDICT_H
#define HPP_GUARD_ARGMIN_DETAIL_DECORATOR_VERDICT_H

#include "argmin/detail/iterate_rank.h"
#include "argmin/detail/status_claim.h"
#include "argmin/detail/bound_measure.h"
#include "argmin/detail/terminal_step.h"
#include "argmin/detail/hint_precedence.h"
#include "argmin/detail/bound_projection.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/options/convergence_defaults.h"

#include <Eigen/Core>

#include <limits>
#include <cstdint>
#include <type_traits>

namespace argmin::detail
{

// A decorator judges each member run as a solve of the inner policy alone is judged: by the inner policy's own status,
// else by its own copy of the solve's convergence settings under the inner policy's hints, both admitted at the
// tolerance in force. A member whose objective improves on none of its earlier steps for the stagnation limit stalls.
// The copy is the caller's object when it is default_convergence of this scalar, a default one otherwise.
template <typename Scalar>
class member_judge
{
public:
    member_judge()
        : m_tolerance{convergence_defaults<Scalar>::feasibility_tolerance}
        , m_best_objective{std::numeric_limits<Scalar>::infinity()}
        , m_iteration{0}
        , m_stagnation{0}
        , m_stagnation_limit{std::numeric_limits<std::uint32_t>::max()}
        , m_convergence{}
    {}

    template <typename Convergence, typename Inner>
    member_judge(const solver_options<Convergence>& opts, const Inner& inner, std::uint32_t stagnation_limit)
        : m_tolerance{static_cast<Scalar>(opts.feasibility_tolerance_in_force())}
        , m_best_objective{std::numeric_limits<Scalar>::infinity()}
        , m_iteration{0}
        , m_stagnation{0}
        , m_stagnation_limit{stagnation_limit}
        , m_convergence{convergence_of(opts)}
    {
        if constexpr(requires { inner.options; })
            write_hints(m_convergence, inner.options);
        m_convergence.reset();
    }

    solver_status judge(const step_result<Scalar>& r)
    {
        const solver_status verdict = resolve_step_status(r, m_convergence, ++m_iteration, m_tolerance);
        if(verdict != solver_status::running)
            return verdict;
        if(r.objective_value < m_best_objective)
        {
            m_best_objective = r.objective_value;
            m_stagnation = 0;
        }
        else
            ++m_stagnation;
        return m_stagnation < m_stagnation_limit ? solver_status::running : admit_policy_status(solver_status::stalled, r, m_tolerance);
    }

    void restart()
    {
        m_convergence.reset();
        m_best_objective = std::numeric_limits<Scalar>::infinity();
        m_iteration = 0;
        m_stagnation = 0;
    }

    Scalar tolerance() const { return m_tolerance; }

private:
    Scalar m_tolerance;
    Scalar m_best_objective;
    std::uint32_t m_iteration;
    std::uint32_t m_stagnation;
    std::uint32_t m_stagnation_limit;
    default_convergence<Scalar> m_convergence;

    template <typename Convergence>
    static default_convergence<Scalar> convergence_of(const solver_options<Convergence>& opts)
    {
        if constexpr(std::is_same_v<Convergence, default_convergence<Scalar>>)
            return opts.convergence;
        else
            return {};
    }
};

// The best member a decorator has run, ranked as a solve ranks its returned iterate, every field it reports taken from
// that member's last step and state.
template <typename Vector, typename Scalar>
class best_member
{
public:
    best_member()
        : m_x{}
        , m_status{solver_status::running}
        , m_step{}
        , m_c_eq{}
        , m_c_ineq{}
    {}

    template <typename State>
    void offer(solver_status status, const step_result<Scalar>& step, const State& s, Scalar feas_tol)
    {
        if(m_status != solver_status::running && !is_better(rank_of(status, step), rank_of(m_status, m_step), feas_tol))
            return;
        m_x = s.x;
        m_status = status;
        m_step = step;
        m_c_eq = s.c_eq;
        m_c_ineq = s.c_ineq;
    }

    // Puts the member back into the decorator state, so the solve returns its point, and reports its last step.
    template <typename State>
    step_result<Scalar> surface(State& s, std::uint32_t evaluations) const
    {
        s.x = m_x;
        s.c_eq = m_c_eq;
        s.c_ineq = m_c_ineq;
        if constexpr(requires { s.objective_value; })
            s.objective_value = m_step.objective_value;
        return member_end_step(m_step, m_status, evaluations);
    }

    const Vector& x() const { return m_x; }

    void clear() { m_status = solver_status::running; }

private:
    Vector m_x;
    solver_status m_status;
    step_result<Scalar> m_step;
    Eigen::VectorX<Scalar> m_c_eq;
    Eigen::VectorX<Scalar> m_c_ineq;

    static iterate_rank<Scalar> rank_of(solver_status status, const step_result<Scalar>& step)
    {
        return {status, step.objective_value, step.constraint_violation, step.bound_violation};
    }
};

// One decorator step after the inner step: a member that has not ended keeps the solve running; an ended member is
// recorded, and either the decorator ends with its best member or the next step starts the next member.
template <typename Problem, typename State, typename Scalar, typename Ends>
step_result<Scalar> member_step(State& s, step_result<Scalar> result, Ends ends)
{
    using vector_type = std::remove_cvref_t<decltype(s.x)>;
    if constexpr(boxed_problem<Problem, Scalar>)
        result.bound_violation = box_violation_inf<Scalar, vector_type::RowsAtCompileTime>(s.x, s.lower, s.upper);
    const solver_status verdict = s.judge.judge(result);
    if(verdict == solver_status::running)
        return working_step(result);
    s.best.offer(verdict, result, s, s.judge.tolerance());
    if(ends(verdict))
        return s.best.surface(s, result.evaluations);
    s.restart_pending = true;
    return working_step(result);
}

}

#endif
