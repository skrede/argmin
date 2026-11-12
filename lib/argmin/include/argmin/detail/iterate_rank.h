#ifndef HPP_GUARD_ARGMIN_DETAIL_ITERATE_RANK_H
#define HPP_GUARD_ARGMIN_DETAIL_ITERATE_RANK_H

#include "argmin/detail/status_claim.h"

#include "argmin/result/status.h"

#include <limits>

namespace argmin
{
namespace detail
{

template <typename Scalar>
struct iterate_rank
{
    solver_status status;
    Scalar objective_value;
    Scalar constraint_violation;
    Scalar bound_violation;
};

// Feasibility-first best-seen comparator selecting the returned iterate, on the violation a claim is judged on (rows
// and box): feasible beats infeasible, two infeasible iterates rank by violation, two feasible ones by objective --
// except that an iterate whose status certifies it beats one whose status does not, since within the tolerance
// an earlier iterate's lower objective can be bought with violation and would not satisfy the status's claim.
//
// Reference: NLopt nlopt_optimize convention (nlopt/src/api/nlopt.c) -- the caller receives the best
//            point encountered, not the terminal trial.
template <typename Scalar>
constexpr bool is_better(const iterate_rank<Scalar>& cand, const iterate_rank<Scalar>& best, Scalar feas_tol) noexcept
{
    const Scalar cand_violation = claim_violation(cand.constraint_violation, cand.bound_violation);
    const Scalar best_violation = claim_violation(best.constraint_violation, best.bound_violation);
    const bool cand_feasible = cand_violation <= feas_tol;
    const bool best_feasible = best_violation <= feas_tol;
    if(cand_feasible != best_feasible)
        return cand_feasible;
    if(!cand_feasible)
        return cand_violation < best_violation;
    if(certifies_iterate(cand.status) != certifies_iterate(best.status))
        return certifies_iterate(cand.status);
    return cand.objective_value < best.objective_value;
}

// The best of the points a method has evaluated, ranked by the comparator that selects the solve's returned iterate,
// so the point a population method holds is the one the solve would choose among those it evaluated. A record built
// without a tolerance counts nothing as feasible.
template <typename Scalar>
class best_evaluated
{
public:
    best_evaluated()
        : best_evaluated(std::numeric_limits<Scalar>::quiet_NaN())
    {}

    explicit best_evaluated(Scalar feasibility_tolerance)
        : m_tolerance{feasibility_tolerance}
        , m_rank{unset()}
    {}

    bool offer(Scalar objective_value, Scalar constraint_violation, Scalar bound_violation)
    {
        const iterate_rank<Scalar> candidate{solver_status::running, objective_value, constraint_violation, bound_violation};
        if(!is_better(candidate, m_rank, m_tolerance))
            return false;
        m_rank = candidate;
        return true;
    }

    bool within_tolerance() const { return claim_violation(m_rank.constraint_violation, m_rank.bound_violation) <= m_tolerance; }

    void clear() { m_rank = unset(); }

private:
    Scalar m_tolerance;
    iterate_rank<Scalar> m_rank;

    static iterate_rank<Scalar> unset()
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return {solver_status::running, inf, inf, inf};
    }
};

}
}

#endif
