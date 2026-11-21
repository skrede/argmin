#ifndef HPP_GUARD_ARGMIN_DETAIL_BEST_POINT_H
#define HPP_GUARD_ARGMIN_DETAIL_BEST_POINT_H

#include "argmin/detail/iterate_rank.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <limits>

namespace argmin
{
namespace detail
{

// The point a solve returns is the best one it encountered, not the terminal trial, and every field reported for it
// is taken from that one iterate. The start point carries no gradient norm, since no step evaluated one there. A
// policy without a problem binding (Problem == void) seeds f at +inf, so its first accepted step becomes the baseline.
//
// Reference: nlopt/src/api/nlopt.c nlopt_optimize (best-solution-returned semantics); N&W 2e Def 12.1 (primal
//            feasibility, L-infinity composition).
template <typename Vector, typename Scalar>
class best_point
{
public:
    best_point(const Vector& x0, const iterate_rank<Scalar>& seed)
        : m_x{x0}
        , m_gradient_norm{std::numeric_limits<Scalar>::quiet_NaN()}
        , m_rank{seed}
    {}

    template <typename X>
    bool offer(solver_status status, const step_result<Scalar>& step, const X& x, Scalar feas_tol)
    {
        const iterate_rank<Scalar> candidate{status, step.objective_value, step.constraint_violation, step.bound_violation};
        if(!is_better(candidate, m_rank, feas_tol))
            return false;
        m_x = x;
        m_gradient_norm = step.gradient_norm;
        m_rank = candidate;
        return true;
    }

    const Vector& x() const { return m_x; }

    Scalar objective_value() const { return m_rank.objective_value; }

    Scalar constraint_violation() const { return m_rank.constraint_violation; }

    Scalar gradient_norm() const { return m_gradient_norm; }

private:
    Vector m_x;
    Scalar m_gradient_norm;
    iterate_rank<Scalar> m_rank;
};

}
}

#endif
