#ifndef HPP_GUARD_ARGMIN_DETAIL_STATUS_CLAIM_H
#define HPP_GUARD_ARGMIN_DETAIL_STATUS_CLAIM_H

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <cstdint>
#include <optional>

namespace argmin
{
namespace detail
{

// A status whose stopping test accepted the final iterate is a claim about that iterate.
constexpr bool certifies_iterate(solver_status status) noexcept
{
    return status == solver_status::converged || status == solver_status::ftol_reached
        || status == solver_status::xtol_reached || status == solver_status::stalled;
}

// Every claim about the final iterate includes primal feasibility, the box included: optimality by N&W 2e
// Definition 12.1, and stalled / xtol_reached assert it outright. A stationarity threshold above the feasibility
// tolerance lets a composite KKT residual pass at an infeasible iterate, so the claim is gated on the violation
// directly, as IPOPT requires constr_viol_tol alongside its overall tol (Ipopt options reference, Termination).
template <typename Scalar>
constexpr std::optional<solver_status> admit_claim(std::optional<solver_status> status, const step_result<Scalar>& r, Scalar feas_tol) noexcept
{
    if(status && certifies_iterate(*status) && !(r.constraint_violation <= feas_tol && r.bound_violation <= feas_tol))
        return std::nullopt;
    return status;
}

// A policy reports a status from a state its own stopping test ended (COBYLA's radius at its floor, collapsed
// ISRES mutation strengths, an exhausted SQP reset), which the next step would report again, so an inadmissible
// policy claim ends the solve without the claim; a criterion is retested on the next iterate, which may be feasible.
template <typename Scalar>
constexpr solver_status admit_policy_status(solver_status status, const step_result<Scalar>& r, Scalar feas_tol) noexcept
{
    return admit_claim(std::optional{status}, r, feas_tol).value_or(solver_status::objective_stalled);
}

// A policy-reported status takes precedence and skips the convergence policy, which is consulted once per step
// because its windowed criteria advance on every call.
template <typename Scalar, typename Convergence>
solver_status resolve_step_status(const step_result<Scalar>& r, const Convergence& convergence, std::uint32_t iteration, Scalar feas_tol)
{
    if(r.policy_status)
        return admit_policy_status(*r.policy_status, r, feas_tol);
    return admit_claim(convergence.check(r, iteration), r, feas_tol).value_or(solver_status::running);
}

}
}

#endif
