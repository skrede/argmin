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

constexpr bool claims_optimality(solver_status status) noexcept
{
    return status == solver_status::converged || status == solver_status::ftol_reached;
}

// Optimality includes primal feasibility (N&W 2e Definition 12.1), and a stationarity threshold above the
// feasibility tolerance lets a composite KKT residual pass at an infeasible iterate, so the claim is gated on
// the violation directly, as IPOPT requires constr_viol_tol alongside its overall tol (Ipopt options reference,
// Termination). An inadmissible claim is no claim: the solve goes on.
template <typename Scalar>
constexpr std::optional<solver_status> admit_claim(std::optional<solver_status> status, Scalar cv, Scalar feas_tol) noexcept
{
    if(status && claims_optimality(*status) && !(cv <= feas_tol))
        return std::nullopt;
    return status;
}

// A policy-reported status takes precedence and skips the convergence policy, which is consulted once per step
// because its windowed criteria advance on every call.
template <typename Scalar, typename Convergence>
solver_status resolve_step_status(const step_result<Scalar>& r, const Convergence& convergence, std::uint32_t iteration, Scalar feas_tol)
{
    const std::optional<solver_status> status = r.policy_status ? r.policy_status : convergence.check(r, iteration);
    return admit_claim(status, r.constraint_violation, feas_tol).value_or(solver_status::running);
}

}
}

#endif
