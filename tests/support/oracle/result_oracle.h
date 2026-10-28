#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_RESULT_ORACLE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_RESULT_ORACLE_H

#include "oracle/kkt_measure.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <string>
#include <cstdint>
#include <utility>

namespace argmin::oracle
{

// Least-squares multipliers minimize the 2-norm, so with matching active sets the infinity-norm
// residual is at most sqrt(n) times the solver's own, and n <= 24 in the fixtures.
inline constexpr double safety_factor = 10.0;

enum class claim : std::uint8_t
{
    unfinished,
    none,
    feasibility,
    ftol_optimality,
    optimality
};

inline claim claim_of(solver_status status)
{
    switch(status)
    {
    case solver_status::running:
        return claim::unfinished;
    case solver_status::converged:
        return claim::optimality;
    case solver_status::ftol_reached:
        return claim::ftol_optimality;
    case solver_status::stalled:
    case solver_status::xtol_reached:
        return claim::feasibility;
    case solver_status::max_iterations: case solver_status::budget_exhausted:
    case solver_status::maxeval_reached: case solver_status::time_limit_reached:
    case solver_status::diverged: case solver_status::roundoff_limited: case solver_status::line_search_failed:
    case solver_status::trust_region_step_rejected: case solver_status::objective_stalled:
    case solver_status::aborted: case solver_status::invalid_problem:
        return claim::none;
    }
    return claim::none;
}

template <typename Scalar>
struct verdict
{
    verdict(claim claimed_by_status, const kkt_legs<Scalar>& measured, std::string failing_leg)
        : ok(failing_leg.empty())
        , claimed(claimed_by_status)
        , failing(std::move(failing_leg))
        , legs(measured)
    {
    }

    bool ok;
    claim claimed;
    std::string failing;
    kkt_legs<Scalar> legs;
};

template <typename Scalar>
bool within(Scalar leg, Scalar tolerance)
{
    return leg <= static_cast<Scalar>(safety_factor) * tolerance;
}

template <typename Scalar>
std::string feasibility_failure(const kkt_legs<Scalar>& legs, Scalar feasibility)
{
    if(!within(legs.equality_violation, feasibility))
        return "equality_violation";
    if(!within(legs.inequality_violation, feasibility))
        return "inequality_violation";
    if(!within(legs.bound_violation, feasibility))
        return "bound_violation";
    return {};
}

template <typename Scalar>
std::string optimality_failure(const kkt_legs<Scalar>& legs, Scalar stationarity, Scalar feasibility)
{
    if(!within(legs.stationarity, stationarity))
        return "stationarity";
    std::string infeasible = feasibility_failure(legs, feasibility);
    if(!infeasible.empty())
        return infeasible;
    if(!within(legs.complementarity, stationarity))
        return "complementarity";
    if(!within(legs.dual_sign_breach, stationarity))
        return "dual_sign_breach";
    return {};
}

template <typename Scalar>
std::string claim_failure(claim claimed, const kkt_legs<Scalar>& legs, const tolerances<Scalar>& tol)
{
    switch(claimed)
    {
    case claim::optimality:
        return optimality_failure(legs, tol.stationarity, tol.feasibility);
    case claim::ftol_optimality:
        return optimality_failure(legs, tol.ftol_stationarity, tol.feasibility);
    case claim::feasibility:
        return feasibility_failure(legs, tol.feasibility);
    case claim::none:
    case claim::unfinished:
        return {};
    }
    return {};
}

// Every status must terminate the run and return a finite x of the problem's dimension before its
// claim is judged; the evaluation is skipped otherwise, so a malformed x cannot reach a callback.
template <typename Problem, typename Scalar, int N>
verdict<Scalar> judge(const Problem& problem, const solve_result<Scalar, N>& result, const tolerances<Scalar>& tol)
{
    const claim claimed = claim_of(result.status);
    if(claimed == claim::unfinished)
        return verdict<Scalar>(claimed, kkt_legs<Scalar>(), "status");
    if(result.x.size() != static_cast<Eigen::Index>(problem.dimension()))
        return verdict<Scalar>(claimed, kkt_legs<Scalar>(), "dimension");
    if(!result.x.allFinite())
        return verdict<Scalar>(claimed, kkt_legs<Scalar>(), "finite");
    const Scalar active_tol = static_cast<Scalar>(safety_factor) * tol.feasibility;
    const kkt_legs<Scalar> legs = measure_kkt(problem, result.x, active_tol);
    return verdict<Scalar>(claimed, legs, claim_failure(claimed, legs, tol));
}

}

#endif
