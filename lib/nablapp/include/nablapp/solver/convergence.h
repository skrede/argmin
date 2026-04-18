#ifndef HPP_GUARD_NABLAPP_SOLVER_CONVERGENCE_H
#define HPP_GUARD_NABLAPP_SOLVER_CONVERGENCE_H

#include "nablapp/result/step_result.h"
#include "nablapp/result/status.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

namespace nablapp
{

struct gradient_tolerance_criterion
{
    std::optional<double> threshold{};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t /*iteration*/) const
    {
        if(!threshold)
            return std::nullopt;
        if(r.gradient_norm < *threshold)
            return solver_status::converged;
        return std::nullopt;
    }
};

struct objective_tolerance_criterion
{
    std::optional<double> threshold{};
    std::optional<double> stationarity_threshold{};
    // Primal feasibility is folded directly into r.kkt_residual via the
    // full first-order optimality error composition (stationarity + primal
    // equality + primal inequality + dual feasibility + complementarity).
    // A separate feasibility knob here would be two-knobs-for-one-test.
    //
    // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
    //            Definition 12.1 (KKT conditions); eq. 12.34 (Lagrangian
    //            stationarity leg); max-composition across the five
    //            optimality legs.

    // K&W 2e Section 4.4: ftol convergence requires a small objective
    // change AND a small composite KKT residual (which now carries the
    // primal-feasibility information). Stationarity is evaluated against
    // kkt_residual (populated by gradient-aware policies) with a fallback
    // to gradient_norm for policies that leave kkt_residual nullopt.
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 && std::abs(r.objective_change) < *threshold)
        {
            const double gate = stationarity_threshold.value_or(1e-8);
            const double kkt  = r.kkt_residual.value_or(r.gradient_norm);
            if(kkt < gate)
                return solver_status::ftol_reached;
        }
        return std::nullopt;
    }
};

struct step_tolerance_criterion
{
    std::optional<double> threshold{};

    // A null step (step_size == 0 because the policy intentionally made
    // no move -- SQP zero-step degeneracy, trust-region rho contraction,
    // restoration exhaustion) is not a stall: it is an algorithmic
    // signal that the policy needs more iterations, not fewer. Exempting
    // null steps here avoids false stall detection on iteration 2 for
    // any policy that sets is_null_step.
    //
    // Reference: N&W 2e Section 18.4 (SQP convergence analysis).
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(r.is_null_step)
            return std::nullopt;
        if(iteration > 1 && r.step_size < *threshold)
            return solver_status::stalled;
        return std::nullopt;
    }
};

// Relative function decrease test (N&W 2e Section 2.2; K&W 2e Section 2.3)
struct objective_tolerance_rel_criterion
{
    std::optional<double> threshold{};
    std::optional<double> stationarity_threshold{};
    // Primal feasibility is folded directly into r.kkt_residual via the
    // full first-order optimality error composition (stationarity + primal
    // equality + primal inequality + dual feasibility + complementarity).
    // A separate feasibility knob here would be two-knobs-for-one-test.
    //
    // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
    //            Definition 12.1 (KKT conditions); eq. 12.34 (Lagrangian
    //            stationarity leg); max-composition across the five
    //            optimality legs.

    // K&W 2e Section 4.4: ftol convergence requires a small relative
    // objective change AND a small composite KKT residual (which now
    // carries the primal-feasibility information). Stationarity is
    // evaluated against kkt_residual (populated by gradient-aware
    // policies) with a fallback to gradient_norm for policies that leave
    // kkt_residual nullopt.
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 &&
           std::abs(r.objective_change) / std::max(std::abs(r.objective_value), 1.0) < *threshold)
        {
            const double gate = stationarity_threshold.value_or(1e-8);
            const double kkt  = r.kkt_residual.value_or(r.gradient_norm);
            if(kkt < gate)
                return solver_status::ftol_reached;
        }
        return std::nullopt;
    }
};

// Relative step length test (N&W 2e Section 2.2; K&W 2e Section 2.3)
struct step_tolerance_rel_criterion
{
    std::optional<double> threshold{};

    // Null steps are exempt for the same reason as in
    // step_tolerance_criterion: a policy-intentional zero move is an
    // algorithmic signal, not a stall. See step_tolerance_criterion::check
    // for the detailed rationale.
    //
    // Reference: N&W 2e Section 18.4 (SQP convergence analysis).
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(r.is_null_step)
            return std::nullopt;
        if(iteration > 1 &&
           r.step_size / std::max(r.x_norm, 1.0) < *threshold)
            return solver_status::xtol_reached;
        return std::nullopt;
    }
};

// Stall detection: terminates when objective + constraint violation makes
// no progress over a rolling window of K iterations.
//
// Unlike the other criteria which are stateless, this maintains a circular
// buffer of combined metric values. The mutable buffer keeps convergence
// logic self-contained within the criterion.
//
// The combined metric (objective + constraint_violation) catches SQP
// solvers that cycle between feasible and infeasible points, where the
// objective alone may appear to oscillate while overall progress stalls.
//
// Reference: K&W 2e Section 4.4 (convergence criteria).
struct stall_tolerance_criterion
{
    std::optional<double> threshold{};
    std::uint16_t window{50};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;

        auto effective_window = std::min(window, max_window);

        double metric = r.objective_value + r.constraint_violation;
        buffer_[iteration % max_window] = metric;

        if(iteration < effective_window)
            return std::nullopt;

        double old_metric = buffer_[(iteration - effective_window + 1) % max_window];

        if(std::abs(metric - old_metric) < *threshold * std::max(std::abs(metric), 1.0))
            return solver_status::stalled;

        return std::nullopt;
    }

private:
    static constexpr std::uint16_t max_window = 512;
    mutable std::array<double, max_window> buffer_{};
};

template <typename... Criteria>
struct convergence_policy
{
    std::tuple<Criteria...> criteria;

    // Per-criterion statuses from the most recent check() call, exposed
    // via last_check_results(). Kept public (not private) so the struct
    // stays an aggregate for designated-initializer construction through
    // solver_options<>. Mutable because check() is const.
    mutable std::array<std::optional<solver_status>, sizeof...(Criteria)>
        last_check_results_{};

    // Populates last_check_results_ with every criterion's per-iteration
    // status, then returns the first-firing one as the reported
    // terminator. Short-circuit semantics for the return value are
    // preserved; per-criterion outcomes are available via
    // last_check_results(). Reference: K&W 2e Section 4.4.
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        std::size_t idx = 0;
        auto call_one = [&](const auto& c) {
            last_check_results_[idx] = c.check(r, iteration);
            ++idx;
        };
        std::apply([&](const auto&... cs) { (call_one(cs), ...); }, criteria);

        for(const auto& status : last_check_results_)
        {
            if(status)
                return status;
        }
        return std::nullopt;
    }

    // Per-criterion statuses from the most recent check() call. Index
    // matches declaration order of Criteria...; a nullopt entry means
    // the criterion did not fire. Reachable from basic_solver via
    // solver.convergence().last_check_results().
    [[nodiscard]] const std::array<std::optional<solver_status>,
                                   sizeof...(Criteria)>&
    last_check_results() const noexcept
    {
        return last_check_results_;
    }
};

using default_convergence = convergence_policy<
    gradient_tolerance_criterion,
    objective_tolerance_criterion,
    step_tolerance_criterion,
    stall_tolerance_criterion
>;

// NLopt-SLSQP-compatible convergence: matches NLopt's slsqp.c:1140-1220
// which only stops on ftol_rel/ftol_abs (objective_tolerance_rel_criterion)
// and xtol_rel/xtol_abs (step_tolerance_rel_criterion). NLopt SLSQP sets
// acc=0 at slsqp.c:1047 with the literal comment "we do our own convergence
// tests below", which disables its internal gradient-norm + constraint-
// violation check. Consumers who want iter-count parity with NLopt on
// IK-style workloads should instantiate basic_solver with this alias
// instead of default_convergence. stall_tolerance_criterion is retained
// as a safety cap -- NLopt relies on its outer max_eval for the same
// purpose and the stall detector fills the equivalent role here.
//
// Reference: K&W 2e Section 4.4 (convergence criteria), N&W 2e Section 2.2
//            (relative/absolute tolerance tests).
using slsqp_compatible_convergence = convergence_policy<
    objective_tolerance_rel_criterion,
    step_tolerance_rel_criterion,
    stall_tolerance_criterion
>;

}

#endif
