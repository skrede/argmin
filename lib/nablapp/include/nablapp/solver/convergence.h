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

    // K&W 2e Section 4.4: convergence requires stationarity. Small objective
    // change alone is insufficient -- the gradient must also be small to
    // distinguish genuine convergence from a non-stationary plateau.
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 && std::abs(r.objective_change) < *threshold)
        {
            double gate = stationarity_threshold.value_or(1e-8);
            if(r.gradient_norm < gate)
                return solver_status::ftol_reached;
            return std::nullopt;
        }
        return std::nullopt;
    }
};

struct step_tolerance_criterion
{
    std::optional<double> threshold{};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
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

    // K&W 2e Section 4.4: stationarity gate applies to relative criterion too.
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 &&
           std::abs(r.objective_change) / std::max(std::abs(r.objective_value), 1.0) < *threshold)
        {
            double gate = stationarity_threshold.value_or(1e-8);
            if(r.gradient_norm < gate)
                return solver_status::ftol_reached;
            return std::nullopt;
        }
        return std::nullopt;
    }
};

// Relative step length test (N&W 2e Section 2.2; K&W 2e Section 2.3)
struct step_tolerance_rel_criterion
{
    std::optional<double> threshold{};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
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

        double metric = r.objective_value + r.constraint_violation;
        buffer_[iteration % max_window] = metric;

        if(iteration < window)
            return std::nullopt;

        double old_metric = buffer_[(iteration - window + 1) % max_window];

        if(std::abs(metric - old_metric) < *threshold * std::max(std::abs(metric), 1.0))
            return solver_status::stalled;

        return std::nullopt;
    }

private:
    static constexpr std::uint16_t max_window = 64;
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

// Constrained convergence: gates inner convergence on feasibility.
//
// Returns std::nullopt (no convergence) if constraint_violation >=
// feasibility_threshold, regardless of what the inner policy says.
// This ensures constrained solvers only declare convergence when
// both optimality criteria AND feasibility are satisfied.
//
// Reference: N&W 2e Section 12.1 (KKT conditions require feasibility).
template <typename Inner = default_convergence>
struct constrained_convergence_policy
{
    Inner inner{};
    std::optional<double> feasibility_threshold{};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(feasibility_threshold)
        {
            if(r.constraint_violation >= *feasibility_threshold)
                return std::nullopt;
        }
        return inner.check(r, iteration);
    }
};

using constrained_convergence = constrained_convergence_policy<default_convergence>;

}

#endif
