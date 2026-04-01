#ifndef HPP_GUARD_NABLAPP_SOLVER_CONVERGENCE_H
#define HPP_GUARD_NABLAPP_SOLVER_CONVERGENCE_H

#include "nablapp/result/step_result.h"
#include "nablapp/result/status.h"

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

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 && std::abs(r.objective_change) < *threshold)
            return solver_status::ftol_reached;
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

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(iteration > 1 &&
           std::abs(r.objective_change) / std::max(std::abs(r.objective_value), 1.0) < *threshold)
            return solver_status::ftol_reached;
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

template <typename... Criteria>
struct convergence_policy
{
    std::tuple<Criteria...> criteria;

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        std::optional<solver_status> result;
        auto try_one = [&](const auto& c) {
            if(!result) result = c.check(r, iteration);
        };
        std::apply([&](const auto&... cs) { (try_one(cs), ...); }, criteria);
        return result;
    }
};

using default_convergence = convergence_policy<
    gradient_tolerance_criterion,
    objective_tolerance_criterion,
    step_tolerance_criterion
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
