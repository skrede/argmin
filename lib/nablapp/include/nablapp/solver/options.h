#ifndef HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H
#define HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H

#include "nablapp/solver/convergence.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <tuple>

namespace nablapp
{

// Solver configuration carrying convergence policy and iteration limits.
//
// The Convergence template parameter is a convergence_policy<Criteria...>
// that composes criterion types via fold expression. Individual tolerance
// thresholds live in the criterion types, not here.

template <typename Convergence = default_convergence>
struct solver_options
{
    using convergence_type = Convergence;

    std::uint32_t max_iterations{1000};
    std::uint8_t verbosity{0};
    std::optional<std::chrono::nanoseconds> max_time{};
    std::optional<double> constraint_tolerance{};
    Convergence convergence{};

    // Convenience accessors for common convergence criteria thresholds.
    // These delegate to the corresponding criterion in the convergence policy.

    template <typename C = Convergence>
    void set_gradient_threshold(double v)
        requires requires(C& c) { std::get<gradient_tolerance_criterion>(c.criteria); }
    {
        std::get<gradient_tolerance_criterion>(convergence.criteria).threshold = v;
    }

    template <typename C = Convergence>
    void set_objective_threshold(double v)
        requires requires(C& c) { std::get<objective_tolerance_criterion>(c.criteria); }
    {
        std::get<objective_tolerance_criterion>(convergence.criteria).threshold = v;
        std::get<objective_tolerance_criterion>(convergence.criteria).stationarity_threshold = v;
    }

    template <typename C = Convergence>
    void set_stationarity_threshold(double v)
        requires requires(C& c) { std::get<objective_tolerance_criterion>(c.criteria); }
    {
        std::get<objective_tolerance_criterion>(convergence.criteria).stationarity_threshold = v;
    }

    template <typename C = Convergence>
    void set_step_threshold(double v)
        requires requires(C& c) { std::get<step_tolerance_criterion>(c.criteria); }
    {
        std::get<step_tolerance_criterion>(convergence.criteria).threshold = v;
    }
};

}

#endif
