#ifndef HPP_GUARD_ARGMIN_SOLVER_OPTIONS_H
#define HPP_GUARD_ARGMIN_SOLVER_OPTIONS_H

#include "argmin/solver/convergence.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace argmin
{

// Shared, wall-clock-free solver configuration: convergence policy, iteration
// budget, and the feasibility-first tolerances. This is the common core
// embedded by every driver -- the step-budget driver consumes it directly,
// and the time-budget drivers embed it as a member alongside their own
// wall-clock knobs (a deadline plus a poll cadence). Keeping the clock out of
// this struct is what makes the step-budget path structurally clock-free: a
// translation unit that only budgets by iterations never pulls in the
// standard-library clock header.
//
// The Convergence template parameter is a convergence_policy<Criteria...>
// that composes criterion types via fold expression. Individual tolerance
// thresholds live in the criterion types, not here.

template <typename Convergence = default_convergence>
struct solver_options
{
    using convergence_type = Convergence;

    std::uint32_t max_iterations{1000};
    std::optional<double> constraint_tolerance{};
    // Threshold used by step_budget_solver's best-seen feasibility-first
    // comparator: an iterate with constraint_violation <= this value
    // is treated as feasible when selecting the returned point.
    //
    // Default 1e-6 matches argmin's KKT primal-feasibility convention
    // (detail::kkt_residual L-infinity leg threshold). Callers whose
    // inner solver floors primal feasibility above 1e-6 (derivative-free
    // families, coarse-tolerance AL loops, ...) must widen this per-call
    // via opts.constraint_tolerance (which takes precedence) or by
    // setting opts.feasibility_tolerance directly. The namespace default
    // reflects the numerical convention, not the loosest tolerated test
    // residual.
    //
    // Reference: NLopt nlopt_optimize best-solution-returned convention
    //            (nlopt/src/api/nlopt.c); N&W 2e Definition 12.1
    //            (primal feasibility).
    double feasibility_tolerance{1e-6};
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

    template <typename C = Convergence>
    void set_objective_threshold_rel(double v)
        requires requires(C& c) { std::get<objective_tolerance_rel_criterion>(c.criteria); }
    {
        std::get<objective_tolerance_rel_criterion>(convergence.criteria).threshold = v;
        std::get<objective_tolerance_rel_criterion>(convergence.criteria).stationarity_threshold = v;
    }

    template <typename C = Convergence>
    void set_step_threshold_rel(double v)
        requires requires(C& c) { std::get<step_tolerance_rel_criterion>(c.criteria); }
    {
        std::get<step_tolerance_rel_criterion>(convergence.criteria).threshold = v;
    }

    template <typename C = Convergence>
    void set_stall_threshold(double v)
        requires requires(C& c) { std::get<stall_tolerance_criterion>(c.criteria); }
    {
        std::get<stall_tolerance_criterion>(convergence.criteria).threshold = v;
    }

    template <typename C = Convergence>
    void set_stall_window(std::uint16_t v)
        requires requires(C& c) { std::get<stall_tolerance_criterion>(c.criteria); }
    {
        std::get<stall_tolerance_criterion>(convergence.criteria).window = v;
    }
};

}

#endif
