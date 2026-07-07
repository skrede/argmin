#ifndef HPP_GUARD_ARGMIN_SOLVER_CONVERGENCE_H
#define HPP_GUARD_ARGMIN_SOLVER_CONVERGENCE_H

#include "argmin/result/step_result.h"
#include "argmin/result/status.h"

#include <array>
#include <cmath>
#include <tuple>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin
{

struct gradient_tolerance_criterion
{
    // Direct-value literature default (no optional/inertness): a
    // default-constructed criterion is active out of the box.
    //
    // 1e-5 matches the pgtol convergence test of Byrd, Lu, Nocedal, Zhu
    // (1995), "A Limited Memory Algorithm for Bound Constrained
    // Optimization," SIAM J. Sci. Comput. 16(5):1190-1208 -- the
    // reference L-BFGS-B implementation's default gradient-norm stopping
    // tolerance, widely adopted as the conventional first-order
    // stationarity gate (e.g. scipy.optimize.fmin_l_bfgs_b default
    // pgtol=1e-5). Empirically swept against the Rosenbrock/HS regression
    // suite (see plan summary): fires well after L-BFGS-B reaches
    // machine-zero on Rosenbrock and does not trip prematurely on the HS
    // suite's constrained problems.
    double threshold{1e-5};

    // Stationarity is gated on the composite KKT residual (Lagrangian
    // stationarity + primal feasibility + dual feasibility +
    // complementarity) when populated by the policy, with a fallback to
    // gradient_norm for unconstrained / derivative-free policies that
    // leave kkt_residual nullopt. Under correct Lagrangian-gradient
    // reporting on constrained SQP policies, ||grad L|| can vanish at
    // intermediate iterates that are not yet KKT points (the multiplier
    // closure can drive the stationarity leg to zero while primal or
    // dual feasibility legs are still large). Gating on raw gradient_norm
    // there fires premature converged at non-KKT iterates; the composite
    // residual is the correct convergence quantity.
    //
    // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
    //            Definition 12.1 (KKT first-order optimality conditions);
    //            eq. 12.34 (Lagrangian stationarity leg of the
    //            composite measure).
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t /*iteration*/) const
    {
        const double kkt = r.kkt_residual.value_or(r.gradient_norm);
        if(kkt < threshold)
            return solver_status::converged;
        return std::nullopt;
    }
};

struct objective_tolerance_criterion
{
    // Direct-value literature default: 1e-10 matches the ftol convention
    // of Dennis and Schnabel's machine-precision-scaled function-value
    // stopping test (reproduced in N&W 2e Section 2.2), consistent with
    // the MINPACK default-tolerance convention (More, Garbow, Hillstrom
    // 1980, "User Guide for MINPACK-1," ANL-80-74) of sqrt(machine
    // epsilon)-to-eps^(2/3) for double precision (eps ~ 2.22e-16).
    // Empirically swept (see plan summary) against Rosenbrock/HS.
    double threshold{1e-10};
    // Direct-value literature default: 1e-8 matches N&W 2e Definition
    // 12.1's typical first-order KKT stationarity gate.
    double stationarity_threshold{1e-8};
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
        if(iteration > 1 && std::abs(r.objective_change) < threshold)
        {
            const double kkt = r.kkt_residual.value_or(r.gradient_norm);
            if(kkt < stationarity_threshold)
                return solver_status::ftol_reached;
        }
        return std::nullopt;
    }
};

struct step_tolerance_criterion
{
    // Direct-value literature default: 1e-8 matches the xtol convention
    // of sqrt(machine epsilon) (~1.49e-8 for double precision) used by
    // MINPACK / Dennis and Schnabel-style step-length stopping tests
    // (More, Garbow, Hillstrom 1980, "User Guide for MINPACK-1,"
    // ANL-80-74; N&W 2e Section 2.2). Empirically swept (see plan
    // summary) against Rosenbrock/HS.
    double threshold{1e-8};
    double feasibility_tolerance{1e-6};

    // A null step (step_size == 0 because the policy intentionally made
    // no move -- SQP zero-step degeneracy, trust-region rho contraction,
    // restoration exhaustion) is not a stall: it is an algorithmic
    // signal that the policy needs more iterations, not fewer. Exempting
    // null steps here avoids false stall detection on iteration 2 for
    // any policy that sets is_null_step.
    //
    // A small step_size at an INFEASIBLE iterate is also not a converged
    // stall: the SQP merit function can have local minima at infeasible
    // points (Maratos-effect trap on HS071: nw_sqp's L1 merit accepts an
    // iter-0 direction that satisfies linearized constraints but
    // nonlinearly violates the inequality, then collapses to step_size <
    // 1e-10 while c_ineq remains at -6.5). Reporting `stalled` there
    // implies "we got to a stationary point" -- the iterate is far from
    // any KKT point. Gate on r.constraint_violation; let stall_tolerance
    // (which uses the combined obj + cv metric) and max_iterations carry
    // termination in that case. Unconstrained policies leave
    // constraint_violation at the default 0.0, so the gate is satisfied
    // trivially for them.
    //
    // Reference: N&W 2e Section 18.4 (SQP convergence analysis);
    //            N&W 2e Section 18.3 (Maratos effect).
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(r.is_null_step)
            return std::nullopt;
        if(r.constraint_violation > feasibility_tolerance)
            return std::nullopt;
        if(iteration > 1 && r.step_size < threshold)
            return solver_status::stalled;
        return std::nullopt;
    }
};

// Relative function decrease test (N&W 2e Section 2.2; K&W 2e Section 2.3)
struct objective_tolerance_rel_criterion
{
    // No literature default: this is the NLopt-style ftol_rel override
    // (see slsqp_compatible_convergence below) rather than a member of
    // default_convergence, so it stays optional/inert until the caller
    // opts in via set_objective_threshold_rel(...).
    std::optional<double> threshold{};
    // Direct-value literature default: 1e-8, matching the absolute
    // objective_tolerance_criterion's stationarity_threshold convention
    // (N&W 2e Definition 12.1).
    double stationarity_threshold{1e-8};
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
            const double kkt = r.kkt_residual.value_or(r.gradient_norm);
            if(kkt < stationarity_threshold)
                return solver_status::ftol_reached;
        }
        return std::nullopt;
    }
};

// Relative step length test (N&W 2e Section 2.2; K&W 2e Section 2.3)
struct step_tolerance_rel_criterion
{
    std::optional<double> threshold{};
    double feasibility_tolerance{1e-6};

    // Null steps and infeasible iterates are exempt for the same reasons
    // as in step_tolerance_criterion. See step_tolerance_criterion::check
    // for the detailed rationale on both gates (xtol_reached at an
    // infeasible iterate would falsely report convergence to a non-KKT
    // point).
    //
    // Reference: N&W 2e Section 18.4 (SQP convergence analysis);
    //            N&W 2e Section 18.3 (Maratos effect).
    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;
        if(r.is_null_step)
            return std::nullopt;
        if(r.constraint_violation > feasibility_tolerance)
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
    // No literature default: stall detection is a window/scale-dependent
    // safety net (auto-enabled per-policy via forward_policy_hints, e.g.
    // the MMA/GCMMA stall_tolerance_threshold option), not a universal
    // stopping test, so it stays optional/inert until a caller or policy
    // opts in.
    std::optional<double> threshold{};
    std::uint16_t window{50};
    bool track_best_seen_feasible{false};
    // Feasibility floor for the best-seen-feasible metric below. Direct-
    // value default matching the project's KKT primal-feasibility
    // convention (solver_options::feasibility_tolerance; N&W 2e
    // Definition 12.1), configurable per-criterion instead of a hardcoded
    // local constant.
    double feasibility_tolerance{1e-6};

    std::optional<solver_status> check(const step_result<double>& r,
                                       std::uint32_t iteration) const
    {
        if(!threshold)
            return std::nullopt;

        auto effective_window = std::min(window, max_window);

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence convergence precondition
        // requires finite-iteration termination detection.  The
        // best_seen_feasible metric is robust against the oscillation mode
        // documented in in-tree diagnosis traces (HS043 objective oscillates
        // with amplitude ~2.75 at move_limit=0.9 while the best-seen iterate
        // is reached at outer iter ~4 and never improved), because the best-
        // seen metric is monotone-non-increasing by construction.  Policies
        // that don't opt in fall through to the combined
        // (objective + constraint_violation) metric byte-for-byte.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes", IJNME
        //            24:359-373, Theorem 5.1 (Cauchy-sequence convergence).
        double metric;
        if(track_best_seen_feasible)
        {
            if(r.constraint_violation <= feasibility_tolerance
               && r.objective_value < best_seen_feasible_)
                best_seen_feasible_ = r.objective_value;
            metric = best_seen_feasible_;
        }
        else
        {
            metric = r.objective_value + r.constraint_violation;
        }
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
    mutable double best_seen_feasible_{std::numeric_limits<double>::infinity()};
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
    // the criterion did not fire. Reachable from step_budget_solver via
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
// IK-style workloads should instantiate step_budget_solver with this alias
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
