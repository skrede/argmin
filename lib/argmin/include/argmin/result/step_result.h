#ifndef HPP_GUARD_ARGMIN_RESULT_STEP_RESULT_H
#define HPP_GUARD_ARGMIN_RESULT_STEP_RESULT_H

#include "argmin/result/status.h"

#include <cstddef>
#include <optional>

namespace argmin
{

// Metrics from a single solver iteration.
//
// basic_solver inspects these after each Policy::step() to decide
// convergence, stall detection, and divergence.

template <typename Scalar = double>
struct step_result
{
    Scalar objective_value{};
    // Policy-native gradient-like quantity for telemetry/logging.
    // Heterogeneous semantics across policies: raw ||grad f|| for
    // unconstrained gradient methods, ||grad L|| for SQP variants,
    // trust-region model proxy for BOBYQA/COBYLA, sigma * D.maxCoeff
    // for CMA-ES. Convergence criteria should prefer
    // kkt_residual.value_or(gradient_norm) for stationarity gating.
    Scalar gradient_norm{};
    Scalar step_size{};
    Scalar objective_change{};
    bool improved{false};
    // True when the policy took a null/no-move step for algorithmic
    // reasons (SQP zero-step degeneracy, trust-region contraction,
    // restoration exhaustion). Convergence criteria that test
    // step_size must exempt null steps to avoid false stall detection.
    bool is_null_step{false};
    Scalar constraint_violation{};
    Scalar x_norm{};
    // KKT residual: L-infinity norm combining Lagrangian stationarity
    // and complementarity. Populated by gradient-aware policies that
    // maintain multiplier estimates; left as nullopt by derivative-free
    // policies where no multiplier estimate exists. Convergence criteria
    // should prefer kkt_residual.value_or(gradient_norm) for stationarity
    // gating so criterion comparisons remain meaningful across all policies.
    //
    // Reference: N&W 2e Section 12.3 / eq. 12.34 (Lagrangian stationarity).
    std::optional<Scalar> kkt_residual{};
    std::optional<solver_status> policy_status{};

    // Per-step solver diagnostics for telemetry. Defaulted-member
    // sub-struct so future fields can be added without touching
    // existing call sites. Caller composition for line-search SQP
    // policies: is_null_step == true && diagnostics.bfgs_reset_count > 0
    // signals BFGS-reset cap exhaustion.
    //
    // Reference: PITFALLS section L (line-search exhaustion fallback);
    //            NLopt slsqp.c:1890-1895 (ireset retry parity).
    //
    // argmin variant: scalar bfgs_reset_count only;
    //                 a templated / variant per-policy diagnostics
    //                 type is the natural pairing with a future
    //                 step_result type-system redesign.
    struct solver_diagnostics
    {
        std::size_t bfgs_reset_count{0};
        // Fast-mode BFGS-update skip counter. Increments at the policy-level
        // hessian.push() guard whenever the curvature pair has s^T y <= 0
        // and the policy mode dispatches to the skip branch (N&W Procedure
        // 18.2 Powell damping is bypassed in fast mode in favor of leaving
        // the prior B unchanged for wall-time-budgeted contexts).
        //
        // Reference: N&W 2e eq. 18.22-18.24 (Powell damping; accurate-mode
        //            path is preserved unchanged in dense_ldl_bfgs::push).
        std::size_t bfgs_skip_count{0};
    };
    solver_diagnostics diagnostics{};
};

}

#endif
