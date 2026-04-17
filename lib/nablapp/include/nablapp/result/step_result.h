#ifndef HPP_GUARD_NABLAPP_RESULT_STEP_RESULT_H
#define HPP_GUARD_NABLAPP_RESULT_STEP_RESULT_H

#include "nablapp/result/status.h"

#include <optional>

namespace nablapp
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
};

}

#endif
