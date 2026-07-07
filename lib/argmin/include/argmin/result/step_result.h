#ifndef HPP_GUARD_ARGMIN_RESULT_STEP_RESULT_H
#define HPP_GUARD_ARGMIN_RESULT_STEP_RESULT_H

#include "argmin/result/status.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace argmin
{

// Metrics from a single solver iteration.
//
// step_budget_solver inspects these after each Policy::step() to decide
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

    // Objective/merit evaluations performed during this single step, e.g.
    // the trial-point evaluations a line search makes. Defaults to 1 --
    // every real step evaluates the objective at least once. step_budget_solver
    // accumulates this across steps so solve_result::function_evaluations
    // reports the genuine evaluation count instead of aliasing the
    // iteration count. Policies that perform inner evaluation loops (line
    // searches, restoration phases) should report the true count here.
    std::uint32_t evaluations{1};

    // Per-step solver diagnostics for telemetry. Defaulted-member
    // sub-struct so future fields can be added without touching
    // existing call sites. Caller composition for line-search SQP
    // policies: is_null_step == true && diagnostics.bfgs_reset_count > 0
    // signals BFGS-reset cap exhaustion.
    //
    // Reference: NLopt slsqp.c slsqpb_ outer loop (line-search exhaustion
    //            fallback);
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
        // Armijo NaN/Inf recovery counter. Increments whenever the
        // Armijo backtracker — or any policy's hand-rolled inline
        // backtracking loop — observes a non-finite trial-iterate
        // evaluation and responds by shrinking alpha and continuing
        // rather than propagating the tainted value through the merit
        // comparison. Aggregated across the main line search and any
        // second-order-correction retry. Both modes enable the gate —
        // non-finite trial-point evaluations should never be silently
        // consumed.
        //
        // NaN/Inf gate: see argmin/line_search/armijo.h header comment.
        std::size_t nan_eval_count{0};
        // Second-order correction retry counter for trust-region SQP.
        // Increments once per SOC retry attempt at the composite-step
        // rejection site (regardless of whether the retry succeeds). Zero
        // for any policy that does not invoke a SOC retry. The retry
        // shape is the trust-region analog of the kraft_slsqp_policy
        // Maratos correction: on a rejected primary step, recompute the
        // linearized constraint residual at the trial point with the
        // original Jacobian and re-call the composite-step helper with
        // the corrected RHS.
        //
        // Reference: Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
        //            8(3):682-706 Section 3.1 (v-optimal restoration);
        //            Nocedal and Wright 2e Section 18.3 (Maratos effect
        //            and second-order correction; line-search analog at
        //            kraft_slsqp_policy.h Section 2.2.4).
        std::size_t soc_retry_count{0};
        // Feasibility-restoration inner-iteration counter for the LM
        // helper invoked from the filter trust-region SQP policy's
        // reject-after-radius-collapse branch. Increments by the helper's
        // returned iterations_used count whenever restoration fires;
        // populated on both the converged emit path (when restoration
        // restored a feasible iterate and the policy resumed composite
        // step) and the fall-through trust-radius-collapse null-step
        // emit path (when restoration ran but did not converge). Zero
        // for any policy that does not invoke restoration, including
        // filter_trsqp at its default restoration_max_iter = 0.
        //
        // Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-
        //            Marquardt for least-squares);
        //            Wachter and Biegler 2006 Math. Programming
        //            106:25-57 Section 3.3 (IPOPT restoration phase;
        //            argmin variant ships a minimal-viable LM
        //            simplification).
        std::size_t restoration_iters_used{0};
    };
    solver_diagnostics diagnostics{};
};

}

#endif
