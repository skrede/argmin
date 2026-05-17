#ifndef HPP_GUARD_ARGMIN_DETAIL_RESTORATION_H
#define HPP_GUARD_ARGMIN_DETAIL_RESTORATION_H

// Levenberg-Marquardt feasibility-restoration helper.
//
//   minimize  1/2 ||c(x)||_2^2
//   subject to  lower <= x <= upper
//
// Per outer iter:
//   1. Evaluate c(x) and J(x).
//   2. Solve (J^T J + lambda I) dx = -J^T c via Eigen LDLT on the
//      symmetric positive-semidefinite regularized normal-equation
//      system; the caller-supplied LDLT factor is reused across iters
//      (recomputed in place from the regularized JtJ).
//   3. Project x + dx onto the box [lower, upper] component-wise.
//   4. Acceptance is feasibility-only:
//        if ||c(x + dx)||_2 < ||c(x)||_2:
//            accept; x := x + dx; lambda := max(lambda_min, lambda / 2)
//        else:
//            reject; lambda := min(lambda_max, lambda * 4)
//            if lambda hit lambda_max with no progress:
//                return lambda_grew_unbounded
//   5. If ||c(x)||_2 < feasibility_tol: return converged.
//   6. Loop until iter == max_iter.
//
// Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-Marquardt
//            for least-squares; the algorithmic template);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            Section 3.3 (IPOPT restoration phase; full version);
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
//            8(3):682-706 Section 3.1 (v-optimal restoration; the
//            structurally similar alternative).
//
// argmin variant: minimal-viable prototype. OMITS the slack-augmented
//                 reformulation, the multiplier-restoration leg, the
//                 second-order correction at the inner step, and the
//                 filter-set integration during restoration that the
//                 full IPOPT version uses. The caller owns the
//                 constraint and jacobian callbacks and all workspace
//                 buffers (no allocation in the hot loop). The helper
//                 mutates x in place; on success x is the restored
//                 (feasible or maximally-improved) iterate and the
//                 caller's policy resumes its outer loop at the
//                 returned x with whatever state-reset semantics it
//                 chooses.

#include "argmin/types.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <utility>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace argmin::detail
{

// Lambda bounds for the LM regularizer. Literature defaults: lambda_min
// is small enough that the regularized step recovers the Gauss-Newton
// step in the well-conditioned regime; lambda_max is the divergence
// guard (a step of magnitude ||J^T c|| / lambda_max becomes
// machine-precision small for any reasonable ||J^T c||).
//
// Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-Marquardt
//            damping bracket).
inline constexpr double restoration_lambda_min = 1e-12;
inline constexpr double restoration_lambda_max = 1e8;

// LM lambda update factors (literature defaults; symmetric expansion /
// contraction). The 2x shrink on accept and 4x grow on reject is the
// canonical Marquardt schedule; downstream sweeps may revisit on a
// per-cell basis but the helper exposes no knobs for these factors
// at this prototype's surface.
inline constexpr double restoration_lambda_shrink = 0.5;
inline constexpr double restoration_lambda_grow   = 4.0;

// Status returned by feasibility_restoration. The caller's policy
// dispatches on the value:
//   converged              -- ||c||_2 < feasibility_tol on exit
//   lambda_grew_unbounded  -- lambda hit lambda_max without further
//                             progress; LM regularization can no
//                             longer be increased to stabilize the
//                             step
//   max_iter_reached       -- iter == max_iter with ||c|| still above
//                             tolerance; not necessarily a failure
//                             (progress may have been made)
//   degenerate_no_progress -- J^T c == 0 with infeasible c, or the
//                             LDLT factorization signaled
//                             non-success; the LM step is undefined
enum class restoration_status : std::uint8_t
{
    converged,
    lambda_grew_unbounded,
    max_iter_reached,
    degenerate_no_progress
};

// POD result returned by feasibility_restoration. Carries the status
// classification, the iteration count used (for diagnostics roll-up
// at the caller), the final feasibility residual norm, and the final
// lambda value (for sweep-time tuning of lambda_init).
struct lm_restoration_result
{
    restoration_status status;
    std::size_t        iterations_used;
    double             final_c_norm_l2;
    double             final_lambda;
};

// Levenberg-Marquardt feasibility restoration.
//
// ConstraintFn signature: void(const Eigen::Vector<Scalar, N>&,
//                              Eigen::VectorXd&)
//   -- first argument x, second argument the output residual buffer
//      (sized m_total). The caller's policy passes a closure over its
//      problem-binding state; the helper writes the residual into the
//      caller-supplied buffer reference.
// JacobianFn signature: void(const Eigen::Vector<Scalar, N>&,
//                            Eigen::MatrixXd&)
//   -- first argument x, second argument the output Jacobian (sized
//      m_total x n). Same closure-binding convention.
//
// The helper places (J^T J + lambda I) into JtJ_buf, factorizes it
// into ldlt_buf, and solves the regularized normal-equation system
// for dx_buf. The caller owns every buffer; no allocation happens in
// the inner loop. Sized expectations:
//   c_buf       -- m_total
//   c_trial_buf -- m_total
//   J_buf       -- m_total x n
//   JtJ_buf     -- n x n
//   rhs_buf     -- n
//   dx_buf      -- n
//   x_trial_buf -- n
//   ldlt_buf    -- pre-sized to n via Eigen::LDLT<MatrixXd>(n) at the
//                  caller's allocation site.
template <typename Scalar, int N, typename ConstraintFn,
          typename JacobianFn>
lm_restoration_result feasibility_restoration(
    Eigen::Ref<Eigen::Vector<Scalar, N>>              x,
    ConstraintFn&&                                    constraints,
    JacobianFn&&                                      jacobian,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper,
    std::size_t                                       max_iter,
    Scalar                                            lambda_init,
    Scalar                                            feasibility_tol,
    Eigen::VectorXd&                                  c_buf,
    Eigen::VectorXd&                                  c_trial_buf,
    Eigen::MatrixXd&                                  J_buf,
    Eigen::MatrixXd&                                  JtJ_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>>              rhs_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>>              dx_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>>              x_trial_buf,
    Eigen::LDLT<Eigen::MatrixXd>&                     ldlt_buf)
{
    using std::min;
    using std::max;

    lm_restoration_result result{};
    result.status          = restoration_status::max_iter_reached;
    result.iterations_used = 0;
    result.final_c_norm_l2 =
        std::numeric_limits<double>::quiet_NaN();
    result.final_lambda = static_cast<double>(lambda_init);

    // Initial residual evaluation. An exit on the entry feasibility
    // check is the cleanest no-op behavior for a restoration call on
    // an already-feasible iterate.
    constraints(x, c_buf);
    double c_norm = c_buf.norm();
    result.final_c_norm_l2 = c_norm;

    if(c_norm < static_cast<double>(feasibility_tol))
    {
        result.status = restoration_status::converged;
        return result;
    }

    Scalar lambda = lambda_init;

    for(std::size_t iter = 0; iter < max_iter; ++iter)
    {
        // ── J(x), rhs = -J^T c, JtJ = J^T J ────────────────────────
        jacobian(x, J_buf);

        // rhs = -J^T c. Caller-owned buffer; noalias is safe because
        // J_buf and c_buf are distinct allocations from rhs_buf.
        rhs_buf.noalias() = -J_buf.transpose() * c_buf;

        // Degeneracy short-circuit: when J^T c == 0 and c != 0, no LM
        // step can reduce ||c|| (the regularized solve returns dx = 0).
        // This avoids a pointless LDLT factorization on a known-stuck
        // configuration.
        if(rhs_buf.template lpNorm<Eigen::Infinity>() == Scalar(0))
        {
            result.status = restoration_status::degenerate_no_progress;
            result.iterations_used = iter;
            result.final_c_norm_l2 = c_norm;
            result.final_lambda    = static_cast<double>(lambda);
            return result;
        }

        // JtJ = J^T J. Symmetric semidefinite; LDLT handles
        // rank-deficient cases as long as lambda > 0.
        JtJ_buf.noalias() = J_buf.transpose() * J_buf;

        // Try the LM solve at the current lambda; on accept, shrink
        // lambda and commit x; on reject, grow lambda and retry the
        // factorization inside the same outer iter. We bound the
        // inner reject sequence by lambda_max -- once lambda hits the
        // cap and the step still fails to reduce ||c||, return
        // lambda_grew_unbounded.
        //
        // Allocation-free loop body: the lambda diagonal regularizer
        // is applied directly to JtJ_buf in place via a delta-add
        // (`diagonal += new_lambda - applied_lambda`) so JtJ_buf
        // carries the correct (J^T J + lambda I) at the top of every
        // LDLT compute without ever copying the full n x n matrix.
        // `lambda_applied` is reset to 0 once per outer iter to match
        // the fresh JtJ = J^T J written at the top of the outer body.
        // The LDLT solve writes into the caller-owned dx_buf via
        // .noalias() to suppress the temporary that bare
        // `dx_buf = ldlt_buf.solve(rhs_buf)` would otherwise
        // materialize. Net: no heap traffic inside the
        // while(!inner_accepted) block, matching the helper's
        // documented contract.
        bool inner_accepted = false;
        Scalar lambda_applied = Scalar(0);
        while(!inner_accepted)
        {
            // Add the lambda I regularizer in place. The diagonal
            // adjustment is the standard LM perturbation; the LDLT
            // factorization picks up the regularized matrix. On the
            // next inner trial (lambda grew) we first undo the prior
            // diagonal bump, then apply the new one, so JtJ_buf
            // alternates between the bare J^T J and the regularized
            // form without ever copying the full n x n matrix.
            JtJ_buf.diagonal().array() += (lambda - lambda_applied);
            lambda_applied = lambda;
            ldlt_buf.compute(JtJ_buf);

            if(ldlt_buf.info() != Eigen::Success)
            {
                // Factorization failed despite the diagonal
                // regularization (extreme numerical conditioning).
                // Grow lambda and retry; if we are already at the
                // cap, declare lambda_grew_unbounded.
                if(lambda >= static_cast<Scalar>(restoration_lambda_max))
                {
                    result.status =
                        restoration_status::lambda_grew_unbounded;
                    result.iterations_used = iter;
                    result.final_c_norm_l2 = c_norm;
                    result.final_lambda    =
                        static_cast<double>(lambda);
                    return result;
                }
                lambda = min(static_cast<Scalar>(
                                 restoration_lambda_max),
                             lambda
                                 * static_cast<Scalar>(
                                     restoration_lambda_grow));
                continue;
            }

            // Solve into dx_buf. LDLT::solve returns an
            // Eigen::Solve expression; assigning through .noalias()
            // evaluates it directly into the caller-owned dx_buf
            // without materializing an intermediate VectorXd. Safe
            // because dx_buf does not alias rhs_buf (separate
            // caller-owned allocations per the helper contract).
            dx_buf.noalias() = ldlt_buf.solve(rhs_buf);

            if(!dx_buf.allFinite())
            {
                // Non-finite step propagation guard. Treat as an LM
                // reject and grow lambda.
                if(lambda >= static_cast<Scalar>(restoration_lambda_max))
                {
                    result.status =
                        restoration_status::lambda_grew_unbounded;
                    result.iterations_used = iter;
                    result.final_c_norm_l2 = c_norm;
                    result.final_lambda    =
                        static_cast<double>(lambda);
                    return result;
                }
                lambda = min(static_cast<Scalar>(
                                 restoration_lambda_max),
                             lambda
                                 * static_cast<Scalar>(
                                     restoration_lambda_grow));
                continue;
            }

            // Box-project x + dx onto [lower, upper]. The helper
            // returns the projected vector; we materialize it into
            // x_trial_buf which is a caller-owned scratch.
            x_trial_buf =
                (x + dx_buf).cwiseMax(lower).cwiseMin(upper).eval();

            // Trial residual at the projected point.
            constraints(x_trial_buf, c_trial_buf);
            const double c_trial_norm = c_trial_buf.norm();

            if(c_trial_norm < c_norm)
            {
                // Accept: commit x, shrink lambda, refresh c_buf.
                x      = x_trial_buf;
                c_buf  = c_trial_buf;
                c_norm = c_trial_norm;
                lambda = max(static_cast<Scalar>(
                                 restoration_lambda_min),
                             lambda
                                 * static_cast<Scalar>(
                                     restoration_lambda_shrink));
                inner_accepted = true;
            }
            else
            {
                // Reject: grow lambda. If we are already at the cap,
                // declare lambda_grew_unbounded.
                if(lambda >= static_cast<Scalar>(restoration_lambda_max))
                {
                    result.status =
                        restoration_status::lambda_grew_unbounded;
                    result.iterations_used = iter;
                    result.final_c_norm_l2 = c_norm;
                    result.final_lambda    =
                        static_cast<double>(lambda);
                    return result;
                }
                lambda = min(static_cast<Scalar>(
                                 restoration_lambda_max),
                             lambda
                                 * static_cast<Scalar>(
                                     restoration_lambda_grow));
                // Inner retry continues the while loop with the
                // grown lambda; no fresh J(x) needed because x has
                // not moved on a reject.
            }
        }

        // Iteration committed; check feasibility.
        if(c_norm < static_cast<double>(feasibility_tol))
        {
            result.status          = restoration_status::converged;
            result.iterations_used = iter + 1;
            result.final_c_norm_l2 = c_norm;
            result.final_lambda    = static_cast<double>(lambda);
            return result;
        }
    }

    // max_iter reached without converging.
    result.status          = restoration_status::max_iter_reached;
    result.iterations_used = max_iter;
    result.final_c_norm_l2 = c_norm;
    result.final_lambda    = static_cast<double>(lambda);
    return result;
}

}

#endif
