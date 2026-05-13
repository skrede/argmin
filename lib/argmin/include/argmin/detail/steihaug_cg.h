#ifndef HPP_GUARD_ARGMIN_DETAIL_STEIHAUG_CG_H
#define HPP_GUARD_ARGMIN_DETAIL_STEIHAUG_CG_H

// Steihaug-Toint truncated CG for the trust-region subproblem
//
//   min  q(p) = g^T p + 0.5 p^T B p     s.t.  ||p|| <= delta
//
// Exit conditions:
//   1. forcing             — residual norm satisfies caller-supplied
//                            forcing tolerance ||r|| <= eps * ||r_0||.
//   2. negative_curvature  — d^T B d <= 0; follow d to the TR boundary.
//   3. boundary            — trial iterate exits the trust region;
//                            truncate to ||p|| = delta along d.
//   4. max_iterations      — inner-iteration cap reached; return last
//                            accepted iterate (the outer ratio test
//                            decides whether to accept or reject).
//
// The Hessian enters as a generic Hessian-vector-product callable so
// this helper stays Hessian-source-blind (consumers wire a
// dense_ldl_bfgs::multiply closure at the call site without dragging
// the BFGS header into this translation unit).
//
// Bounds are passed in displaced form (lower - z_k, upper - z_k) so
// the projection inside the inner loop operates on the step p, not
// the iterate z. On bound activation the loop performs a Lin-More
// free-variable restart: it re-seeds p_k to the projected step,
// recomputes the residual r = g + B p_k, resets d = -r, and continues.
//
// Adopted from: scipy/optimize/_trustregion_constr/qp_subproblem.py
//               (projected_cg; argmin variant: no explicit null-space
//                projector — bounds-projection at each inner iteration
//                via detail::project handles slack + box bounds
//                uniformly on the joint (x, s) space).
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 7.3 Algorithm 7.2 (truncated CG);
//            Steihaug 1983, SIAM J. Numer. Anal. 20(3):626-637;
//            Toint 1981 in Duff (ed.), "Sparse Matrices and Their
//            Uses", Academic Press, 57-88;
//            Conn, Gould, Toint 2000 MOS-SIAM "Trust-Region Methods"
//            Chapter 7 Section 7.5 (Steihaug-Toint termination);
//            Lin and More 1999, SIAM J. Optim. 9(4):1100-1127
//            (free-variable restart on bound activation).
//
// argmin variant: bounds-projection inner loop on the joint (x, s)
//                 space, treating slack lower-bound s >= 0 uniformly
//                 with x's box bounds via the displaced-box
//                 detail::project invocation; the forcing tolerance
//                 is a caller-supplied scalar so per-mode forcing
//                 (Eisenstat-Walker 1996, Dembo-Eisenstat-Steihaug
//                 1982) lives at the policy site, not in this helper.

#include "argmin/types.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace argmin::detail
{

enum class cg_exit_status : std::uint8_t
{
    forcing,
    negative_curvature,
    boundary,
    max_iterations
};

// Steihaug-Toint truncated CG inner loop.
//
// Inputs:
//   g                 — gradient of the quadratic at p = 0.
//   hessian_op        — Hessian-vector-product callable; must accept
//                       a const Eigen::Ref to the step vector and
//                       return an Eigen::Vector of the same dimension
//                       (assignable into Bd_buf).
//   delta             — trust-region radius (positive).
//   eps               — forcing tolerance; the loop terminates when
//                       ||r_k|| <= eps * ||r_0||.
//   lower_displaced   — displaced lower bound (lower - z_k).
//   upper_displaced   — displaced upper bound (upper - z_k).
//   max_iter          — inner-iteration cap.
//   p_out             — caller-owned step output (written on every exit).
//   r_buf, d_buf,
//   Bd_buf            — caller-owned scratch buffers, each pre-sized to
//                       g.size(); the helper allocates no heap memory in
//                       the hot path.
//
// Workspaces follow the same caller-owned-Ref convention as
// equality_feasibility_warmstart in sqp_common.h.
template <typename Scalar, int N = argmin::dynamic_dimension,
          typename HessianOp>
ARGMIN_FORCE_INLINE cg_exit_status steihaug_cg(
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& g,
    HessianOp&& hessian_op,
    Scalar delta,
    Scalar eps,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& lower_displaced,
    const Eigen::Ref<const Eigen::Vector<Scalar, N>>& upper_displaced,
    std::size_t max_iter,
    Eigen::Ref<Eigen::Vector<Scalar, N>> p_out,
    Eigen::Ref<Eigen::Vector<Scalar, N>> r_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> d_buf,
    Eigen::Ref<Eigen::Vector<Scalar, N>> Bd_buf)
{
    using std::sqrt;

    const Scalar delta_sq = delta * delta;

    // p_0 = 0; r_0 = g; d_0 = -g.
    p_out.setZero();
    r_buf = g;
    d_buf.noalias() = -g;

    const Scalar r0_norm_sq = r_buf.squaredNorm();
    const Scalar r0_norm = sqrt(r0_norm_sq);

    // Degenerate stationary-point check. eps * ||r_0|| is the forcing
    // bound; if ||r_0|| already lies below it (including the ||r_0|| == 0
    // case) the zero step is the forcing-exit solution.
    if(r0_norm <= eps * r0_norm)
    {
        // Trivially satisfied when r0_norm == 0; for r0_norm > 0 the
        // condition collapses to (1 - eps) * r0_norm <= 0, i.e. eps >= 1.
        // Both branches return the zero step at the forcing exit.
        if(r0_norm == Scalar(0))
            return cg_exit_status::forcing;
    }

    // Boundary intersection: positive root of
    //   tau^2 ||d||^2 + 2 tau <p, d> + ||p||^2 - delta^2 = 0.
    // Discriminant clamp std::max(0, disc) is load-bearing — it absorbs
    // floating-point noise when ||p|| == delta and d is co-linear with p.
    //
    // Reference: Nocedal and Wright eq. 7.13; Conn, Gould, Toint 2000
    //            §7.5 (the tau-quadratic for Steihaug-Toint).
    const auto tau_to_boundary =
        [delta_sq](const Eigen::Ref<const Eigen::Vector<Scalar, N>>& p_k,
                   const Eigen::Ref<const Eigen::Vector<Scalar, N>>& d_k)
        -> Scalar
    {
        const Scalar d_sq = d_k.squaredNorm();
        if(d_sq <= Scalar(0))
            return Scalar(0);
        const Scalar p_dot_d = p_k.dot(d_k);
        const Scalar p_sq = p_k.squaredNorm();
        const Scalar disc = p_dot_d * p_dot_d - d_sq * (p_sq - delta_sq);
        const Scalar disc_clamped =
            disc > Scalar(0) ? disc : Scalar(0);
        return (-p_dot_d + sqrt(disc_clamped)) / d_sq;
    };

    Scalar r_dot_r = r0_norm_sq;
    const Scalar eps_sq_r0_sq = (eps * eps) * r0_norm_sq;

    for(std::size_t k = 0; k < max_iter; ++k)
    {
        // Bd = B * d_k.
        Bd_buf = hessian_op(d_buf);
        const Scalar kappa = d_buf.dot(Bd_buf);

        // Negative-curvature exit: follow d_k to the TR boundary, then
        // project onto the displaced box. The projection is the
        // bounds-aware analog of the unconstrained Steihaug exit and
        // is consistent with Lin-More restart on activated bounds.
        if(kappa <= Scalar(0))
        {
            const Scalar tau = tau_to_boundary(p_out, d_buf);
            p_out.noalias() += tau * d_buf;
            p_out = detail::project<Scalar, N>(
                p_out, lower_displaced, upper_displaced);
            return cg_exit_status::negative_curvature;
        }

        const Scalar alpha = r_dot_r / kappa;

        // Trust-region boundary truncation. Use squared norm to skip
        // the sqrt on the common-case "still inside" check.
        const Scalar p_trial_norm_sq =
            p_out.squaredNorm()
            + Scalar(2) * alpha * p_out.dot(d_buf)
            + alpha * alpha * d_buf.squaredNorm();
        if(p_trial_norm_sq >= delta_sq)
        {
            const Scalar tau = tau_to_boundary(p_out, d_buf);
            p_out.noalias() += tau * d_buf;
            p_out = detail::project<Scalar, N>(
                p_out, lower_displaced, upper_displaced);
            return cg_exit_status::boundary;
        }

        // Provisional iterate; project on the displaced box. On bound
        // activation perform a Lin-More free-variable restart:
        // re-seed p_k to the projected iterate, recompute the residual
        // at the new point, reset the search direction, and continue.
        Eigen::Vector<Scalar, N> p_trial = p_out + alpha * d_buf;
        Eigen::Vector<Scalar, N> p_proj = detail::project<Scalar, N>(
            p_trial, lower_displaced, upper_displaced);

        if((p_proj - p_trial).squaredNorm() > Scalar(0))
        {
            p_out = p_proj;
            // r = g + B * p_k at the projected iterate.
            r_buf = g;
            Bd_buf = hessian_op(p_out);
            r_buf.noalias() += Bd_buf;
            d_buf.noalias() = -r_buf;
            r_dot_r = r_buf.squaredNorm();

            // Forcing-sequence check at the projected restart point.
            if(r_dot_r <= eps_sq_r0_sq)
                return cg_exit_status::forcing;
            continue;
        }

        // Standard CG step: accept p_trial; update residual via the
        // three-term recurrence r_{k+1} = r_k + alpha * Bd; check the
        // forcing-sequence early-exit; then build the new direction.
        p_out = p_trial;
        r_buf.noalias() += alpha * Bd_buf;
        const Scalar r_new_dot_r_new = r_buf.squaredNorm();

        if(r_new_dot_r_new <= eps_sq_r0_sq)
            return cg_exit_status::forcing;

        const Scalar beta = r_new_dot_r_new / r_dot_r;
        d_buf = -r_buf + beta * d_buf;
        r_dot_r = r_new_dot_r_new;
    }

    return cg_exit_status::max_iterations;
}

}

#endif
