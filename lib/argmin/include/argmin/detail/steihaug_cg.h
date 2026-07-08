#ifndef HPP_GUARD_ARGMIN_DETAIL_STEIHAUG_CG_H
#define HPP_GUARD_ARGMIN_DETAIL_STEIHAUG_CG_H

// Projected Steihaug-Toint truncated CG for the trust-region subproblem
//
//   min  q(p) = g^T p + 0.5 p^T B p     s.t.  ||p|| <= delta,  A p = 0,
//                                             lower <= p <= upper
//
// Every residual and search direction is projected onto null(A) by the
// caller-supplied projector, so all iterates stay in null(A) by
// construction (Gould-Hribar-Nocedal 2001 Algorithm 6.2; scipy
// projected_cg). Consumers that do not constrain the step to a null
// space (the pure box/TR trust-region subproblem) pass the identity
// projector and recover the ordinary Steihaug-Toint iteration.
//
// Exit conditions:
//   1. forcing             — projected residual norm satisfies the
//                            caller-supplied forcing tolerance
//                            ||Z r|| <= eps * ||Z r_0||.
//   2. negative_curvature  — d^T B d <= 0; follow d to the nearest of
//                            the TR boundary and the blocking box face.
//   3. boundary            — the CG minimizer along d would leave the
//                            trust region or cross a box face; truncate
//                            AT that face/boundary before stepping.
//   4. max_iterations      — inner-iteration cap reached.
//
// Boundary/box handling is truncate-before-step: along the current
// direction the step to the TR boundary and the step to the nearest box
// face are computed, the smaller is taken, and the iterate is truncated
// there BEFORE it is applied. There is no post-step box projection: q is
// monotone along d up to the CG step length (convex case) and for all
// positive step lengths under negative curvature, so truncation
// preserves the Steihaug model-decrease guarantee, whereas clipping an
// overshooting iterate back onto the box does not.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 7.3 Algorithm 7.2 (truncated CG) and Section 16.3
//            Algorithm 16.2 (projected CG);
//            Gould, Hribar, Nocedal 2001 SIAM J. Sci. Comput. 23(6)
//            Algorithm 6.2 (projected preconditioned CG; Polak-Ribiere
//            beta); scipy optimize/_trustregion_constr/qp_subproblem.py
//            (projected_cg; box/spherical truncation);
//            Steihaug 1983 SIAM J. Numer. Anal. 20(3):626-637;
//            Conn, Gould, Toint 2000 MOS-SIAM Chapter 7 Section 7.5
//            (Steihaug-Toint termination; model-decrease on truncation).

#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace argmin::detail
{

enum class cg_exit_status : std::uint8_t
{
    forcing,
    negative_curvature,
    boundary,
    max_iterations
};

// Default (identity) projector: leaves its argument unchanged. Consumers
// that solve the unconstrained-in-null-space subproblem (no A p = 0
// side constraint) pass this and recover the ordinary Steihaug-Toint
// iteration.
struct identity_null_projector
{
    template <typename Derived>
    void operator()(Eigen::MatrixBase<Derived>&) const
    {
    }
};

// Projected Steihaug-Toint truncated CG inner loop.
//
// Inputs mirror the previous helper, with two additions:
//   project_null   — callable invoked as project_null(v) that projects
//                    the Eigen vector v onto null(A) in place. Defaults
//                    to the identity (no side constraint).
//   blocking_out   — optional; when non-null it receives the index of
//                    the box coordinate that truncated the step (a box
//                    face was hit strictly before the TR boundary), or
//                    -1 if the exit was a TR-boundary / interior event.
//                    Consumers implementing a free-set restart read it;
//                    the default box-face exit ignores it.
//
// r_buf holds the PROJECTED residual Z r (not the full residual); the
// three-buffer caller-owned scratch convention is unchanged.
template <typename Scalar, int N = argmin::dynamic_dimension,
          typename HessianOp,
          typename Projector = identity_null_projector>
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
    Eigen::Ref<Eigen::Vector<Scalar, N>> Bd_buf,
    Projector project_null = {},
    int* blocking_out = nullptr,
    bool warm_start = false)
{
    using std::sqrt;

    const Scalar delta_sq = delta * delta;
    const Scalar inf = std::numeric_limits<Scalar>::infinity();
    if(blocking_out != nullptr)
        *blocking_out = -1;

    // Residual buffer holds the PROJECTED residual Z r; d_0 = -Z r_0
    // stays in null(A). Cold start begins at p_0 = 0 (r_0 = g); warm
    // start (Lin-More free-set restart) continues from the current
    // p_out with r_0 = g + B p_out, so a coordinate pinned by the
    // projector keeps its already-reached box-face value.
    if(warm_start)
    {
        // hessian_op writes B * p_out into the caller-owned Bd_buf (not
        // yet consumed by the CG loop), avoiding a return-by-value temporary.
        hessian_op(p_out, Bd_buf);
        r_buf.noalias() = g + Bd_buf;
    }
    else
    {
        p_out.setZero();
        r_buf = g;
    }
    project_null(r_buf);
    d_buf.noalias() = -r_buf;

    // rg = ||Z r||^2 = r^T Z r (Z symmetric idempotent). The forcing
    // test and the CG scalars are driven by this projected quantity.
    Scalar rg = r_buf.squaredNorm();
    const Scalar rg0 = rg;
    if(rg0 <= Scalar(0))
        return cg_exit_status::forcing;  // stationary within null(A).
    const Scalar eps_sq_rg0 = (eps * eps) * rg0;

    // Step to the TR boundary along d: positive root of
    //   tau^2 ||d||^2 + 2 tau <p, d> + ||p||^2 - delta^2 = 0.
    // Reference: N&W eq. 7.13; Conn-Gould-Toint 2000 §7.5.
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
        const Scalar disc_clamped = disc > Scalar(0) ? disc : Scalar(0);
        return (-p_dot_d + sqrt(disc_clamped)) / d_sq;
    };

    // Step to the nearest displaced box face along d (componentwise
    // ratio test); returns {tau, blocking coordinate}. A coordinate
    // already at/beyond its face in the direction of motion yields
    // tau = 0 (the step is blocked immediately).
    const auto tau_to_face =
        [&lower_displaced, &upper_displaced, inf](
            const Eigen::Ref<const Eigen::Vector<Scalar, N>>& p_k,
            const Eigen::Ref<const Eigen::Vector<Scalar, N>>& d_k)
        -> std::pair<Scalar, int>
    {
        Scalar tau = inf;
        int coord = -1;
        for(int i = 0; i < d_k.size(); ++i)
        {
            Scalar t;
            if(d_k[i] > Scalar(0) && upper_displaced[i] < inf)
                t = (upper_displaced[i] - p_k[i]) / d_k[i];
            else if(d_k[i] < Scalar(0) && lower_displaced[i] > -inf)
                t = (lower_displaced[i] - p_k[i]) / d_k[i];
            else
                continue;
            if(t < Scalar(0))
                t = Scalar(0);
            if(t < tau)
            {
                tau = t;
                coord = i;
            }
        }
        return {tau, coord};
    };

    for(std::size_t k = 0; k < max_iter; ++k)
    {
        hessian_op(d_buf, Bd_buf);
        const Scalar kappa = d_buf.dot(Bd_buf);

        const Scalar tau_tr = tau_to_boundary(p_out, d_buf);
        const auto [tau_box, coord] = tau_to_face(p_out, d_buf);
        const Scalar tau_block = tau_box < tau_tr ? tau_box : tau_tr;

        // Negative-curvature exit: q decreases without bound along d, so
        // step as far as the trust region and box permit, then stop.
        if(kappa <= Scalar(0))
        {
            p_out.noalias() += tau_block * d_buf;
            if(blocking_out != nullptr && tau_box < tau_tr)
                *blocking_out = coord;
            return cg_exit_status::negative_curvature;
        }

        const Scalar alpha = rg / kappa;

        // Boundary/box truncation: if the CG minimizer along d would
        // leave the feasible region, truncate at the blocking face
        // BEFORE stepping (no post-step projection).
        if(alpha >= tau_block)
        {
            p_out.noalias() += tau_block * d_buf;
            if(blocking_out != nullptr && tau_box < tau_tr)
                *blocking_out = coord;
            return cg_exit_status::boundary;
        }

        // Interior CG step.
        p_out.noalias() += alpha * d_buf;

        // Update the projected residual in place:
        //   g_{k+1} = Z r_{k+1} = Z r_k + alpha Z (B d_k)
        //           = r_buf + alpha * project(B d_k).
        // r_buf already holds Z r_k; project B d_k and accumulate.
        project_null(Bd_buf);
        r_buf.noalias() += alpha * Bd_buf;
        const Scalar rg_new = r_buf.squaredNorm();

        if(rg_new <= eps_sq_rg0)
            return cg_exit_status::forcing;

        const Scalar beta = rg_new / rg;
        d_buf = -r_buf + beta * d_buf;  // stays in null(A).
        rg = rg_new;
    }

    return cg_exit_status::max_iterations;
}

}

#endif
