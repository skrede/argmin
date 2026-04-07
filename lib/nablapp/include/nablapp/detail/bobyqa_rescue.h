#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_RESCUE_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_RESCUE_H

// Rescue procedure for BOBYQA (Powell 2009).
//
// Rebuilds the interpolation system from scratch when the H-matrix
// denominators have collapsed, indicating a degenerate interpolation
// set. The procedure preserves x_opt and replaces distant/degenerate
// points with new function evaluations along coordinate directions
// from x_opt. BMAT, ZMAT, GOPT, HQ, and PQ are recomputed.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.

#include "nablapp/detail/bobyqa_model.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp::detail
{

// RESCUE: rebuild interpolation system from scratch (Powell 2009).
//
// When the interpolation denominators collapse, the H-matrix inverse
// factorization (BMAT/ZMAT) becomes unreliable. This procedure:
//   1. Preserves x_opt and its function value
//   2. Replaces all other points with new evaluations along coordinate
//      directions from x_opt, spaced at the current rho
//   3. Rebuilds BMAT/ZMAT from scratch using the new point set
//   4. Recomputes GOPT, HQ, PQ from interpolation conditions
//
// The eval_fn callback evaluates f(x) at new points.
//
// Reference: Powell 2009.
template <typename Scalar, int N, int NPT>
void rescue(
    bobyqa_model<Scalar, N, NPT>& model,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    auto&& eval_fn)
{
    const int n = model.gopt.size();
    const int npt = model.pq.size();
    const Scalar rho = model.rho;

    // Step 1: Shift base to x_opt for numerical stability.
    //
    // After shift, x_opt = 0 and all xpt rows are relative to the
    // current best point. Reference: Powell 2009.
    if(model.x_opt.norm() > Scalar(1e-10) * rho)
        model.shift_base(n);

    // Step 2: Build fresh interpolation set centered at x_base (= old x_base + x_opt).
    //
    // Point 0 is x_base itself (x_opt = 0 after shift).
    // Points 1..n: x_base + rho * e_i (adjusted for bounds).
    // Points n+1..2n: x_base - rho * e_i (adjusted for bounds).
    //
    // This is the same layout as the initial PRELIM step, but centered
    // at the current best point and using the current rho.
    Eigen::Vector<Scalar, N> steps(n);
    for(int i = 0; i < n; ++i)
    {
        Scalar range = upper[i] - lower[i];
        steps[i] = std::min(rho, range / Scalar(2));
    }

    // Keep x_opt function value (point 0 = origin).
    model.xpt.setZero();
    model.fval[0] = model.f_opt;

    for(int i = 0; i < n; ++i)
    {
        Scalar step = steps[i];

        // Positive direction.
        Scalar pos = model.x_base[i] + step;
        if(pos > upper[i])
            pos = model.x_base[i] - step;
        model.xpt(1 + i, i) = pos - model.x_base[i];

        // Negative direction.
        Scalar neg = model.x_base[i] - step;
        if(neg < lower[i])
            neg = model.x_base[i] + step;
        if(std::abs((neg - model.x_base[i]) - model.xpt(1 + i, i))
               < Scalar(1e-15) * step)
            neg = model.x_base[i] + Scalar(0.5) * step;
        model.xpt(1 + n + i, i) = neg - model.x_base[i];
    }

    // Step 3: Evaluate function at new points.
    for(int k = 1; k < npt; ++k)
    {
        Eigen::Vector<Scalar, N> pt = model.x_base + model.xpt.row(k).transpose();
        pt = detail::project<Scalar, N>(pt, lower, upper);
        model.fval[k] = eval_fn(pt);
    }

    // Step 4: Find new best point.
    model.k_opt = 0;
    model.f_opt = model.fval[0];
    for(int k = 1; k < npt; ++k)
    {
        if(model.fval[k] < model.f_opt)
        {
            model.f_opt = model.fval[k];
            model.k_opt = static_cast<std::uint16_t>(k);
        }
    }
    model.x_opt = model.xpt.row(model.k_opt).transpose();

    // Step 5: Rebuild BMAT/ZMAT from scratch (not incremental).
    //
    // Uses the same closed-form expressions as initialize_h_matrix
    // in bobyqa_model::initialize, since the point layout is identical
    // (coordinate-based 2N+1 set).
    model.bmat.setZero();
    model.zmat.setZero();

    for(int i = 0; i < n; ++i)
    {
        Scalar sp = model.xpt(1 + i, i);
        Scalar sn = model.xpt(1 + n + i, i);
        Scalar denom = sp * sn;

        if(std::abs(denom) < std::numeric_limits<Scalar>::epsilon())
            continue;

        model.bmat(0, i) += -(sp + sn) / denom;
        model.bmat(1 + i, i) = -sn / denom;
        model.bmat(1 + n + i, i) = -sp / denom;
        model.bmat(npt + i, i) = -Scalar(1) / sp - Scalar(1) / sn;

        Scalar z_base = std::sqrt(Scalar(2)) / std::abs(denom);
        Scalar recip_sp = Scalar(1) / sp;
        Scalar recip_sn = Scalar(1) / sn;

        model.zmat(0, i) = z_base;
        model.zmat(1 + i, i) = recip_sp / std::abs(sp) * std::sqrt(Scalar(0.5));
        model.zmat(1 + n + i, i) = recip_sn / std::abs(sn) * std::sqrt(Scalar(0.5));
    }

    // Step 6: Recompute GOPT, HQ, PQ from the new interpolation conditions.
    //
    // Reference: Powell 2009, Sec. 2 (PRELIM coefficient initialization).
    model.hq.setZero();
    model.gopt.setZero();
    model.pq.setZero();

    Scalar f0 = model.fval[0];
    for(int i = 0; i < n; ++i)
    {
        Scalar sp = model.xpt(1 + i, i);
        Scalar sn = model.xpt(1 + n + i, i);
        Scalar fp = model.fval[1 + i];
        Scalar fn = model.fval[1 + n + i];
        Scalar denom = sp * sn;

        if(std::abs(denom) < std::numeric_limits<Scalar>::epsilon())
            continue;

        Scalar sp2 = sp * sp;
        Scalar sn2 = sn * sn;
        Scalar diff = sp - sn;

        if(std::abs(diff) < std::numeric_limits<Scalar>::epsilon())
        {
            model.gopt[i] = (fp - fn) / (Scalar(2) * sp);
        }
        else
        {
            model.gopt[i] = -(fp * sn2 - fn * sp2 - f0 * (sn2 - sp2))
                              / (denom * diff);
        }

        model.pq[1 + i] = (fn - f0 - sn * model.gopt[i]) / (sn * sn);
        model.pq[1 + n + i] = (fp - f0 - sp * model.gopt[i]) / (sp * sp);
    }

    // Adjust GOPT to be at x_opt if k_opt != 0.
    if(model.k_opt != 0)
    {
        Eigen::Vector<Scalar, N> hv = model.hessian_vector_product(model.x_opt);
        model.gopt += hv;
    }
}

}

#endif
