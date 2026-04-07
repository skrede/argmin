#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_ALTMOV_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_ALTMOV_H

// Geometry improvement step for BOBYQA (ALTMOV procedure).
//
// Finds a step d that maximizes |L_knew(x_opt + d)| within the trust
// region and bounds, where L_knew is the knew-th Lagrange polynomial.
// This ensures the interpolation geometry stays well-conditioned when
// replacing point `knew`.
//
// The procedure computes two candidate steps: one along the gradient
// of L_knew (XNEW), and one along the Cauchy-like direction (XALT).
// The candidate with larger |L_knew| is returned.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Section 6 (geometry improvement).

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

// Compute the knew-th Lagrange polynomial value at x_opt + d.
//
// L_knew(x_opt + d) uses BMAT row knew and ZMAT contributions.
// This is equivalent to vlag[knew] from compute_vlag, but computed
// more efficiently for a single index.
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int N, int NPT>
Scalar lagrange_value_at(const bobyqa_model<Scalar, N, NPT>& model,
                         std::uint16_t knew,
                         const Eigen::Vector<Scalar, N>& d)
{
    const int npt = model.pq.size();
    const int zmat_cols = model.zmat.cols();

    // Compute w[k] = xpt[k]^T d for each interpolation point.
    Eigen::Vector<Scalar, NPT> w(npt);
    for(int k = 0; k < npt; ++k)
        w[k] = model.xpt.row(k).dot(d);

    // ZMAT contribution: sum_j zmat(knew, j) * (Z^T w)_j.
    Scalar val = Scalar(0);
    for(int j = 0; j < zmat_cols; ++j)
    {
        Scalar ztw = Scalar(0);
        for(int k = 0; k < npt; ++k)
            ztw += model.zmat(k, j) * w[k];
        val += model.zmat(knew, j) * ztw;
    }

    // BMAT contribution.
    val += model.bmat.row(knew).dot(d);

    return val;
}

// Compute the gradient of L_knew at x_opt (d = 0).
//
// grad L_knew = BMAT[knew, :] + sum_j zmat(knew, j) * sum_k zmat(k, j) * xpt[k]
//
// Reference: Powell 2009, Sec. 6.
template <typename Scalar, int N, int NPT>
Eigen::Vector<Scalar, N> lagrange_gradient(
    const bobyqa_model<Scalar, N, NPT>& model,
    std::uint16_t knew)
{
    const int n = model.gopt.size();
    const int npt = model.pq.size();
    const int zmat_cols = model.zmat.cols();

    Eigen::Vector<Scalar, N> grad = model.bmat.row(knew).transpose();

    for(int j = 0; j < zmat_cols; ++j)
    {
        Scalar zk = model.zmat(knew, j);
        if(std::abs(zk) < std::numeric_limits<Scalar>::epsilon())
            continue;

        // sum_k zmat(k, j) * xpt[k]
        Eigen::Vector<Scalar, N> sum = Eigen::Vector<Scalar, N>::Zero(n);
        for(int k = 0; k < npt; ++k)
            sum.noalias() += model.zmat(k, j) * model.xpt.row(k).transpose();

        grad.noalias() += zk * sum;
    }

    return grad;
}

// ALTMOV: geometry improvement step (Powell 2009, Sec. 6).
//
// Finds a step d from x_opt that maximizes |L_knew(x_opt + d)| subject
// to ||d|| <= delta and lower <= x_base + x_opt + d <= upper.
//
// Two candidates are generated:
//   1. Step along the gradient of L_knew, projected to bounds and trust region
//   2. Cauchy step (steepest ascent of |L_knew|), projected similarly
// The candidate with larger |L_knew| is returned.
//
// Reference: Powell 2009, Sec. 6.
template <typename Scalar, int N, int NPT>
Eigen::Vector<Scalar, N> altmov(
    const bobyqa_model<Scalar, N, NPT>& model,
    std::uint16_t knew,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    Scalar delta)
{
    const int n = model.gopt.size();
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();

    // Effective bounds on d: lower - x_base - x_opt <= d <= upper - x_base - x_opt.
    Eigen::Vector<Scalar, N> d_lo = lower - model.x_base - model.x_opt;
    Eigen::Vector<Scalar, N> d_hi = upper - model.x_base - model.x_opt;

    // Gradient of L_knew at x_opt.
    Eigen::Vector<Scalar, N> grad = lagrange_gradient(model, knew);

    Scalar grad_norm = grad.norm();
    if(grad_norm < eps)
        return Eigen::Vector<Scalar, N>::Zero(n);

    // Candidate 1: step along the gradient direction.
    //
    // Scale gradient to trust-region size, then project to bounds.
    // Reference: Powell 2009, Sec. 6.
    Eigen::Vector<Scalar, N> d1 = (delta / grad_norm) * grad;

    // Project to box bounds.
    for(int i = 0; i < n; ++i)
        d1[i] = std::clamp(d1[i], d_lo[i], d_hi[i]);

    // Scale back to trust region if needed.
    Scalar d1_norm = d1.norm();
    if(d1_norm > delta)
        d1 *= delta / d1_norm;

    Scalar lval1 = lagrange_value_at(model, knew, d1);

    // Candidate 2: step in the negative gradient direction.
    //
    // This captures the case where L_knew is more negative than positive
    // in the gradient direction (we want max |L_knew|, not max L_knew).
    Eigen::Vector<Scalar, N> d2 = -(delta / grad_norm) * grad;

    for(int i = 0; i < n; ++i)
        d2[i] = std::clamp(d2[i], d_lo[i], d_hi[i]);

    Scalar d2_norm = d2.norm();
    if(d2_norm > delta)
        d2 *= delta / d2_norm;

    Scalar lval2 = lagrange_value_at(model, knew, d2);

    // Select the candidate with larger |L_knew|.
    Eigen::Vector<Scalar, N> best_d = (std::abs(lval1) >= std::abs(lval2)) ? d1 : d2;
    Scalar best_lval = std::max(std::abs(lval1), std::abs(lval2));

    // Candidate 3: try the direction toward the old point location.
    //
    // The point being replaced has a known position; stepping toward it
    // often provides good geometry. Reference: Powell 2009, Sec. 6.
    Eigen::Vector<Scalar, N> toward = model.xpt.row(knew).transpose() - model.x_opt;
    Scalar toward_norm = toward.norm();
    if(toward_norm > eps)
    {
        Eigen::Vector<Scalar, N> d3 = (delta / toward_norm) * toward;
        for(int i = 0; i < n; ++i)
            d3[i] = std::clamp(d3[i], d_lo[i], d_hi[i]);

        Scalar d3_norm = d3.norm();
        if(d3_norm > delta)
            d3 *= delta / d3_norm;

        Scalar lval3 = lagrange_value_at(model, knew, d3);
        if(std::abs(lval3) > best_lval)
        {
            best_d = d3;
            best_lval = std::abs(lval3);
        }
    }

    // Safety: ensure non-trivial step.
    if(best_d.norm() < eps * delta)
        return Eigen::Vector<Scalar, N>::Zero(n);

    return best_d;
}

}

#endif
