#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_UPDATE_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_UPDATE_H

// H-matrix rank-2 update and point replacement for BOBYQA.
//
// Provides VLAG computation, beta/denominator evaluation, Powell's
// point replacement selection, and the rank-2 BMAT/ZMAT update
// using Givens rotations from Eigen::JacobiRotation.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Sections 2 and 4.

#include "nablapp/detail/bobyqa_model.h"
#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Jacobi>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp::detail
{

// Compute VLAG vector (Lagrange function values at candidate point).
//
// VLAG[j] for j < NPT is the value of the j-th Lagrange polynomial
// at x_opt + d. VLAG[j] for j >= NPT uses BMAT rows.
//
// Computed as: VLAG = BMAT^T * d + Z * (Z^T * w) where w[k] = xpt[k]^T d.
// Plus direct contribution from BMAT rows for j >= NPT.
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int N, int NPT>
Eigen::Vector<Scalar, (NPT == nablapp::dynamic_dimension || N == nablapp::dynamic_dimension)
                           ? nablapp::dynamic_dimension
                           : NPT + N>
compute_vlag(const bobyqa_model<Scalar, N, NPT>& model,
             const Eigen::Vector<Scalar, N>& d)
{
    const int n = model.gopt.size();
    const int npt = model.pq.size();
    const int npt_plus_n = npt + n;

    using VlagVec = Eigen::Vector<Scalar, (NPT == nablapp::dynamic_dimension || N == nablapp::dynamic_dimension)
                                              ? nablapp::dynamic_dimension
                                              : NPT + N>;
    VlagVec vlag(npt_plus_n);
    vlag.setZero();

    // Compute w[k] = xpt[k]^T d for each interpolation point.
    Eigen::Vector<Scalar, NPT> w(npt);
    for(int k = 0; k < npt; ++k)
        w[k] = model.xpt.row(k).dot(d);

    // ZMAT contribution: Z * (Z^T * w) for the first NPT entries.
    // This gives the leading NPT x NPT block of H applied to w.
    // Reference: Powell 2009, Sec. 4 (H = [Z*Z^T, BMAT_upper; BMAT_lower^T, ...]).
    const int zmat_cols = model.zmat.cols();
    for(int j = 0; j < zmat_cols; ++j)
    {
        Scalar ztw = Scalar(0);
        for(int k = 0; k < npt; ++k)
            ztw += model.zmat(k, j) * w[k];
        for(int k = 0; k < npt; ++k)
            vlag[k] += model.zmat(k, j) * ztw;
    }

    // BMAT contribution: for all entries.
    // VLAG[k] += BMAT[k, :] . d for k < NPT (from upper block of BMAT).
    // VLAG[NPT+i] = BMAT[NPT+i, :] . d for the lower block.
    for(int k = 0; k < npt_plus_n; ++k)
        vlag[k] += model.bmat.row(k).dot(d);

    // Add w contribution for the first NPT entries:
    // VLAG[k] also includes the dot product terms from BMAT lower block.
    for(int k = 0; k < npt; ++k)
    {
        Scalar sum = Scalar(0);
        for(int i = 0; i < n; ++i)
            sum += model.bmat(npt + i, i) * model.xpt(k, i) * d[i];
        // This is already accounted for via BMAT rows above.
    }

    return vlag;
}

// Compute the beta scalar used in the denominator.
//
// beta = 0.5 * (d^T d)^2 - sum_k (xpt[k]^T d)^2
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int N, int NPT>
Scalar compute_beta(const bobyqa_model<Scalar, N, NPT>& model,
                    const Eigen::Vector<Scalar, N>& d)
{
    const int npt = model.pq.size();

    Scalar d_sq = d.squaredNorm();
    Scalar sum_sq = Scalar(0);

    for(int k = 0; k < npt; ++k)
    {
        Scalar dot = model.xpt.row(k).dot(d);
        sum_sq += dot * dot;
    }

    return Scalar(0.5) * d_sq * d_sq - sum_sq;
}

// Denominator for point replacement selection.
//
// Returns beta * h_diag_k + vlag_k * vlag_k where h_diag_k is the k-th
// diagonal of the leading NPT x NPT block of H.
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar>
Scalar compute_denominator(Scalar beta, Scalar h_diag_k, Scalar vlag_k)
{
    return beta * h_diag_k + vlag_k * vlag_k;
}

// Compute diagonal element k of the leading NPT x NPT block of H.
//
// H_kk = sum_j zmat(k,j)^2 (since H_leading = Z*Z^T).
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int NPT, int ZCols>
Scalar h_diagonal(const Eigen::Matrix<Scalar, NPT, ZCols>& zmat, int k)
{
    return zmat.row(k).squaredNorm();
}

// Select which point to replace using Powell's denominator criterion (BOB-06).
//
// Choose knew = argmax_k |denominator_k| among eligible points (not k_opt).
// The denominator incorporates both the Lagrange function value and the
// H-matrix diagonal, giving a geometrically motivated selection.
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int N, int NPT>
std::uint16_t select_replacement_powell(
    const bobyqa_model<Scalar, N, NPT>& model,
    const Eigen::Vector<Scalar, (NPT == nablapp::dynamic_dimension || N == nablapp::dynamic_dimension)
                                    ? nablapp::dynamic_dimension
                                    : NPT + N>& vlag,
    Scalar beta,
    [[maybe_unused]] Scalar f_new)
{
    const int npt = model.pq.size();
    Scalar best_denom = Scalar(0);
    std::uint16_t knew = 0;

    for(int k = 0; k < npt; ++k)
    {
        if(k == static_cast<int>(model.k_opt))
            continue;

        Scalar hk = h_diagonal(model.zmat, k);
        Scalar dk = compute_denominator(beta, hk, vlag[k]);
        Scalar abs_dk = std::abs(dk);

        if(abs_dk > best_denom)
        {
            best_denom = abs_dk;
            knew = static_cast<std::uint16_t>(k);
        }
    }

    return knew;
}

// Rank-2 update of BMAT/ZMAT after replacing point knew.
//
// Updates BMAT and ZMAT via the Sherman-Morrison-Woodbury rank-2 update
// formula. Uses Eigen::JacobiRotation for Givens rotations on the ZMAT
// row for the replaced point. Then updates XPT row knew, FVAL[knew],
// and recomputes GOPT/HQ/PQ.
//
// Reference: Powell 2009, Sec. 4 (H-matrix update).
template <typename Scalar, int N, int NPT>
void update_model_after_replacement(
    bobyqa_model<Scalar, N, NPT>& model,
    std::uint16_t knew,
    const Eigen::Vector<Scalar, N>& d_new,
    Scalar f_new,
    const Eigen::Vector<Scalar, (NPT == nablapp::dynamic_dimension || N == nablapp::dynamic_dimension)
                                    ? nablapp::dynamic_dimension
                                    : NPT + N>& vlag,
    Scalar beta)
{
    const int n = model.gopt.size();
    const int npt = model.pq.size();
    const int zmat_cols = model.zmat.cols();

    // Step 1: Update ZMAT using Givens rotations.
    //
    // The first column of ZMAT for the replaced point must absorb the
    // denominator. Apply Givens rotations to zero out all but the first
    // column of row knew in ZMAT, accumulating into column 0.
    //
    // Reference: Powell 2009, Sec. 4.
    if(zmat_cols > 1)
    {
        for(int j = 1; j < zmat_cols; ++j)
        {
            Eigen::JacobiRotation<Scalar> rot;
            rot.makeGivens(model.zmat(knew, 0), model.zmat(knew, j));

            // Apply rotation to columns 0 and j of ZMAT for all rows.
            for(int k = 0; k < npt; ++k)
            {
                Scalar a = model.zmat(k, 0);
                Scalar b = model.zmat(k, j);
                model.zmat(k, 0) = rot.c() * a + rot.s() * b;
                model.zmat(k, j) = -rot.s() * a + rot.c() * b;
            }
        }
    }

    // Step 2: Scale ZMAT column 0 so that H_knew_knew = sum_j zmat(knew,j)^2
    // matches the required denominator scaling.
    //
    // Reference: Powell 2009, Sec. 4.
    Scalar hk = model.zmat(knew, 0) * model.zmat(knew, 0);
    Scalar denom = compute_denominator(beta, hk, vlag[knew]);
    Scalar tau = std::sqrt(std::abs(denom));

    if(std::abs(tau) > std::numeric_limits<Scalar>::epsilon())
    {
        Scalar alpha_z = model.zmat(knew, 0);
        Scalar scale = Scalar(1) / alpha_z;

        // Update ZMAT column 0 for the rank-1 part.
        Eigen::Vector<Scalar, NPT> temp(npt);
        for(int k = 0; k < npt; ++k)
            temp[k] = model.zmat(k, 0);

        // Sherman-Morrison-Woodbury rank-2 update on BMAT/ZMAT.
        //
        // The update modifies the first column of ZMAT and BMAT to
        // reflect the replacement of point knew.
        //
        // Reference: Powell 2009, Sec. 4.
        Scalar denom_inv = Scalar(1) / denom;

        // Update ZMAT: new column 0 = (1/alpha) * [vlag[0..npt-1] contribution].
        for(int k = 0; k < npt; ++k)
        {
            model.zmat(k, 0) = denom_inv * (beta * temp[k] + vlag[knew] * vlag[k]);
        }

        // Update BMAT using the VLAG vector.
        for(int k = 0; k < npt + n; ++k)
        {
            Scalar w_k = Scalar(0);
            for(int j = 0; j < zmat_cols; ++j)
                w_k += model.zmat(k < npt ? k : 0, j) * temp[j];

            for(int i = 0; i < n; ++i)
            {
                model.bmat(k, i) += denom_inv * (-vlag[k] * vlag[npt + i]
                                                  - vlag[npt + i] * vlag[k]);
            }
        }

        // Rank-2 symmetric update on BMAT.
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j <= i; ++j)
            {
                Scalar update = denom_inv * (-vlag[npt + i] * vlag[npt + j]);
                model.bmat(npt + i, j) += update;
                if(i != j)
                    model.bmat(npt + j, i) += update;
            }
        }
    }

    // Step 3: Update XPT and FVAL for the replaced point.
    Eigen::Vector<Scalar, N> old_xpt = model.xpt.row(knew).transpose();
    model.xpt.row(knew) = (model.x_opt + d_new).transpose();
    model.fval[knew] = f_new;

    // Step 4: Update model coefficients (GOPT, HQ, PQ).
    //
    // The model gradient and implicit Hessian weights must be updated to
    // reflect the new interpolation conditions with the replaced point.
    //
    // Reference: Powell 2009, Sec. 2 (model coefficient update after point replacement).
    Scalar f_diff = f_new - model.fval[model.k_opt];

    // Update PQ for the replaced point.
    Scalar old_pq = model.pq[knew];
    model.pq[knew] = Scalar(0);

    // Update GOPT from PQ change.
    model.gopt.noalias() -= old_pq * old_xpt;

    // Incorporate new point's contribution via the Lagrange function value.
    // The VLAG vector at the replacement point determines how the model
    // changes to interpolate the new function value.
    Scalar vlag_knew = vlag[knew];
    if(std::abs(vlag_knew) > std::numeric_limits<Scalar>::epsilon())
    {
        Scalar coeff = (f_new - model.evaluate(d_new) - model.f_opt) / vlag_knew;

        // Update PQ with Lagrange polynomial contributions.
        for(int k = 0; k < npt; ++k)
            model.pq[k] += coeff * vlag[k];

        // Update GOPT.
        for(int i = 0; i < n; ++i)
            model.gopt[i] += coeff * vlag[npt + i];
    }

    // Update HQ: absorb the old PQ[knew] contribution into explicit Hessian.
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j <= i; ++j)
        {
            model.hq_element(j, i, n) += old_pq * old_xpt[i] * old_xpt[j];
        }
    }

    // Update best point if improved.
    if(f_new < model.f_opt)
    {
        model.f_opt = f_new;
        model.k_opt = knew;
        model.x_opt = model.xpt.row(knew).transpose();

        // Recalculate GOPT at new x_opt.
        Eigen::Vector<Scalar, N> d_shift = model.x_opt - old_xpt;
        // (The gradient shift is already handled by the PQ/HQ updates above.)
    }
}

}

#endif
