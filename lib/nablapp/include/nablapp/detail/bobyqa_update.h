#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_UPDATE_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_UPDATE_H

// H-matrix update and point replacement for BOBYQA.
//
// Provides VLAG computation, beta/denominator evaluation, and Powell's
// point replacement selection. Model updates after replacement use a
// full SVD rebuild for numerical robustness.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Sections 2 and 4.

#include "nablapp/detail/bobyqa_model.h"
#include "nablapp/types.h"

#include <Eigen/Core>

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
    // Reference: Powell 2009, Sec. 4.
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
    for(int k = 0; k < npt_plus_n; ++k)
        vlag[k] += model.bmat.row(k).dot(d);

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
// Returns beta * h_diag_k + vlag_k * vlag_k.
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

// Select which point to replace using Powell's denominator criterion.
//
// Choose knew = argmax_k |denominator_k| among eligible points (not k_opt).
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

// Update model after replacing point knew.
//
// Replaces point knew in XPT/FVAL with the new point (x_opt + d_new),
// updates the best point tracking, and rebuilds the model coefficients
// (GOPT, HQ, PQ, BMAT, ZMAT) from scratch via SVD for robustness.
//
// Reference: Powell 2009, Sec. 4.
template <typename Scalar, int N, int NPT>
void update_model_after_replacement(
    bobyqa_model<Scalar, N, NPT>& model,
    std::uint16_t knew,
    const Eigen::Vector<Scalar, N>& d_new,
    Scalar f_new,
    [[maybe_unused]] const Eigen::Vector<Scalar,
        (NPT == nablapp::dynamic_dimension || N == nablapp::dynamic_dimension)
            ? nablapp::dynamic_dimension
            : NPT + N>& vlag,
    [[maybe_unused]] Scalar beta)
{
    // Update XPT and FVAL for the replaced point.
    model.xpt.row(knew) = (model.x_opt + d_new).transpose();
    model.fval[knew] = f_new;

    // Update best point if improved.
    if(f_new < model.f_opt)
    {
        model.f_opt = f_new;
        model.k_opt = knew;
        model.x_opt = model.xpt.row(knew).transpose();
    }

    // Rebuild model from scratch for numerical robustness.
    // The incremental BMAT/ZMAT rank-2 update is numerically fragile
    // for small N; a full rebuild is O(n^3) which is negligible for n < 20.
    model.rebuild_model();
}

}

#endif
