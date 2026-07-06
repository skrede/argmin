#ifndef HPP_GUARD_ARGMIN_DETAIL_CMAES_COVARIANCE_H
#define HPP_GUARD_ARGMIN_DETAIL_CMAES_COVARIANCE_H

// CMA-ES covariance matrix operations.
//
// Eigendecomposition of the covariance matrix and rank-one + rank-mu
// covariance update with negative weight handling.
//
// Reference: K&W Section 8.7, Eq. 8.28-8.31.
//            Hansen (2023) "The CMA Evolution Strategy: A Tutorial",
//            arXiv:1604.00772, Section 4.

#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace argmin::detail
{

// Eigendecompose C into B * diag(D)^2 * B^T.
// B receives eigenvectors (columns), D receives sqrt(eigenvalues).
// Eigenvalues clamped to min 1e-20 for numerical safety.
// Reference: Hansen tutorial Section 4.
//
// Overload accepting a pre-allocated eigensolver to avoid per-call
// workspace allocation (the solver's internal buffers are reused).
template <typename Scalar = double, int N = argmin::dynamic_dimension>
void eigendecompose(const Eigen::Matrix<Scalar, N, N>& C,
                    Eigen::Matrix<Scalar, N, N>& B,
                    Eigen::Vector<Scalar, N>& D,
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, N, N>>& es)
{
    es.compute(C);
    B = es.eigenvectors();
    D = es.eigenvalues().cwiseMax(Scalar(1e-20)).cwiseSqrt();
}

template <typename Scalar = double, int N = argmin::dynamic_dimension>
void eigendecompose(const Eigen::Matrix<Scalar, N, N>& C,
                    Eigen::Matrix<Scalar, N, N>& B,
                    Eigen::Vector<Scalar, N>& D)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, N, N>> es(C);
    B = es.eigenvectors();
    D = es.eigenvalues().cwiseMax(Scalar(1e-20)).cwiseSqrt();
}

// Rank-one + rank-mu covariance update (vanilla CMA-ES; positive weights only).
//
// C_new = (1 + c_1*(1-h_sigma)*c_c*(2-c_c) - c_1 - c_mu) * C
//       + c_1 * p_c * p_c^T
//       + c_mu * sum_{i=0..mu-1}(w_i * delta_i * delta_i^T)
//
// Enforces symmetry via (C + C^T) / 2 after update.
//
// The B and D parameters are kept in the signature for ABI stability with
// callers that still pass them (cmaes_policy.h:396-399). With vanilla
// positive-only weights they are unreferenced; active-CMA negative-weight
// rescaling using B*D^{-1}*B^T is out of scope for this milestone (see
// cmaes_constants.h tail-zero block).
//
// deltas: n x lambda matrix where column i = (x_{i:lambda} - mean_old) / sigma.
// Reference: K&W Eq. 8.28-8.31; Hansen (2023) arXiv:1604.00772 §B.2 eq (47);
//            libcmaes covarianceupdate.cc:67-78.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int Lambda = argmin::dynamic_dimension, typename DeltaMatrix = Eigen::Matrix<Scalar, N, Lambda>>
void update_covariance(Eigen::Matrix<Scalar, N, N>& C,
                       const Eigen::Vector<Scalar, N>& p_c,
                       Scalar h_sigma,
                       Scalar c_1,
                       Scalar c_mu,
                       Scalar c_c,
                       const Eigen::Vector<Scalar, Lambda>& weights,
                       const DeltaMatrix& deltas,
                       [[maybe_unused]] const Eigen::Matrix<Scalar, N, N>& B,
                       [[maybe_unused]] const Eigen::Vector<Scalar, N>& D,
                       int mu,
                       bool& covariance_dirty)
{
    const int n = C.rows();

    // Diagonal decay factor (vanilla CMA-ES; positive-weight sum = 1.0).
    //
    // References:
    //   Hansen (2023) arXiv:1604.00772 §B.2 eq (47).
    //   libcmaes covarianceupdate.cc:78.
    Scalar alpha = Scalar(1) + c_1 * (Scalar(1) - h_sigma) * c_c * (Scalar(2) - c_c)
                 - c_1 - c_mu;
    alpha = std::max(alpha, Scalar(0));

    // Rank-mu contribution (vanilla CMA-ES; positive weights only).
    //
    // References:
    //   Hansen (2023) arXiv:1604.00772 §B.2 eq (47).
    //   libcmaes covarianceupdate.cc:67-75.
    Eigen::Matrix<Scalar, N, N> rank_mu = Eigen::Matrix<Scalar, N, N>::Zero(n, n);
    for(int i = 0; i < mu; ++i)
    {
        const Scalar w_i = weights[i];
        const auto di = deltas.col(i);
        rank_mu.noalias() += w_i * di * di.transpose();
    }

    // Full update
    C = alpha * C + c_1 * p_c * p_c.transpose() + c_mu * rank_mu;

    // Enforce symmetry
    C = (C + C.transpose()) * Scalar(0.5);

    // C changed this generation: mark the cached eigenbasis stale so the
    // next decomposition on the periodic cadence refreshes B and D. The
    // caller reuses the stale eigenbasis until then.
    covariance_dirty = true;
}

}

#endif
