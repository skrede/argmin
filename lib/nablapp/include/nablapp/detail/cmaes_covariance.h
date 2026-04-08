#ifndef HPP_GUARD_NABLAPP_DETAIL_CMAES_COVARIANCE_H
#define HPP_GUARD_NABLAPP_DETAIL_CMAES_COVARIANCE_H

// CMA-ES covariance matrix operations.
//
// Eigendecomposition of the covariance matrix and rank-one + rank-mu
// covariance update with negative weight handling.
//
// Reference: K&W Section 8.7, Eq. 8.28-8.31.
//            Hansen (2023) "The CMA Evolution Strategy: A Tutorial",
//            arXiv:1604.00772, Section 4.

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// Eigendecompose C into B * diag(D)^2 * B^T.
// B receives eigenvectors (columns), D receives sqrt(eigenvalues).
// Eigenvalues clamped to min 1e-20 for numerical safety.
// Reference: Hansen tutorial Section 4.
//
// Overload accepting a pre-allocated eigensolver to avoid per-call
// workspace allocation (the solver's internal buffers are reused).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void eigendecompose(const Eigen::Matrix<Scalar, N, N>& C,
                    Eigen::Matrix<Scalar, N, N>& B,
                    Eigen::Vector<Scalar, N>& D,
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, N, N>>& es)
{
    es.compute(C);
    B = es.eigenvectors();
    D = es.eigenvalues().cwiseMax(Scalar(1e-20)).cwiseSqrt();
}

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void eigendecompose(const Eigen::Matrix<Scalar, N, N>& C,
                    Eigen::Matrix<Scalar, N, N>& B,
                    Eigen::Vector<Scalar, N>& D)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, N, N>> es(C);
    B = es.eigenvectors();
    D = es.eigenvalues().cwiseMax(Scalar(1e-20)).cwiseSqrt();
}

// Rank-one + rank-mu covariance update.
//
// C_new = (1 + c_1*(1-h_sigma)*c_c*(2-c_c) - c_1 - c_mu*sum(|w_i|)) * C
//       + c_1 * p_c * p_c^T
//       + c_mu * sum(w'_i * delta_i * delta_i^T)
//
// For negative weights (i >= mu), scale by n / ||C^{-1/2} * delta_i||^2.
// Enforces symmetry via (C + C^T) / 2 after update.
//
// deltas: n x lambda matrix where column i = (x_{i:lambda} - mean_old) / sigma.
// Reference: K&W Eq. 8.28-8.31.
template <typename Scalar = double, int N = nablapp::dynamic_dimension,
          int Lambda = nablapp::dynamic_dimension, typename DeltaMatrix = Eigen::Matrix<Scalar, N, Lambda>>
void update_covariance(Eigen::Matrix<Scalar, N, N>& C,
                       const Eigen::Vector<Scalar, N>& p_c,
                       Scalar h_sigma,
                       Scalar c_1,
                       Scalar c_mu,
                       Scalar c_c,
                       const Eigen::Vector<Scalar, Lambda>& weights,
                       const DeltaMatrix& deltas,
                       const Eigen::Matrix<Scalar, N, N>& B,
                       const Eigen::Vector<Scalar, N>& D,
                       int mu,
                       bool& covariance_dirty)
{
    const int n = C.rows();
    const int lambda = weights.size();

    // Sum of absolute weights for the diagonal decay factor
    Scalar sum_abs_w = Scalar(0);
    for(int i = 0; i < lambda; ++i)
        sum_abs_w += std::abs(weights[i]);

    // Diagonal decay factor
    Scalar alpha = Scalar(1) + c_1 * (Scalar(1) - h_sigma) * c_c * (Scalar(2) - c_c)
                 - c_1 - c_mu * sum_abs_w;
    alpha = std::max(alpha, Scalar(0));

    // Rank-mu contribution
    Eigen::Matrix<Scalar, N, N> rank_mu = Eigen::Matrix<Scalar, N, N>::Zero(n, n);
    for(int i = 0; i < lambda; ++i)
    {
        Scalar w_i = weights[i];
        const auto di = deltas.col(i);

        if(i >= mu && w_i < Scalar(0))
        {
            // Negative weight scaling: n / ||C^{-1/2} * delta_i||^2
            // C^{-1/2} * v = B * D^{-1} * B^T * v
            // K&W Eq. 8.31
            Eigen::Vector<Scalar, N> inv_sqrt_delta =
                (B * D.cwiseInverse().asDiagonal() * (B.transpose() * di)).eval();
            Scalar denom = inv_sqrt_delta.squaredNorm();
            if(denom > Scalar(1e-30))
                w_i *= static_cast<Scalar>(n) / denom;
        }

        rank_mu.noalias() += w_i * di * di.transpose();
    }

    // Full update
    C = alpha * C + c_1 * p_c * p_c.transpose() + c_mu * rank_mu;

    // Enforce symmetry
    C = (C + C.transpose()) * Scalar(0.5);

    covariance_dirty = true;
}

}

#endif
