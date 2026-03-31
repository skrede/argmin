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
inline void eigendecompose(const Eigen::MatrixXd& C,
                           Eigen::MatrixXd& B,
                           Eigen::VectorXd& D)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    B = es.eigenvectors();
    D = es.eigenvalues().cwiseMax(1e-20).cwiseSqrt();
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
inline void update_covariance(Eigen::MatrixXd& C,
                              const Eigen::VectorXd& p_c,
                              double h_sigma,
                              double c_1,
                              double c_mu,
                              double c_c,
                              const Eigen::VectorXd& weights,
                              const Eigen::MatrixXd& deltas,
                              const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& D,
                              int mu)
{
    const int n = C.rows();
    const int lambda = weights.size();

    // Sum of absolute weights for the diagonal decay factor
    double sum_abs_w = 0.0;
    for(int i = 0; i < lambda; ++i)
        sum_abs_w += std::abs(weights[i]);

    // Diagonal decay factor
    double alpha = 1.0 + c_1 * (1.0 - h_sigma) * c_c * (2.0 - c_c)
                 - c_1 - c_mu * sum_abs_w;
    alpha = std::max(alpha, 0.0);

    // Rank-mu contribution
    Eigen::MatrixXd rank_mu = Eigen::MatrixXd::Zero(n, n);
    for(int i = 0; i < lambda; ++i)
    {
        double w_i = weights[i];
        Eigen::VectorXd di = deltas.col(i);

        if(i >= mu && w_i < 0.0)
        {
            // Negative weight scaling: n / ||C^{-1/2} * delta_i||^2
            // C^{-1/2} * v = B * D^{-1} * B^T * v
            // K&W Eq. 8.31
            Eigen::VectorXd inv_sqrt_delta =
                (B * D.cwiseInverse().asDiagonal() * (B.transpose() * di)).eval();
            double denom = inv_sqrt_delta.squaredNorm();
            if(denom > 1e-30)
                w_i *= static_cast<double>(n) / denom;
        }

        rank_mu.noalias() += w_i * di * di.transpose();
    }

    // Full update
    C = alpha * C + c_1 * p_c * p_c.transpose() + c_mu * rank_mu;

    // Enforce symmetry
    C = (C + C.transpose()) * 0.5;
}

}

#endif
