#ifndef HPP_GUARD_NABLAPP_DETAIL_CMAES_CONSTANTS_H
#define HPP_GUARD_NABLAPP_DETAIL_CMAES_CONSTANTS_H

// CMA-ES strategy parameter computation.
//
// Precomputes all population parameters, learning rates, and recombination
// weights from the problem dimension n and optional population size override.
//
// Reference: K&W Section 8.7, Algorithm 8.10, Eq. 8.19, 8.22, 8.25-8.27,
//            8.33-8.35.
//            Hansen (2023) "The CMA Evolution Strategy: A Tutorial",
//            arXiv:1604.00772, Section 3.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

struct cmaes_params
{
    int lambda;
    int mu;
    Eigen::VectorXd weights;
    double mu_eff;
    double c_sigma;
    double d_sigma;
    double c_c;
    double c_1;
    double c_mu;
    double chi_n;
};

// Compute CMA-ES strategy parameters from dimension n.
// If lambda_override > 0, use that population size; otherwise auto-compute.
// Reference: K&W Eq. 8.19, 8.22, 8.25-8.27, 8.33-8.35; Hansen tutorial Sec. 3.
inline cmaes_params compute_constants(int n, int lambda_override = 0)
{
    cmaes_params p{};

    // Population size: K&W Eq. 8.19
    p.lambda = (lambda_override > 0)
        ? lambda_override
        : 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(n))));

    // Parent count: K&W Eq. 8.22
    p.mu = p.lambda / 2;

    // Raw weights: w'_i = ln((lambda+1)/2) - ln(i), i = 1..lambda
    // K&W Eq. 8.25
    p.weights.resize(p.lambda);
    double log_half_lambda_plus_one = std::log(0.5 * static_cast<double>(p.lambda) + 0.5);
    for(int i = 0; i < p.lambda; ++i)
        p.weights[i] = log_half_lambda_plus_one - std::log(static_cast<double>(i + 1));

    // Normalize positive weights (i < mu) to sum to 1
    double sum_pos = 0.0;
    for(int i = 0; i < p.mu; ++i)
        sum_pos += p.weights[i];

    // mu_eff from positive weights before normalization: K&W Eq. 8.27
    double sum_pos_sq = 0.0;
    for(int i = 0; i < p.mu; ++i)
    {
        double w_norm = p.weights[i] / sum_pos;
        sum_pos_sq += w_norm * w_norm;
    }
    p.mu_eff = 1.0 / sum_pos_sq;

    double nd = static_cast<double>(n);

    // Learning rates: K&W Eq. 8.33-8.35
    p.c_sigma = (p.mu_eff + 2.0) / (nd + p.mu_eff + 5.0);
    p.d_sigma = 1.0 + 2.0 * std::max(0.0, std::sqrt((p.mu_eff - 1.0) / (nd + 1.0)) - 1.0)
              + p.c_sigma;

    p.c_c = (4.0 + p.mu_eff / nd) / (nd + 4.0 + 2.0 * p.mu_eff / nd);
    p.c_1 = 2.0 / ((nd + 1.3) * (nd + 1.3) + p.mu_eff);
    p.c_mu = std::min(1.0 - p.c_1,
                      2.0 * (p.mu_eff - 2.0 + 1.0 / p.mu_eff)
                          / ((nd + 2.0) * (nd + 2.0) + p.mu_eff));

    // E||N(0,I)|| approximation: Hansen tutorial
    p.chi_n = std::sqrt(nd) * (1.0 - 1.0 / (4.0 * nd) + 1.0 / (21.0 * nd * nd));

    // Normalize positive weights to sum to 1
    for(int i = 0; i < p.mu; ++i)
        p.weights[i] /= sum_pos;

    // Normalize negative weights: sum of |w_i| for i >= mu equals
    // (1 + c_1/c_mu) if feasible, else normalize to sum to -1.
    // Hansen tutorial, Section 3.
    double sum_neg = 0.0;
    for(int i = p.mu; i < p.lambda; ++i)
        sum_neg += std::abs(p.weights[i]);

    if(sum_neg > 0.0)
    {
        double target = 1.0 + p.c_1 / std::max(p.c_mu, 1e-20);
        // Clamp target to reasonable range
        double scale = std::min(target, 1.0 + 2.0 * p.mu_eff / (p.mu_eff + 2.0));
        for(int i = p.mu; i < p.lambda; ++i)
            p.weights[i] = p.weights[i] / sum_neg * (-scale);
    }

    return p;
}

}

#endif
