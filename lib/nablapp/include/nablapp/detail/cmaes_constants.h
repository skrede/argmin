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

#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double, int Lambda = nablapp::dynamic_dimension>
struct cmaes_params
{
    int lambda;
    int mu;
    Eigen::Vector<Scalar, Lambda> weights;
    Scalar mu_eff;
    Scalar c_sigma;
    Scalar d_sigma;
    Scalar c_c;
    Scalar c_1;
    Scalar c_mu;
    Scalar chi_n;
};

// Compute CMA-ES strategy parameters from dimension n.
// If lambda_override > 0, use that population size; otherwise auto-compute.
// Reference: K&W Eq. 8.19, 8.22, 8.25-8.27, 8.33-8.35; Hansen tutorial Sec. 3.
template <typename Scalar = double, int Lambda = nablapp::dynamic_dimension>
cmaes_params<Scalar, Lambda> compute_constants(int n, int lambda_override = 0)
{
    cmaes_params<Scalar, Lambda> p{};

    // Population size: K&W Eq. 8.19
    p.lambda = (lambda_override > 0)
        ? lambda_override
        : 4 + static_cast<int>(std::floor(Scalar(3) * std::log(static_cast<Scalar>(n))));

    // Parent count: K&W Eq. 8.22
    p.mu = p.lambda / 2;

    // Raw weights: w'_i = ln((lambda+1)/2) - ln(i), i = 1..lambda
    // K&W Eq. 8.25
    p.weights.resize(p.lambda);
    Scalar log_half_lambda_plus_one = std::log(Scalar(0.5) * static_cast<Scalar>(p.lambda) + Scalar(0.5));
    for(int i = 0; i < p.lambda; ++i)
        p.weights[i] = log_half_lambda_plus_one - std::log(static_cast<Scalar>(i + 1));

    // Normalize positive weights (i < mu) to sum to 1
    Scalar sum_pos = Scalar(0);
    for(int i = 0; i < p.mu; ++i)
        sum_pos += p.weights[i];

    // mu_eff from positive weights before normalization: K&W Eq. 8.27
    Scalar sum_pos_sq = Scalar(0);
    for(int i = 0; i < p.mu; ++i)
    {
        Scalar w_norm = p.weights[i] / sum_pos;
        sum_pos_sq += w_norm * w_norm;
    }
    p.mu_eff = Scalar(1) / sum_pos_sq;

    Scalar nd = static_cast<Scalar>(n);

    // Learning rates: K&W Eq. 8.33-8.35
    p.c_sigma = (p.mu_eff + Scalar(2)) / (nd + p.mu_eff + Scalar(5));
    p.d_sigma = Scalar(1) + Scalar(2) * std::max(Scalar(0), std::sqrt((p.mu_eff - Scalar(1)) / (nd + Scalar(1))) - Scalar(1))
              + p.c_sigma;

    p.c_c = (Scalar(4) + p.mu_eff / nd) / (nd + Scalar(4) + Scalar(2) * p.mu_eff / nd);
    p.c_1 = Scalar(2) / ((nd + Scalar(1.3)) * (nd + Scalar(1.3)) + p.mu_eff);
    p.c_mu = std::min(Scalar(1) - p.c_1,
                      Scalar(2) * (p.mu_eff - Scalar(2) + Scalar(1) / p.mu_eff)
                          / ((nd + Scalar(2)) * (nd + Scalar(2)) + p.mu_eff));

    // E||N(0,I)|| approximation: Hansen tutorial
    p.chi_n = std::sqrt(nd) * (Scalar(1) - Scalar(1) / (Scalar(4) * nd) + Scalar(1) / (Scalar(21) * nd * nd));

    // Normalize positive weights to sum to 1
    for(int i = 0; i < p.mu; ++i)
        p.weights[i] /= sum_pos;

    // Vanilla CMA-ES: zero the tail (i >= mu). Active-CMA negative-weight
    // rescaling is out of scope for this milestone; a future variant may
    // surface it as a guarded option under solver/alternative/cmaes/.
    //
    // References:
    //   Hansen (2023) arXiv:1604.00772 §B.1 eq (49)-(50).
    //   libcmaes covarianceupdate.cc:69-75 (positive-weights only).
    //   K&W (2025) §8.7.
    for(int i = p.mu; i < p.lambda; ++i)
        p.weights[i] = Scalar(0);

    return p;
}

}

#endif
