#ifndef HPP_GUARD_NABLAPP_DETAIL_SUBSPACE_MINIMIZATION_H
#define HPP_GUARD_NABLAPP_DETAIL_SUBSPACE_MINIMIZATION_H

#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <vector>

namespace nablapp::detail
{

// Subspace minimization over free variables after the generalized Cauchy point.
//
// Given the Cauchy point x_cauchy and the set of free indices (variables not at
// bounds), solve the reduced system B_FF * d_hat = -r_F where B_FF is the reduced
// Hessian and r_F is the reduced gradient at the Cauchy point. The result is
// projected to bounds for numerical safety.
//
// Reference: N&W Section 16.6, p. 478, eq. 16.49.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

template <typename Scalar = double>
Eigen::VectorX<Scalar> subspace_minimize(
    const Eigen::VectorX<Scalar>& x,
    const Eigen::VectorX<Scalar>& x_cauchy,
    const Eigen::VectorX<Scalar>& g,
    const Eigen::VectorX<Scalar>& lower,
    const Eigen::VectorX<Scalar>& upper,
    const std::vector<int>& free_indices,
    const compact_lbfgs<Scalar>& B)
{
    // If no free variables, return Cauchy point (all at bounds, pitfall 2)
    if(free_indices.empty())
        return x_cauchy;

    const int nf = static_cast<int>(free_indices.size());

    // Compute reduced gradient at Cauchy point: r_F[j] = g[F[j]] + (B*(x_c - x))[F[j]]
    Eigen::VectorX<Scalar> Bd = B.multiply(x_cauchy - x);
    Eigen::VectorX<Scalar> r_F(nf);
    for(int j = 0; j < nf; ++j)
        r_F[j] = g[free_indices[j]] + Bd[free_indices[j]];

    // Build reduced Hessian B_FF
    Eigen::MatrixX<Scalar> B_FF = B.reduced_hessian(free_indices);

    // Solve B_FF * d_hat = -r_F using LDLT
    Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(B_FF);
    Eigen::VectorX<Scalar> d_hat;

    if(ldlt.info() == Eigen::Success && ldlt.isPositive())
    {
        d_hat = ldlt.solve(-r_F);
    }
    else
    {
        // Fallback: steepest descent on reduced space, scaled by 1/theta
        d_hat = -r_F / B.theta();
    }

    // Assemble full step: x_new = x_cauchy + d (d_i = 0 for fixed variables)
    Eigen::VectorX<Scalar> x_new = x_cauchy;
    for(int j = 0; j < nf; ++j)
        x_new[free_indices[j]] += d_hat[j];

    // Project to bounds for numerical safety
    x_new = project(x_new, lower, upper);

    return x_new;
}

}

#endif
