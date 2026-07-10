#ifndef HPP_GUARD_ARGMIN_DETAIL_SUBSPACE_MINIMIZATION_H
#define HPP_GUARD_ARGMIN_DETAIL_SUBSPACE_MINIMIZATION_H

#include "argmin/detail/compact_lbfgs.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <vector>

namespace argmin::detail
{

// Stateful subspace minimizer with pre-allocated workspace.
//
// Given the Cauchy point x_cauchy and the set of free indices (variables not at
// bounds), solves the reduced system B_FF * d_hat = -r_F where B_FF is the reduced
// Hessian and r_F is the reduced gradient at the Cauchy point. The result is
// projected to bounds for numerical safety.
//
// Reference: N&W Section 16.6, p. 478, eq. 16.49.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

template <typename Scalar = double, int N = argmin::dynamic_dimension>
class subspace_minimizer
{
public:
    explicit subspace_minimizer(int n)
        : Bd_(n)
        , r_F_(n)
        , d_hat_(n)
        , x_new_(n)
        , B_FF_(n, n)
        , ldlt_(n)
    {
    }

    subspace_minimizer() = default;

    // Solve the subspace minimization problem using pre-allocated workspace.
    //
    // Reference: N&W Section 16.6, p. 478, eq. 16.49.
    Eigen::Vector<Scalar, N> solve(
        const Eigen::Vector<Scalar, N>& x,
        const Eigen::Vector<Scalar, N>& x_cauchy,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const std::vector<int>& free_indices,
        const auto& B)
    {
        if(free_indices.empty())
            return x_cauchy;

        const int nf = static_cast<int>(free_indices.size());

        // Compute reduced gradient at Cauchy point: r_F[j] = g[F[j]] + (B*(x_c - x))[F[j]]
        Bd_ = B.multiply((x_cauchy - x).eval());
        r_F_.resize(nf);
        for(int j = 0; j < nf; ++j)
            r_F_[j] = g[free_indices[static_cast<std::size_t>(j)]] + Bd_[free_indices[static_cast<std::size_t>(j)]];

        // Build reduced Hessian B_FF
        B_FF_ = B.reduced_hessian(free_indices);

        // Solve B_FF * d_hat = -r_F using LDLT (max-bounded N×N)
        ldlt_.compute(B_FF_);

        if(ldlt_.info() == Eigen::Success && ldlt_.isPositive())
        {
            d_hat_.resize(nf);
            d_hat_ = ldlt_.solve(-r_F_);
        }
        else
        {
            d_hat_ = -r_F_ / B.theta();
        }

        // Assemble full step: x_new = x_cauchy + d (d_i = 0 for fixed variables)
        x_new_ = x_cauchy;
        for(int j = 0; j < nf; ++j)
            x_new_[free_indices[static_cast<std::size_t>(j)]] += d_hat_[j];

        x_new_ = project(x_new_, lower, upper);

        return x_new_;
    }

private:
    // Max-bounded reduced Hessian: free-variable count varies per call but ≤ N.
    using reduced_matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, N, N>;
    using reduced_vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, N, 1>;

    reduced_vector Bd_;
    reduced_vector r_F_;
    reduced_vector d_hat_;
    Eigen::Vector<Scalar, N> x_new_;
    reduced_matrix B_FF_;
    Eigen::LDLT<reduced_matrix> ldlt_;
};

// Backward-compatible free function wrapper.
//
// Reference: N&W Section 16.6, p. 478, eq. 16.49.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = 7>
Eigen::Vector<Scalar, N> subspace_minimize(
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& x_cauchy,
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const std::vector<int>& free_indices,
    const compact_lbfgs<Scalar, N, M>& B)
{
    subspace_minimizer<Scalar, N> sm(static_cast<int>(x.size()));
    return sm.solve(x, x_cauchy, g, lower, upper, free_indices, B);
}

}

#endif
