#ifndef HPP_GUARD_NABLAPP_DETAIL_QUADRATIC_MODEL_H
#define HPP_GUARD_NABLAPP_DETAIL_QUADRATIC_MODEL_H

// Quadratic interpolation model for derivative-free optimization.
//
// Represents Q(x) = c + g^T(x - x_base) + 0.5*(x - x_base)^T * H * (x - x_base)
// where the model interpolates f at m points. For m = 2n+1 (the default),
// the system is underdetermined; Powell's minimum Frobenius norm approach
// is used: minimize ||H||_F subject to interpolation conditions Q(y_i) = f_i.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Sections 2 (model construction) and 4 (model update).

#include <Eigen/Core>
#include <Eigen/SVD>

#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double>
struct quadratic_model
{
    Eigen::VectorX<Scalar> g;
    Eigen::MatrixX<Scalar> H;
    Scalar c{};
    Eigen::VectorX<Scalar> x_base;
};

// Build quadratic model from m interpolation points and their function values.
//
// The polynomial basis for Q(x) at shifted point s = x - x_base is:
//   1, s_1, ..., s_n, s_1^2, s_1*s_2, ..., s_n^2
// giving (n+1)(n+2)/2 coefficients total. For m < (n+1)(n+2)/2, the system
// is underdetermined. We use minimum-norm SVD solution which minimizes
// ||coefficients||_2, effectively minimizing ||H||_F among all interpolants.
//
// Reference: Powell 2009, Section 2.
template <typename Scalar = double>
quadratic_model<Scalar> build_model(
    const Eigen::MatrixX<Scalar>& Y,
    const Eigen::VectorX<Scalar>& f_values,
    const Eigen::VectorX<Scalar>& x_base)
{
    const int n = Y.rows();
    const int m = Y.cols();

    // Number of polynomial basis terms: 1 + n + n*(n+1)/2
    const int p = 1 + n + n * (n + 1) / 2;

    // Build the Vandermonde-like matrix Phi (m x p)
    // Columns: [1 | s | vech(s*s^T)] where s = y_i - x_base
    Eigen::MatrixX<Scalar> Phi(m, p);

    for(int i = 0; i < m; ++i)
    {
        Eigen::VectorX<Scalar> s = Y.col(i) - x_base;

        // Constant term
        Phi(i, 0) = Scalar(1);

        // Linear terms
        for(int j = 0; j < n; ++j)
            Phi(i, 1 + j) = s[j];

        // Quadratic terms: 0.5 * s_j * s_k for j <= k
        int idx = 1 + n;
        for(int j = 0; j < n; ++j)
        {
            for(int k = j; k < n; ++k)
            {
                Scalar factor = (j == k) ? Scalar(0.5) : Scalar(1);
                Phi(i, idx) = factor * s[j] * s[k];
                ++idx;
            }
        }
    }

    // Solve min ||theta||_2 s.t. Phi * theta = f_values via SVD
    auto svd = Phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(Scalar(1e-12));
    Eigen::VectorX<Scalar> theta = svd.solve(f_values);

    // Extract model coefficients from theta
    quadratic_model<Scalar> model;
    model.x_base = x_base;
    model.c = theta[0];
    model.g = theta.segment(1, n);

    // Reconstruct symmetric H from upper-triangular entries
    model.H.setZero(n, n);
    int idx = 1 + n;
    for(int j = 0; j < n; ++j)
    {
        for(int k = j; k < n; ++k)
        {
            model.H(j, k) = theta[idx];
            model.H(k, j) = theta[idx];
            ++idx;
        }
    }

    return model;
}

// Update model after replacing interpolation point k with a new point.
//
// Full rebuild is O(m * p^2) which for n=6, m=13 is negligible.
// A rank-1 update via Sherman-Morrison would be faster for large n but
// adds complexity not justified for n < 20.
//
// Reference: Powell 2009, Section 4.
template <typename Scalar = double>
void update_model(
    quadratic_model<Scalar>& model,
    const Eigen::MatrixX<Scalar>& Y,
    const Eigen::VectorX<Scalar>& f_values,
    [[maybe_unused]] int replaced_index,
    const Eigen::VectorX<Scalar>& x_base)
{
    model = build_model(Y, f_values, x_base);
}

// Evaluate model at point x.
//
// Q(x) = c + g^T * (x - x_base) + 0.5 * (x - x_base)^T * H * (x - x_base)
template <typename Scalar = double>
Scalar evaluate_model(const quadratic_model<Scalar>& model,
                      const Eigen::VectorX<Scalar>& x)
{
    Eigen::VectorX<Scalar> s = x - model.x_base;
    return model.c + model.g.dot(s) + Scalar(0.5) * s.dot(model.H * s);
}

// Model gradient at point x: g + H * (x - x_base).
template <typename Scalar = double>
Eigen::VectorX<Scalar> model_gradient(const quadratic_model<Scalar>& model,
                                       const Eigen::VectorX<Scalar>& x)
{
    return (model.g + model.H * (x - model.x_base)).eval();
}

}

#endif
