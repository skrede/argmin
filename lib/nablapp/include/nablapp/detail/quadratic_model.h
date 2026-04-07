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

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
struct quadratic_model
{
    Eigen::Vector<Scalar, N> g;
    Eigen::Matrix<Scalar, N, N> H;
    Scalar c{};
    Eigen::Vector<Scalar, N> x_base;

    // Lagrange polynomial values L_k(x_base) for each interpolation point k.
    // L_k(x) = 1 at point k and 0 at all other interpolation points.
    // At x_base, sum_k L_k(x_base) = 1 (partition of unity).
    //
    // Reference: Powell 2009, Section 2 — Lagrange polynomials for interpolation.
    Eigen::VectorXd lagrange_values;
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
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
quadratic_model<Scalar, N> build_model(
    const Eigen::Matrix<Scalar, N, P>& Y,
    const Eigen::Vector<Scalar, P>& f_values,
    const Eigen::Vector<Scalar, N>& x_base)
{
    const int n = Y.rows();
    const int m = Y.cols();

    // Number of polynomial basis terms: 1 + n + n*(n+1)/2
    const int p = 1 + n + n * (n + 1) / 2;

    // Phi is (m x p) where both m and p are derived quantities.
    // When N is compile-time, p = 1 + N + N*(N+1)/2 is also compile-time,
    // but m = P (number of interpolation points) varies. Use dynamic.
    static constexpr int Pcoeff = (N == nablapp::dynamic_dimension)
        ? nablapp::dynamic_dimension
        : 1 + N + N * (N + 1) / 2;
    Eigen::Matrix<Scalar, P, Pcoeff> Phi(m, p);

    for(int i = 0; i < m; ++i)
    {
        Eigen::Vector<Scalar, N> s = (Y.col(i) - x_base).eval();

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
    auto svd = Phi.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.setThreshold(Scalar(1e-12));
    Eigen::Vector<Scalar, Pcoeff> theta = svd.solve(f_values);

    // Extract model coefficients from theta
    quadratic_model<Scalar, N> model;
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

    // Compute Lagrange polynomial values L_k(x_base) for each interpolation point k.
    // For any function f, f(x_base) = sum_k L_k(x_base) * f(y_k). The Lagrange
    // values are lambda = pinv(Phi)^T * phi(x_base), which is m-dimensional.
    // Using SVD(Phi) = U * Sigma * V^T, this is U * Sigma^{-1} * V^T * phi(x_base).
    //
    // phi(x_base) = [1, 0, ..., 0] since x_base - x_base = 0 makes all linear
    // and quadratic terms vanish.
    //
    // Reference: Powell 2009, Section 2.
    {
        Eigen::Vector<Scalar, Pcoeff> phi_xbase = Eigen::Vector<Scalar, Pcoeff>::Zero(p);
        phi_xbase[0] = Scalar(1);

        const auto& sv = svd.singularValues();
        Scalar thr = svd.threshold() * sv[0];
        auto V = svd.matrixV();  // p x p
        auto U = svd.matrixU();  // m x m

        // w = V^T * phi(x_base), then scale by Sigma^{-1}, then multiply by U
        Eigen::VectorXd w = V.transpose() * phi_xbase.template cast<double>();
        int rank = 0;
        for(int i = 0; i < sv.size(); ++i)
        {
            if(sv[i] > thr)
            {
                w[i] /= sv[i];
                ++rank;
            }
            else
                w[i] = 0.0;
        }
        // Zero out components beyond singular value count
        for(int i = sv.size(); i < w.size(); ++i)
            w[i] = 0.0;

        model.lagrange_values = U.leftCols(w.size()).template cast<double>() * w;
    }

    return model;
}

// Compute Lagrange polynomial values at an arbitrary query point.
//
// L_k(x_query) = row k of pinv(Phi) applied to phi(x_query).
// Rebuilds the SVD — for n <= 20 this is negligible per Powell 2009.
//
// Reference: Powell 2009, Section 2.
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
Eigen::VectorXd compute_lagrange_at_point(
    const Eigen::Matrix<Scalar, N, P>& Y,
    const Eigen::Vector<Scalar, P>& f_values,
    const Eigen::Vector<Scalar, N>& x_base,
    const Eigen::Vector<Scalar, N>& x_query)
{
    const int n = Y.rows();
    const int m = Y.cols();
    const int p = 1 + n + n * (n + 1) / 2;

    static constexpr int Pcoeff = (N == nablapp::dynamic_dimension)
        ? nablapp::dynamic_dimension
        : 1 + N + N * (N + 1) / 2;
    Eigen::Matrix<Scalar, P, Pcoeff> Phi(m, p);

    for(int i = 0; i < m; ++i)
    {
        Eigen::Vector<Scalar, N> s = (Y.col(i) - x_base).eval();
        Phi(i, 0) = Scalar(1);
        for(int j = 0; j < n; ++j)
            Phi(i, 1 + j) = s[j];
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

    auto svd = Phi.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.setThreshold(Scalar(1e-12));

    // Build phi(x_query)
    Eigen::Vector<Scalar, N> sq = (x_query - x_base).eval();
    Eigen::Vector<Scalar, Pcoeff> phi_q = Eigen::Vector<Scalar, Pcoeff>::Zero(p);
    phi_q[0] = Scalar(1);
    for(int j = 0; j < n; ++j)
        phi_q[1 + j] = sq[j];
    int idx = 1 + n;
    for(int j = 0; j < n; ++j)
    {
        for(int k = j; k < n; ++k)
        {
            Scalar factor = (j == k) ? Scalar(0.5) : Scalar(1);
            phi_q[idx] = factor * sq[j] * sq[k];
            ++idx;
        }
    }

    // L = pinv(Phi) * phi(x_query), computed via SVD
    const auto& sv = svd.singularValues();
    Scalar thr = svd.threshold() * sv[0];
    auto V = svd.matrixV();
    auto U = svd.matrixU();

    Eigen::VectorXd w = V.transpose() * phi_q.template cast<double>();
    for(int i = 0; i < sv.size(); ++i)
    {
        if(sv[i] > thr)
            w[i] /= sv[i];
        else
            w[i] = 0.0;
    }
    for(int i = sv.size(); i < w.size(); ++i)
        w[i] = 0.0;

    return U.leftCols(w.size()).template cast<double>() * w;
}

// Update model after replacing interpolation point k with a new point.
//
// Full rebuild is O(m * p^2) which for n=6, m=13 is negligible.
// A rank-1 update via Sherman-Morrison would be faster for large n but
// adds complexity not justified for n < 20.
//
// Reference: Powell 2009, Section 4.
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int P = nablapp::dynamic_dimension>
void update_model(
    quadratic_model<Scalar, N>& model,
    const Eigen::Matrix<Scalar, N, P>& Y,
    const Eigen::Vector<Scalar, P>& f_values,
    [[maybe_unused]] int replaced_index,
    const Eigen::Vector<Scalar, N>& x_base)
{
    model = build_model<Scalar, N>(Y, f_values, x_base);
}

// Evaluate model at point x.
//
// Q(x) = c + g^T * (x - x_base) + 0.5 * (x - x_base)^T * H * (x - x_base)
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Scalar evaluate_model(const quadratic_model<Scalar, N>& model,
                      const Eigen::Vector<Scalar, N>& x)
{
    Eigen::Vector<Scalar, N> s = (x - model.x_base).eval();
    return model.c + model.g.dot(s) + Scalar(0.5) * s.dot(model.H * s);
}

// Model gradient at point x: g + H * (x - x_base).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> model_gradient(const quadratic_model<Scalar, N>& model,
                                         const Eigen::Vector<Scalar, N>& x)
{
    return (model.g + model.H * (x - model.x_base)).eval();
}

}

#endif
