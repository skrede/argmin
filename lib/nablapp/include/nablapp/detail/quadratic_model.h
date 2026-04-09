#ifndef HPP_GUARD_NABLAPP_DETAIL_QUADRATIC_MODEL_H
#define HPP_GUARD_NABLAPP_DETAIL_QUADRATIC_MODEL_H

// Quadratic interpolation model for derivative-free optimization.
//
// Represents Q(x) = c + g^T(x - x_base) + 0.5*(x - x_base)^T * H * (x - x_base)
// where the model interpolates f at m points. For m = 2n+1 (the default),
// the system is underdetermined; Powell's minimum Frobenius norm approach
// is used: minimize ||H||_F subject to interpolation conditions Q(y_i) = f_i.
//
// The incremental update path (compute_lagrange_incremental, update_model_incremental)
// eliminates the per-step O(m*p^2) SVD by storing the pseudoinverse of Phi and
// the basis matrix Phi itself. Per-step cost becomes O(m^2*p) via LDLT-based
// pseudoinverse refresh, yielding ~2x speedup at IK scale (n=6, m=13).
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06,
//            Sections 2 (model construction) and 4 (model update).

#include "nablapp/types.h"

#include <Eigen/Cholesky>
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

    // Polynomial basis matrix Phi (m x p) where p = 1 + n + n*(n+1)/2.
    // Row i contains the polynomial basis vector evaluated at Y[:,i] - x_base.
    // Stored to enable O(m^2*p) pseudoinverse refresh on point replacement.
    Eigen::MatrixX<Scalar> Phi;

    // Pseudoinverse of Phi: pinv_Phi = (Phi * Phi^T)^{-1} * Phi, dimensions m x p.
    // Used to evaluate L_k(x_query) = pinv_Phi * phi(x_query) in O(m*p).
    // Initialized by build_model() from SVD; refreshed via LDLT after point replacement.
    Eigen::MatrixX<Scalar> pinv_Phi;

    // Active number of interpolation points (runtime).
    int m_points{0};
};

// Compute polynomial basis dimension p = 1 + n + n*(n+1)/2.
inline int polynomial_basis_dimension(int n)
{
    return 1 + n + n * (n + 1) / 2;
}

// Build the polynomial basis vector phi(x) at shifted point s = x - x_base.
//
// Basis: [1, s_1, ..., s_n, 0.5*s_1^2, s_1*s_2, ..., 0.5*s_n^2]
// giving p = 1 + n + n*(n+1)/2 coefficients.
//
// Reference: Powell 2009, Section 2.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::VectorX<Scalar> build_polynomial_basis(
    const Eigen::Vector<Scalar, N>& s,
    int p)
{
    const int n = s.size();
    Eigen::VectorX<Scalar> phi(p);

    phi[0] = Scalar(1);

    for(int j = 0; j < n; ++j)
        phi[1 + j] = s[j];

    int idx = 1 + n;
    for(int j = 0; j < n; ++j)
    {
        for(int k = j; k < n; ++k)
        {
            Scalar factor = (j == k) ? Scalar(0.5) : Scalar(1);
            phi[idx] = factor * s[j] * s[k];
            ++idx;
        }
    }

    return phi;
}

// Refresh the pseudoinverse of Phi via LDLT decomposition of the Gram matrix.
//
// pinv_Phi = (Phi * Phi^T)^{-1} * Phi, computed as G.ldlt().solve(Phi)
// where G = Phi * Phi^T is m x m. Cost: O(m^2*p + m^3).
//
// Reference: Standard pseudoinverse formula for full row-rank matrices.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void refresh_pseudoinverse(quadratic_model<Scalar, N>& model)
{
    const int m = model.m_points;
    Eigen::MatrixX<Scalar> G = model.Phi.topRows(m) * model.Phi.topRows(m).transpose();
    model.pinv_Phi.topRows(m) = G.ldlt().solve(model.Phi.topRows(m));
}

// Build quadratic model from m interpolation points and their function values.
//
// The polynomial basis for Q(x) at shifted point s = x - x_base is:
//   1, s_1, ..., s_n, s_1^2, s_1*s_2, ..., s_n^2
// giving (n+1)(n+2)/2 coefficients total. For m < (n+1)(n+2)/2, the system
// is underdetermined. We use minimum-norm SVD solution which minimizes
// ||coefficients||_2, effectively minimizing ||H||_F among all interpolants.
//
// Additionally initializes incremental data structures (Phi, pinv_Phi,
// lagrange_values) for use by update_model_incremental().
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
        Phi.row(i) = build_polynomial_basis<Scalar, N>(s, p).transpose();
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

        model.lagrange_values = U.template cast<double>() * w.head(m);
    }

    // Initialize incremental data structures.
    // Store Phi and compute pinv_Phi from SVD: pinv = V * Sigma^{-1} * U^T.
    model.Phi = Phi;
    model.m_points = m;

    {
        const auto& U = svd.matrixU();
        const auto& V = svd.matrixV();
        const auto& sv = svd.singularValues();
        const int rank = svd.rank();

        // pinv_Phi = V * diag(1/sigma_i) * U^T  (transposed to m x p)
        Eigen::MatrixX<Scalar> Sigma_inv = Eigen::MatrixX<Scalar>::Zero(rank, rank);
        for(int i = 0; i < rank; ++i)
            Sigma_inv(i, i) = Scalar(1) / sv[i];

        model.pinv_Phi = (V.leftCols(rank) * Sigma_inv * U.leftCols(rank).transpose()).transpose();
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

    return U.template cast<double>() * w.head(m);
}

// Update model after replacing interpolation point k with a new point.
//
// Full rebuild is O(m * p^2) which for n=6, m=13 is negligible.
// Retained for rescue path and reference.
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

// Compute L_k(x_query) for all k using stored pinv_Phi, without SVD.
//
// L(x_query) = pinv_Phi * phi(x_query), an O(m*p) matrix-vector product.
//
// Reference: Powell 2009, Section 2 (Lagrange polynomial evaluation).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::VectorX<Scalar> compute_lagrange_incremental(
    const quadratic_model<Scalar, N>& model,
    const Eigen::Vector<Scalar, N>& x_base,
    const Eigen::Vector<Scalar, N>& x_query)
{
    const int n = x_query.size();
    const int p = polynomial_basis_dimension(n);
    const int m = model.m_points;

    Eigen::Vector<Scalar, N> s = (x_query - x_base).eval();
    Eigen::VectorX<Scalar> phi = build_polynomial_basis<Scalar, N>(s, p);

    return (model.pinv_Phi.topRows(m) * phi).eval();
}

// Extract model coefficients (c, g, H) from the minimum-norm solution
// theta = Phi^T * (Phi * Phi^T)^{-1} * f = Phi^T * pinv_Phi_transpose * f.
//
// Since pinv_Phi = (Phi * Phi^T)^{-1} * Phi (m x p), and the right
// pseudoinverse is Phi^+ = Phi^T * (Phi * Phi^T)^{-1} = pinv_Phi^T (p x m),
// we compute theta = pinv_Phi^T * f.
//
// Reference: Powell 2009, Section 2 (minimum Frobenius norm model).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void extract_model_from_pseudoinverse(
    quadratic_model<Scalar, N>& model,
    const Eigen::VectorX<Scalar>& f_values)
{
    const int m = model.m_points;
    const int n = model.g.size();

    // theta = Phi^+ * f = pinv_Phi^T * f
    Eigen::VectorX<Scalar> theta = model.pinv_Phi.topRows(m).transpose() * f_values;

    model.c = theta[0];
    model.g = theta.segment(1, n);

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
}

// Incremental model and Lagrange update after point replacement.
//
// Replaces both compute_lagrange_at_point() and update_model() in the
// normal step path. SVD-free: O(m*p + m^2*p) per call via LDLT-based
// pseudoinverse refresh.
//
// Steps:
//   1. Compute lv_xnew = L_k(x_new) via pinv_Phi * phi(x_new).
//   2. Check |lv_xnew[t]| > epsilon to avoid denominator collapse.
//   3. Update Phi row t and refresh pinv_Phi via LDLT.
//   4. Recompute model coefficients from updated pseudoinverse.
//   5. Recompute Lagrange values at x_base from refreshed pinv_Phi.
//
// Returns lv_xnew (Lagrange values at x_new). Empty vector signals
// denominator collapse; caller should fall back to full SVD rebuild.
//
// Reference: Powell 2009, Section 4 (point replacement and model update).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
Eigen::VectorX<Scalar> update_model_incremental(
    quadratic_model<Scalar, N>& model,
    const Eigen::VectorX<Scalar>& f_values,
    const Eigen::Vector<Scalar, N>& x_new,
    int replaced_index,
    const Eigen::Vector<Scalar, N>& x_base)
{
    const int n = x_new.size();
    const int p = polynomial_basis_dimension(n);
    const int m = model.m_points;
    const int t = replaced_index;

    // Step 1: Compute L_k(x_new) for all k via stored pinv_Phi.
    Eigen::VectorX<Scalar> lv_xnew = compute_lagrange_incremental(model, x_base, x_new);

    // Step 2: Check denominator -- |L_t(x_new)| must be non-negligible.
    if(std::abs(lv_xnew[t]) < Scalar(1e-12))
        return Eigen::VectorX<Scalar>{};

    // Step 3: Update Phi row t and refresh pseudoinverse via LDLT.
    Eigen::Vector<Scalar, N> s_new = (x_new - x_base).eval();
    model.Phi.row(t) = build_polynomial_basis<Scalar, N>(s_new, p).transpose();
    refresh_pseudoinverse(model);

    // Step 4: Recompute model coefficients from updated pseudoinverse and
    // the caller's authoritative f_values (already updated at index t).
    extract_model_from_pseudoinverse(model, f_values);

    // Step 5: Recompute Lagrange values at x_base from refreshed pinv_Phi.
    Eigen::VectorX<Scalar> phi_zero = Eigen::VectorX<Scalar>::Zero(p);
    phi_zero[0] = Scalar(1);
    model.lagrange_values = model.pinv_Phi.topRows(m) * phi_zero;

    return lv_xnew;
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
