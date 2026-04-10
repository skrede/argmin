#ifndef HPP_GUARD_NABLAPP_DETAIL_LDP_H
#define HPP_GUARD_NABLAPP_DETAIL_LDP_H

// Least distance programming (LDP).
//
// Solves: min ||x||^2  s.t.  G*x >= h
//
// Formulates the LDP as a dual non-negative least-squares problem and
// hands it off to nnls(). After NNLS returns, the primal solution x is
// recovered from the NNLS residual.
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Prentice-Hall. Chapter 23, Section 23.4 (LDP).

#include "nablapp/detail/nnls.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <limits>
#include <cmath>

namespace nablapp::detail
{

// Solve min ||x||^2 s.t. G*x >= h.
//
// G is (m x n), h is (m), x is (n) output.
// nnls_A/b/x_vec/w are scratch vectors sized for the dual NNLS problem:
//   nnls_A must be at least ((n+1) x m),
//   nnls_b must be at least (n+1),
//   nnls_x_vec must be at least (m),
//   nnls_w must be at least (m).
//
// Return code: 1 on success, 4 if the inequality system is incompatible
// (no feasible point, empty intersection of half-spaces).
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Ch. 23.4, Algorithm LDP.
template <typename Scalar, int M, int N>
int ldp(
    const Eigen::Matrix<Scalar, M, N>& G,
    const Eigen::Vector<Scalar, M>& h,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& nnls_A,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_b,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_x_vec,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_w,
    int m, int n)
{
    using std::sqrt;

    // Trivial case: no constraints -> x = 0 is the optimum.
    if(m <= 0)
    {
        x.setZero();
        return 1;
    }

    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    // Build dual NNLS matrix E = [G^T; h^T] of shape (n+1) x m and
    // target f = [0, ..., 0, 1] of length (n+1).
    nnls_A.resize(n + 1, m);
    nnls_b.resize(n + 1);
    nnls_x_vec.resize(m);
    nnls_w.resize(m);

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            nnls_A(i, j) = G(j, i);
    for(int j = 0; j < m; ++j)
        nnls_A(n, j) = h[j];

    nnls_b.setZero();
    nnls_b[n] = Scalar(1);
    nnls_x_vec.setZero();
    nnls_w.setZero();

    auto result = nnls<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
        nnls_A, nnls_b, nnls_x_vec, nnls_w, n + 1, m);

    if(result.mode != 1)
        return 4;

    // Recover primal from NNLS residual.
    //
    //   E = [G^T; h^T] of shape (n+1) x m
    //   f = [0, ..., 0, 1]
    //   r = E*u - f,  where u = nnls_x_vec
    //
    // The KKT conditions of the primal give x = G^T * lambda with
    // lambda = u / (1 - h^T u). With fac = 1 - h^T u = -r[n], the
    // component formula is
    //
    //   x_i = (G^T u)_i / fac = r[i] / fac   for i = 0..n-1.
    //
    // If the residual is numerically zero (or fac is non-positive) the
    // constraints are incompatible (no strictly feasible x exists).
    Eigen::Vector<Scalar, Eigen::Dynamic> r(n + 1);
    for(int i = 0; i < n; ++i)
    {
        Scalar s = Scalar(0);
        for(int j = 0; j < m; ++j)
            s += G(j, i) * nnls_x_vec[j];
        r[i] = s;  // f_i = 0 for i < n, so r_i = (E u)_i
    }
    {
        Scalar s = Scalar(0);
        for(int j = 0; j < m; ++j)
            s += h[j] * nnls_x_vec[j];
        r[n] = s - Scalar(1);
    }

    // fac = -r[n] = 1 - h^T u
    const Scalar fac = -r[n];
    const Scalar rnorm_sq = r.squaredNorm();

    if(rnorm_sq <= eps || fac <= eps)
        return 4;  // incompatible inequality constraints

    for(int i = 0; i < n; ++i)
        x[i] = r[i] / fac;

    return 1;
}

}

#endif
