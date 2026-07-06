#ifndef HPP_GUARD_ARGMIN_DETAIL_LDP_H
#define HPP_GUARD_ARGMIN_DETAIL_LDP_H

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

#include "argmin/detail/nnls.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <limits>
#include <cmath>

namespace argmin::detail
{

// Solve min ||x||^2 s.t. G*x >= h.
//
// G is (m x n), h is (m), x is (n) output, lambda is (m) output Lagrange
// multipliers (>= 0, complementary with the inequality constraints).
// nnls_A/b/x_vec/w are scratch vectors sized for the dual NNLS problem:
//   nnls_A must be at least ((n+1) x m),
//   nnls_b must be at least (n+1),
//   nnls_x_vec must be at least (m),
//   nnls_w must be at least (m).
//
// The LDP KKT condition x = G^T lambda gives lambda = u / fac where
// u is the dual NNLS primal and fac = 1 - h^T u; this routine returns
// that lambda directly so the caller can use it as the Lagrange
// multiplier vector for inequality-constrained-LS chains (LSI / LSEI /
// kraft QP) without re-solving a separate KKT system.
//
// Return code: 1 on success, 4 if the inequality system is incompatible
// (no feasible point, empty intersection of half-spaces).
//
// Reference: Lawson, C.L. & Hanson, R.J. (1974). Solving Least Squares
//            Problems. Ch. 23.4, Algorithm LDP, eq. 23.18 (KKT lambda).
template <typename Scalar, int M, int N>
int ldp(
    const Eigen::Matrix<Scalar, M, N>& G,
    const Eigen::Vector<Scalar, M>& h,
    Eigen::Vector<Scalar, N>& x,
    Eigen::Vector<Scalar, M>& lambda,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& nnls_A,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_b,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_x_vec,
    Eigen::Vector<Scalar, Eigen::Dynamic>& nnls_w,
    int m, int n)
{
    using std::sqrt;

    // Trivial case: no constraints -> x = 0 is the optimum, lambda is empty.
    if(m <= 0)
    {
        x.setZero();
        if(lambda.size() != 0) lambda.setZero();
        return 1;
    }

    if(lambda.size() != m) lambda.resize(m);
    lambda.setZero();

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

    // Incompatibility (empty half-space intersection) is certified by Farkas:
    // there exists u >= 0 with G^T u = 0 and h^T u = 1, i.e. the primal-
    // recovery residual ||G^T u|| = ||r.head(n)|| vanishes while fac = 1 -
    // h^T u collapses to 0. The former absolute tests (rnorm_sq <= eps ||
    // fac <= eps) mixed a squared residual against an unsquared epsilon and,
    // worse, fired on merely large-norm FEASIBLE solutions: when ||x|| is
    // large the recovery denominator fac shrinks toward eps even though the
    // constraints are perfectly compatible. Test the primal residual against
    // its natural scale ||G||_inf * ||u|| instead (which stays O(||u||) for a
    // feasible boundary solution and only vanishes for the Farkas
    // certificate), and reject on fac <= 0 -- the exact condition under which
    // the recovery x = (G^T u)/fac ceases to be finite and positive.
    const Scalar u_norm = nnls_x_vec.head(m).norm();
    // Only the leading m x n block of G is logical; callers (soc_seed_
    // projection) pass a worst-case-sized scratch whose trailing rows hold
    // stale data. Scope the scale proxy to the block ldp actually reads.
    const Scalar G_scale = G.topLeftCorner(m, n).cwiseAbs().maxCoeff();
    Scalar primal_resid_sq = Scalar(0);
    for(int i = 0; i < n; ++i)
        primal_resid_sq += r[i] * r[i];
    const Scalar farkas_ref = G_scale * u_norm;
    const bool farkas_incompatible =
        (u_norm > Scalar(0))
        && (primal_resid_sq <= eps * farkas_ref * farkas_ref);

    if(farkas_incompatible || fac <= Scalar(0))
        return 4;  // incompatible inequality constraints

    for(int i = 0; i < n; ++i)
        x[i] = r[i] / fac;

    // KKT lambda for the LDP min ||x||^2 s.t. G x >= h is the rescaled
    // dual NNLS primal: lambda = u / fac, lambda >= 0 by NNLS dual
    // feasibility plus fac > 0 verified above.
    //
    // Reference: Lawson & Hanson 1974 eq. 23.18.
    for(int j = 0; j < m; ++j)
        lambda[j] = nnls_x_vec[j] / fac;

    return 1;
}

}

#endif
