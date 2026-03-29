#ifndef HPP_GUARD_NABLAPP_DETAIL_ACTIVE_SET_QP_H
#define HPP_GUARD_NABLAPP_DETAIL_ACTIVE_SET_QP_H

// Dense active-set QP solver.
//
// Solves: min 0.5 * x^T G x + d^T x
//         s.t. A_eq x = b_eq           (equality)
//              A_ineq x >= b_ineq      (inequality)
//
// Reference: N&W Algorithm 16.1, pp. 460-463 (active-set method for convex QP)
//            N&W Section 16.2, eq. 16.16-16.19 (null-space method)
//            N&W eq. 16.29 (blocking step length)
//            N&W Section 16.5 (indefinite QP extensions)

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace nablapp::detail
{

enum class qp_status { optimal, infeasible, max_iterations, indefinite_hessian };

template <typename Scalar = double>
struct qp_result
{
    Eigen::VectorX<Scalar> x;
    Eigen::VectorX<Scalar> lambda;
    qp_status status{qp_status::optimal};
    int iterations{0};
};

template <typename Scalar = double>
struct qp_options
{
    int max_iterations{200};
    Scalar tolerance{Scalar(1e-12)};
};

// Determine initial working set from feasible point x0.
// Equality constraints (indices 0..n_eq-1) are always active.
// Inequality constraints active if |a_i^T x0 - b_i| < tolerance.
//
// Reference: N&W Section 16.4, p. 460.
template <typename Scalar>
std::vector<int> initial_working_set(
    const Eigen::VectorX<Scalar>& x0,
    const Eigen::MatrixX<Scalar>& A_full,
    const Eigen::VectorX<Scalar>& b_full,
    int n_eq,
    Scalar tolerance)
{
    std::vector<int> W;
    const int m = static_cast<int>(A_full.rows());
    W.reserve(m);

    for(int i = 0; i < n_eq; ++i)
        W.push_back(i);

    for(int i = n_eq; i < m; ++i)
    {
        Scalar residual = A_full.row(i).dot(x0) - b_full[i];
        if(std::abs(residual) < tolerance)
            W.push_back(i);
    }
    return W;
}

// Solve equality-constrained QP subproblem via null-space method.
//
// min 0.5 p^T G p + g_k^T p   s.t.  A_W p = 0
//
// Uses ColPivHouseholderQR of A_W^T for rank-revealing null-space computation.
// Uses LDLT for reduced Hessian Z^T G Z solve.
//
// Returns (step p, multipliers lambda_W for working set constraints).
//
// Reference: N&W Section 16.2, eq. 16.16-16.19.
template <typename Scalar>
std::pair<Eigen::VectorX<Scalar>, Eigen::VectorX<Scalar>> solve_equality_qp(
    const Eigen::MatrixX<Scalar>& G,
    const Eigen::VectorX<Scalar>& g_k,
    const Eigen::MatrixX<Scalar>& A_W)
{
    const int n = G.rows();
    const int m = A_W.rows();

    // No active constraints: unconstrained sub-step
    if(m == 0)
    {
        Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(G);
        Eigen::VectorX<Scalar> p = ldlt.solve(-g_k);
        return {p, Eigen::VectorX<Scalar>{}};
    }

    // Fully constrained: p = 0, compute multipliers only
    if(m >= n)
    {
        Eigen::VectorX<Scalar> p = Eigen::VectorX<Scalar>::Zero(n);
        // lambda from (A_W)^T lambda = g_k  (eq. 16.30: sum a_i lambda_i = g)
        auto qr = A_W.transpose().colPivHouseholderQr();
        Eigen::VectorX<Scalar> lambda = qr.solve(g_k);
        return {p, lambda};
    }

    // QR factorization of A_W^T to get null-space basis Z.
    // A_W^T = Q * [R; 0], so Z = last (n-m) columns of Q.
    // Reference: N&W eq. 16.37.
    Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(A_W.transpose());
    const int rank = qr.rank();

    Eigen::MatrixX<Scalar> Q = qr.householderQ() *
        Eigen::MatrixX<Scalar>::Identity(n, n);

    // Y = first rank columns, Z = remaining columns
    Eigen::MatrixX<Scalar> Y = Q.leftCols(rank);
    Eigen::MatrixX<Scalar> Z = Q.rightCols(n - rank);

    // Solve for p_y: A_W * Y * p_y = 0 (since constraints are A_W p = 0)
    // => p_y = 0 (the subproblem has RHS = 0 because we work with p = x - x_k)
    // So p = Z * p_z where (Z^T G Z) p_z = -Z^T g_k  (eq. 16.18)

    Eigen::MatrixX<Scalar> ZtGZ = Z.transpose() * G * Z;
    Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(ZtGZ);

    // Check inertia: if reduced Hessian is not positive (semi)definite,
    // the subproblem direction may point to a saddle.
    // For convex QP this won't happen; for indefinite we fall through.
    Eigen::VectorX<Scalar> rhs = -(Z.transpose() * g_k);
    Eigen::VectorX<Scalar> p_z = ldlt.solve(rhs);
    Eigen::VectorX<Scalar> p = Z * p_z;

    // Multipliers: (A_W Y)^T lambda = Y^T (g_k + G p)  (eq. 16.19)
    Eigen::MatrixX<Scalar> AY = A_W * Y;
    Eigen::VectorX<Scalar> rhs_lam = Y.transpose() * (g_k + G * p);
    auto qr_ay = AY.transpose().colPivHouseholderQr();
    Eigen::VectorX<Scalar> lambda = qr_ay.solve(rhs_lam);

    return {p, lambda};
}

// Compute blocking step length per N&W eq. 16.29.
//
// alpha_k = min(1, min_{i not in W, a_i^T p < 0} (b_i - a_i^T x) / (a_i^T p))
//
// Returns (alpha, blocking constraint index). alpha=1 and index=-1 if no blocking.
template <typename Scalar>
std::pair<Scalar, int> blocking_step_length(
    const Eigen::VectorX<Scalar>& x_k,
    const Eigen::VectorX<Scalar>& p_k,
    const Eigen::MatrixX<Scalar>& A_full,
    const Eigen::VectorX<Scalar>& b_full,
    const std::vector<int>& working_set,
    int n_eq)
{
    Scalar alpha = Scalar(1);
    int blocking_idx = -1;
    const int m = static_cast<int>(A_full.rows());

    for(int i = n_eq; i < m; ++i)
    {
        // Skip constraints already in working set
        bool in_W = false;
        for(int w : working_set)
        {
            if(w == i) { in_W = true; break; }
        }
        if(in_W) continue;

        Scalar atp = A_full.row(i).dot(p_k);
        if(atp < Scalar(0))
        {
            Scalar slack = A_full.row(i).dot(x_k) - b_full[i];
            Scalar ratio = -slack / atp;
            if(ratio < alpha)
            {
                alpha = ratio;
                blocking_idx = i;
            }
        }
    }

    alpha = std::max(alpha, Scalar(0));
    return {alpha, blocking_idx};
}

// Find the inequality constraint in working set with the most negative multiplier.
// Equality constraints (indices < n_eq) are never candidates for removal.
// Returns -1 if all inequality multipliers >= -tolerance.
//
// Reference: N&W Algorithm 16.1 (multiplier check step).
template <typename Scalar>
int most_negative_multiplier(
    const Eigen::VectorX<Scalar>& lambda_W,
    const std::vector<int>& working_set,
    int n_eq,
    Scalar tolerance)
{
    Scalar min_val = -tolerance;
    int drop_idx = -1;
    const int wsize = static_cast<int>(working_set.size());

    for(int k = 0; k < wsize; ++k)
    {
        if(working_set[k] < n_eq) continue;
        if(k < lambda_W.size() && lambda_W[k] < min_val)
        {
            min_val = lambda_W[k];
            drop_idx = k;
        }
    }
    return drop_idx;
}

// Check if LDLT factorization reveals indefiniteness.
// Returns true if reduced Hessian has negative diagonal entries.
//
// Reference: N&W Section 16.5 (detecting indefiniteness via LDL^T).
template <typename Scalar>
bool check_inertia(const Eigen::LDLT<Eigen::MatrixX<Scalar>>& ldlt)
{
    auto D = ldlt.vectorD();
    for(int i = 0; i < D.size(); ++i)
    {
        if(D[i] < Scalar(0)) return true;
    }
    return false;
}

// Extract rows of A_full corresponding to working set indices.
template <typename Scalar>
Eigen::MatrixX<Scalar> extract_working_rows(
    const Eigen::MatrixX<Scalar>& A_full,
    const std::vector<int>& working_set)
{
    const int n = A_full.cols();
    const int wsize = static_cast<int>(working_set.size());
    Eigen::MatrixX<Scalar> A_W(wsize, n);
    for(int k = 0; k < wsize; ++k)
        A_W.row(k) = A_full.row(working_set[k]);
    return A_W;
}

// Build full multiplier vector from working set multipliers.
// Non-active constraint multipliers are set to zero.
template <typename Scalar>
Eigen::VectorX<Scalar> build_full_lambda(
    const Eigen::VectorX<Scalar>& lambda_W,
    const std::vector<int>& working_set,
    int total_constraints)
{
    Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(total_constraints);
    const int wsize = static_cast<int>(working_set.size());
    for(int k = 0; k < wsize && k < lambda_W.size(); ++k)
        lambda[working_set[k]] = lambda_W[k];
    return lambda;
}

// Dense active-set QP solver.
//
// Solves: min 0.5 x^T G x + d^T x
//         s.t. A_eq x = b_eq             (equality)
//              A_ineq x >= b_ineq        (inequality, lower-bound form)
//
// Convention: inequality constraints are A_ineq * x >= b_ineq.
// The initial point x0 must be feasible.
//
// Reference: N&W Algorithm 16.1, pp. 460-463.
template <typename Scalar = double>
qp_result<Scalar> solve_qp(
    const Eigen::MatrixX<Scalar>& G,
    const Eigen::VectorX<Scalar>& d,
    const Eigen::MatrixX<Scalar>& A_eq,
    const Eigen::VectorX<Scalar>& b_eq,
    const Eigen::MatrixX<Scalar>& A_ineq,
    const Eigen::VectorX<Scalar>& b_ineq,
    const Eigen::VectorX<Scalar>& x0,
    const qp_options<Scalar>& opts = {})
{
    const int n = G.rows();
    const int m_eq = A_eq.rows();
    const int m_ineq = A_ineq.rows();
    const int m_total = m_eq + m_ineq;

    // Assemble full constraint matrix: [A_eq; A_ineq] x >= [b_eq; b_ineq]
    // (Equality constraints treated as a_i^T x = b_i, always active.)
    Eigen::MatrixX<Scalar> A_full(m_total, n);
    Eigen::VectorX<Scalar> b_full(m_total);
    if(m_eq > 0)
    {
        A_full.topRows(m_eq) = A_eq;
        b_full.head(m_eq) = b_eq;
    }
    if(m_ineq > 0)
    {
        A_full.bottomRows(m_ineq) = A_ineq;
        b_full.tail(m_ineq) = b_ineq;
    }

    auto W = initial_working_set(x0, A_full, b_full, m_eq, opts.tolerance);
    Eigen::VectorX<Scalar> x = x0;

    for(int iter = 0; iter < opts.max_iterations; ++iter)
    {
        Eigen::VectorX<Scalar> g_k = G * x + d;
        Eigen::MatrixX<Scalar> A_W = extract_working_rows(A_full, W);

        auto [p, lambda_W] = solve_equality_qp(G, g_k, A_W);

        if(p.norm() < opts.tolerance)
        {
            // At stationary point for current working set -- check multipliers
            int drop = most_negative_multiplier(lambda_W, W, m_eq, opts.tolerance);
            if(drop == -1)
            {
                // All inequality multipliers non-negative: optimal
                auto lambda_full = build_full_lambda(lambda_W, W, m_total);
                qp_result<Scalar> res;
                res.x = x;
                res.lambda = lambda_full;
                res.status = qp_status::optimal;
                res.iterations = iter;
                return res;
            }
            // Remove the constraint with most negative multiplier
            W.erase(W.begin() + drop);
        }
        else
        {
            // Step is nonzero: compute blocking step length (eq. 16.29)
            auto [alpha, blocking_idx] = blocking_step_length(
                x, p, A_full, b_full, W, m_eq);
            x += alpha * p;

            if(alpha < Scalar(1) && blocking_idx >= 0)
                W.push_back(blocking_idx);
        }
    }

    Eigen::VectorX<Scalar> zero_lam = Eigen::VectorX<Scalar>::Zero(
        static_cast<int>(W.size()));
    auto lambda_full = build_full_lambda(zero_lam, W, m_total);
    qp_result<Scalar> res;
    res.x = x;
    res.lambda = lambda_full;
    res.status = qp_status::max_iterations;
    res.iterations = opts.max_iterations;
    return res;
}

// Convenience overload with box constraints.
//
// Converts box bounds l <= x <= u to inequality constraints:
//   I*x >= l    (lower bounds)
//  -I*x >= -u   (upper bounds)
// and appends them to A_ineq/b_ineq.
template <typename Scalar = double>
qp_result<Scalar> solve_qp(
    const Eigen::MatrixX<Scalar>& G,
    const Eigen::VectorX<Scalar>& d,
    const Eigen::MatrixX<Scalar>& A_eq,
    const Eigen::VectorX<Scalar>& b_eq,
    const Eigen::MatrixX<Scalar>& A_ineq,
    const Eigen::VectorX<Scalar>& b_ineq,
    const Eigen::VectorX<Scalar>& lower,
    const Eigen::VectorX<Scalar>& upper,
    const Eigen::VectorX<Scalar>& x0,
    const qp_options<Scalar>& opts = {})
{
    const int n = G.rows();
    const int m_ineq = A_ineq.rows();
    const int m_box = 2 * n;

    // Build augmented inequality system: [A_ineq; I; -I] x >= [b_ineq; l; -u]
    Eigen::MatrixX<Scalar> A_aug(m_ineq + m_box, n);
    Eigen::VectorX<Scalar> b_aug(m_ineq + m_box);

    if(m_ineq > 0)
    {
        A_aug.topRows(m_ineq) = A_ineq;
        b_aug.head(m_ineq) = b_ineq;
    }

    A_aug.middleRows(m_ineq, n) = Eigen::MatrixX<Scalar>::Identity(n, n);
    b_aug.segment(m_ineq, n) = lower;

    A_aug.bottomRows(n) = -Eigen::MatrixX<Scalar>::Identity(n, n);
    b_aug.tail(n) = -upper;

    return solve_qp(G, d, A_eq, b_eq, A_aug, b_aug, x0, opts);
}

}

#endif
