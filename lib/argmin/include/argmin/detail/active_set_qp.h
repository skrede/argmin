#ifndef HPP_GUARD_ARGMIN_DETAIL_ACTIVE_SET_QP_H
#define HPP_GUARD_ARGMIN_DETAIL_ACTIVE_SET_QP_H

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

#include "argmin/detail/givens_qr_update.h"
#include "argmin/types.h"
#include "argmin/options/qp_options.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace argmin::detail
{

enum class qp_status { optimal, infeasible, max_iterations, indefinite_hessian };

template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
struct qp_result
{
    Eigen::Vector<Scalar, N> x;
    Eigen::Vector<Scalar, M> lambda;
    qp_status status{qp_status::optimal};
    int iterations{0};

    // Set by kraft_lsq_qp_recovery_solver when the inner cascade returned
    // infeasible/rank-deficient and Kraft 1988 §3.4 augmented-QP recovery
    // produced the step. Range [0, 1]: 0 means recovery was unnecessary
    // (or the direct path succeeded); positive values give the slack
    // variable s at the augmented optimum, where s=1 corresponds to full
    // constraint relaxation. Direct path always returns 0.
    Scalar relaxation_factor{Scalar(0)};
};

// Determine initial working set from feasible point x0.
// Equality constraints (indices 0..n_eq-1) are always active.
// Inequality constraints active if |a_i^T x0 - b_i| < tolerance.
//
// Reference: N&W Section 16.4, p. 460.
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
std::vector<int> initial_working_set(
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
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
// Uses HouseholderQR of A_W^T for null-space computation (working set is
// full rank by construction, so rank-revealing is unnecessary).
// Uses LLT for reduced Hessian Z^T G Z solve (positive definite by construction).
//
// Returns (step p, multipliers lambda_W for working set constraints).
//
// Reference: N&W Section 16.2, eq. 16.16-16.19.
template <typename Scalar, int N = argmin::dynamic_dimension, int Mw = argmin::dynamic_dimension>
std::pair<Eigen::Vector<Scalar, N>, Eigen::Vector<Scalar, Mw>> solve_equality_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& g_k,
    const Eigen::Matrix<Scalar, Mw, N>& A_W)
{
    const int n = G.rows();
    const int m = A_W.rows();

    // No active constraints: unconstrained sub-step
    if(m == 0)
    {
        Eigen::LLT<Eigen::Matrix<Scalar, N, N>> llt(G);
        Eigen::Vector<Scalar, N> p = llt.solve(-g_k);
        return {p, Eigen::Vector<Scalar, Mw>{}};
    }

    // Fully constrained: p = 0, compute multipliers only
    if(m >= n)
    {
        Eigen::Vector<Scalar, N> p = Eigen::Vector<Scalar, N>::Zero(n);
        // lambda from (A_W)^T lambda = g_k  (eq. 16.30: sum a_i lambda_i = g)
        auto qr = A_W.transpose().householderQr();
        Eigen::Vector<Scalar, Mw> lambda = qr.solve(g_k);
        return {p, lambda};
    }

    // QR factorization of A_W^T to get null-space basis Z.
    // A_W^T = Q * [R; 0], so Z = last (n-m) columns of Q.
    // Reference: N&W eq. 16.37.
    Eigen::HouseholderQR<Eigen::Matrix<Scalar, N, Mw>> qr(A_W.transpose());
    const int rank = m;  // working set is full rank by construction
    const int nz = n - rank;

    // Materialize only the Q columns we need via thin products.
    // Q * [0; I_{nz}] gives the null-space basis Z (n x nz).
    // Q * [I_{rank}; 0] gives the range-space basis Y (n x rank).
    Eigen::Matrix<Scalar, N, Eigen::Dynamic> Z(n, nz);
    Z.setZero();
    Z.bottomRightCorner(nz, nz).setIdentity();
    Z.applyOnTheLeft(qr.householderQ());

    Eigen::Matrix<Scalar, N, Eigen::Dynamic> Y(n, rank);
    Y.setZero();
    Y.topLeftCorner(rank, rank).setIdentity();
    Y.applyOnTheLeft(qr.householderQ());

    // Solve for p_y: A_W * Y * p_y = 0 (since constraints are A_W p = 0)
    // => p_y = 0 (the subproblem has RHS = 0 because we work with p = x - x_k)
    // So p = Z * p_z where (Z^T G Z) p_z = -Z^T g_k  (eq. 16.18)

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ZtGZ(nz, nz);
    ZtGZ.noalias() = Z.transpose() * G * Z;
    Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> llt(std::move(ZtGZ));

    // Check inertia: if reduced Hessian is not positive (semi)definite,
    // the subproblem direction may point to a saddle.
    // For convex QP this won't happen; for indefinite we fall through.
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs(nz);
    rhs.noalias() = -(Z.transpose() * g_k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> p_z = llt.solve(rhs);
    Eigen::Vector<Scalar, N> p(n);
    p.noalias() = Z * p_z;

    // Multipliers: (A_W Y)^T lambda = Y^T (g_k + G p)  (eq. 16.19)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs_lam(rank);
    rhs_lam.noalias() = Y.transpose() * (g_k + G * p);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> AY(m, rank);
    AY.noalias() = A_W * Y;
    auto qr_ay = AY.transpose().householderQr();
    Eigen::Vector<Scalar, Mw> lambda = qr_ay.solve(rhs_lam);

    return {p, lambda};
}

// Compute blocking step length per N&W eq. 16.29.
//
// alpha_k = min(1, min_{i not in W, a_i^T p < 0} (b_i - a_i^T x) / (a_i^T p))
//
// Returns (alpha, blocking constraint index). alpha=1 and index=-1 if no blocking.
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
std::pair<Scalar, int> blocking_step_length(
    const Eigen::Vector<Scalar, N>& x_k,
    const Eigen::Vector<Scalar, N>& p_k,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
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
template <typename Scalar, int Mw = argmin::dynamic_dimension>
int most_negative_multiplier(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
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

// Extract rows of A_full corresponding to working set indices.
template <typename Scalar, int M = argmin::dynamic_dimension, int N = argmin::dynamic_dimension>
Eigen::Matrix<Scalar, Eigen::Dynamic, N> extract_working_rows(
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const std::vector<int>& working_set)
{
    const int n = A_full.cols();
    const int wsize = static_cast<int>(working_set.size());
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_W(wsize, n);
    for(int k = 0; k < wsize; ++k)
        A_W.row(k) = A_full.row(working_set[k]);
    return A_W;
}

// Build full multiplier vector from working set multipliers.
// Non-active constraint multipliers are set to zero.
template <typename Scalar, int Mw = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
Eigen::Vector<Scalar, M> build_full_lambda(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
    const std::vector<int>& working_set,
    int total_constraints)
{
    Eigen::Vector<Scalar, M> lambda = Eigen::Vector<Scalar, M>::Zero(total_constraints);
    const int wsize = static_cast<int>(working_set.size());
    for(int k = 0; k < wsize && k < lambda_W.size(); ++k)
        lambda[working_set[k]] = lambda_W[k];
    return lambda;
}

// In-place overload: writes into pre-allocated output vector.
template <typename Scalar, int Mw = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
void build_full_lambda(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
    const std::vector<int>& working_set,
    int total_constraints,
    Eigen::Vector<Scalar, M>& lambda_full)
{
    lambda_full.setZero(total_constraints);
    const int wsize = static_cast<int>(working_set.size());
    for(int k = 0; k < wsize && k < lambda_W.size(); ++k)
        lambda_full[working_set[k]] = lambda_W[k];
}

// In-place initial_working_set: writes into pre-allocated output vector.
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
void initial_working_set(
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
    int n_eq,
    Scalar tolerance,
    std::vector<int>& W)
{
    W.clear();
    const int m = static_cast<int>(A_full.rows());

    for(int i = 0; i < n_eq; ++i)
        W.push_back(i);

    for(int i = n_eq; i < m; ++i)
    {
        Scalar residual = A_full.row(i).dot(x0) - b_full[i];
        if(std::abs(residual) < tolerance)
            W.push_back(i);
    }
}

// Stateful dense active-set QP solver with pre-allocated workspace.
//
// Pre-allocates factorization objects and workspace matrices at construction,
// then recomputes via compute() on each solve. Eliminates per-call dynamic
// allocation for QR, LDLT, working-set matrix extraction, and intermediate
// vectors.
//
// Template parameters:
//   N — problem dimension (compile-time or dynamic_dimension)
//   M — constraint count excluding box bounds (compile-time or dynamic_dimension)
//
// When both N and M are compile-time, max_total_constraints = M + 2*N
// (M constraints + N lower + N upper bounds) and all workspace uses
// stack-allocated max-bounded Eigen types.
//
// Reference: N&W Algorithm 16.1, pp. 460-463 (active-set method for convex QP)
//            N&W Section 16.2, eq. 16.16-16.19 (null-space method)
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
class active_set_qp_solver
{
    static constexpr int max_total_constraints =
        (M == argmin::dynamic_dimension || N == argmin::dynamic_dimension)
            ? Eigen::Dynamic : M + 2 * N;

    // Max-bounded matrix type: rows up to MaxR, cols up to MaxC.
    template <int Rows, int Cols, int MaxR, int MaxC>
    using bounded_matrix = Eigen::Matrix<Scalar, Rows, Cols, 0, MaxR, MaxC>;

    // Max-bounded vector type: size up to MaxN.
    template <int Size, int MaxN>
    using bounded_vector = Eigen::Matrix<Scalar, Size, 1, 0, MaxN, 1>;

    // Constraint-row matrix: Dynamic rows up to max_total_constraints, N cols.
    using constraint_matrix = bounded_matrix<Eigen::Dynamic, N, max_total_constraints, N>;

    // Constraint-length vector: Dynamic size up to max_total_constraints.
    using constraint_vector = bounded_vector<Eigen::Dynamic, max_total_constraints>;

    // N-dimension square matrix (max-bounded).
    using n_square_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, N>;

    // N-dimension vector (max-bounded).
    using n_vector = bounded_vector<Eigen::Dynamic, N>;

    // QR input for A_W^T: always N rows, variable cols. Fixed-row when N is compile-time.
    using qr_matrix = bounded_matrix<N, Eigen::Dynamic, N, max_total_constraints>;

    // QR input for multiplier solves: variable rows (up to N), variable cols.
    // Must be dynamic-row because (A_W * Y)^T has rank rows where rank <= N.
    using qr_lambda_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, max_total_constraints>;

    // Q matrix: N x N (max-bounded).
    using q_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, N>;

    // Periodic full refactorization interval to prevent accumulated
    // Givens rotation floating-point drift.
    static constexpr uint16_t refactorization_interval{20};

public:
    explicit active_set_qp_solver(int n, int max_constraints)
        : n_(n)
        , max_m_(max_constraints)
        , A_full_(max_constraints, n)
        , b_full_(max_constraints)
        , A_W_(max_constraints, n)
        , qr_lambda_(n, n)
        , llt_reduced_(n)
        , Q_explicit_(n, n)
        , R_(n, max_constraints)
        , ZtGZ_(n, n)
        , rhs_(n)
        , p_z_(n)
        , rhs_lam_(n)
        , g_k_(n)
        , lambda_full_(max_constraints)
        , A_aug_(max_constraints, n)
        , b_aug_(max_constraints)
    {
        W_.reserve(max_constraints);
    }

    active_set_qp_solver() = default;

    // Solve QP subproblem using pre-allocated workspace.
    //
    // Reference: N&W Algorithm 16.1, pp. 460-463.
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    qp_result<Scalar, N, M> solve(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts = {})
    {
        const int n = G.rows();
        const int m_eq = A_eq.rows();
        const int m_ineq = A_ineq.rows();
        const int m_total = m_eq + m_ineq;
        const int max_iter = static_cast<int>(opts.max_iterations.value_or(200));
        const auto tol = static_cast<Scalar>(opts.tolerance.value_or(1e-12));

        if(m_eq > 0)
        {
            A_full_.topRows(m_eq) = A_eq;
            b_full_.head(m_eq) = b_eq;
        }
        if(m_ineq > 0)
        {
            A_full_.middleRows(m_eq, m_ineq) = A_ineq;
            b_full_.segment(m_eq, m_ineq) = b_ineq;
        }

        auto A_view = A_full_.topRows(m_total);
        auto b_view = b_full_.head(m_total);

        initial_working_set(
            x0, constraint_matrix(A_view),
            constraint_vector(b_view), m_eq, tol, W_);

        Eigen::Vector<Scalar, N> x = x0;
        qr_valid_ = false;
        givens_update_count_ = 0;

        for(int iter = 0; iter < max_iter; ++iter)
        {
            g_k_.head(n).noalias() = G * x + d;

            const int wsize = static_cast<int>(W_.size());
            for(int k = 0; k < wsize; ++k)
                A_W_.row(k) = A_full_.row(W_[k]);

            auto [p, lambda_W] = solve_equality_subproblem(
                G, g_k_.head(n), A_W_.topRows(wsize), n, wsize);

            if(p.norm() < tol)
            {
                int drop = most_negative_multiplier(lambda_W, W_, m_eq, tol);
                if(drop == -1)
                {
                    build_full_lambda(lambda_W, W_, m_total, lambda_full_);
                    qp_result<Scalar, N> res;
                    res.x = x;
                    res.lambda = lambda_full_.head(m_total);
                    res.status = qp_status::optimal;
                    res.iterations = iter;
                    return res;
                }
                // Remove constraint: Givens QR downdate
                if(qr_valid_ && wsize > 1)
                {
                    remove_constraint_qr<Scalar>(
                        Q_explicit_, R_, drop, wsize);
                    ++givens_update_count_;
                }
                else
                {
                    qr_valid_ = false;
                }
                W_.erase(W_.begin() + drop);
            }
            else
            {
                auto [alpha, blocking_idx] = blocking_step_length(
                    x, Eigen::Vector<Scalar, N>(p),
                    constraint_matrix(A_view),
                    constraint_vector(b_view), W_, m_eq);
                x += alpha * p;

                if(alpha < Scalar(1) && blocking_idx >= 0)
                {
                    W_.push_back(blocking_idx);
                    // Add constraint: Givens QR update
                    const int new_wsize = static_cast<int>(W_.size());
                    if(qr_valid_ && new_wsize <= n)
                    {
                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> a_new =
                            A_full_.row(blocking_idx).head(n).transpose();
                        add_constraint_qr<Scalar>(
                            Q_explicit_, R_, a_new, new_wsize);
                        ++givens_update_count_;
                    }
                    else
                    {
                        qr_valid_ = false;
                    }
                }
            }
        }

        constraint_vector zero_lam = constraint_vector::Zero(
            static_cast<int>(W_.size()));
        build_full_lambda(zero_lam, W_, m_total, lambda_full_);
        qp_result<Scalar, N, M> res;
        res.x = x;
        res.lambda = lambda_full_.head(m_total);
        res.status = qp_status::max_iterations;
        res.iterations = max_iter;
        return res;
    }

    // Box-constraint overload.
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    qp_result<Scalar, N, M> solve(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts = {})
    {
        const int n = G.rows();
        const int m_ineq = A_ineq.rows();
        const int m_box = 2 * n;
        const int m_aug = m_ineq + m_box;

        if(m_ineq > 0)
        {
            A_aug_.topRows(m_ineq) = A_ineq;
            b_aug_.head(m_ineq) = b_ineq;
        }

        A_aug_.middleRows(m_ineq, n) = Eigen::Matrix<Scalar, N, N>::Identity(n, n);
        b_aug_.segment(m_ineq, n) = lower;

        A_aug_.middleRows(m_ineq + n, n) = -Eigen::Matrix<Scalar, N, N>::Identity(n, n);
        b_aug_.segment(m_ineq + n, n) = -upper;

        Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_eq_dyn(A_eq.rows(), n);
        A_eq_dyn = A_eq;
        Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_aug_dyn(m_aug, n);
        A_aug_dyn = A_aug_.topRows(m_aug);
        Eigen::VectorX<Scalar> b_eq_dyn = b_eq;
        Eigen::VectorX<Scalar> b_aug_dyn = b_aug_.head(m_aug);

        return solve(G, d, A_eq_dyn, b_eq_dyn, A_aug_dyn, b_aug_dyn, x0, opts);
    }

private:
    // Full QR factorization of A_W^T using Householder, then extract
    // explicit Q and R matrices. Called on first iteration and
    // periodically to prevent accumulated Givens drift.
    void full_qr_factorize(
        const Eigen::Ref<const constraint_matrix>& A_W,
        int n,
        int m)
    {
        Eigen::HouseholderQR<qr_matrix> qr(A_W.transpose());
        Q_explicit_.setIdentity(n, n);
        Q_explicit_.applyOnTheLeft(qr.householderQ());
        R_.topLeftCorner(n, m).setZero();
        auto R_upper = qr.matrixQR();
        for(int j = 0; j < m; ++j)
            for(int i = 0; i <= j && i < n; ++i)
                R_(i, j) = R_upper(i, j);
        qr_valid_ = true;
        givens_update_count_ = 0;
    }

    // Solve equality-constrained QP subproblem via null-space method.
    //
    // Uses explicit Q/R factorization with Givens-based incremental updates
    // when the working set changes by one constraint. Falls back to full
    // HouseholderQR on the first call and periodically to prevent drift.
    //
    // Reference: N&W Section 16.2, eq. 16.16-16.19.
    //            N&W Algorithm 16.3 (QR update via Givens rotations).
    std::pair<n_vector, constraint_vector>
    solve_equality_subproblem(
        const Eigen::Ref<const n_square_matrix>& G,
        const Eigen::Ref<const n_vector>& g_k,
        const Eigen::Ref<const constraint_matrix>& A_W,
        int n,
        int m)
    {
        if(m == 0)
        {
            llt_reduced_.compute(G);
            n_vector p = llt_reduced_.solve(-g_k);
            qr_valid_ = false;
            return {p, constraint_vector{}};
        }

        if(m >= n)
        {
            n_vector p = n_vector::Zero(n);
            qr_lambda_.compute(A_W.transpose());
            constraint_vector lambda = qr_lambda_.solve(g_k);
            return {p, lambda};
        }

        // Full refactorization if not valid or periodic drift prevention
        if(!qr_valid_ || givens_update_count_ >= refactorization_interval)
            full_qr_factorize(A_W, n, m);

        const int rank = m;
        auto Y_block = Q_explicit_.leftCols(rank);
        auto Z_block = Q_explicit_.rightCols(n - rank);

        ZtGZ_.topLeftCorner(n - rank, n - rank).noalias() =
            Z_block.transpose() * G * Z_block;
        llt_reduced_.compute(ZtGZ_.topLeftCorner(n - rank, n - rank));

        rhs_.head(n - rank).noalias() = -(Z_block.transpose() * g_k);
        p_z_.head(n - rank) = llt_reduced_.solve(rhs_.head(n - rank));
        n_vector p = Z_block * p_z_.head(n - rank);

        auto AY = (A_W * Y_block).eval();
        rhs_lam_.head(rank).noalias() = Y_block.transpose() * (g_k + G * p);
        qr_lambda_.compute(AY.transpose());
        constraint_vector lambda = qr_lambda_.solve(rhs_lam_.head(rank));

        return {p, lambda};
    }

    int n_{0};
    int max_m_{0};

    constraint_matrix A_full_;
    constraint_vector b_full_;
    constraint_matrix A_W_;

    Eigen::HouseholderQR<qr_lambda_matrix> qr_lambda_;
    Eigen::LLT<n_square_matrix> llt_reduced_;
    q_matrix Q_explicit_;
    qr_matrix R_;
    bool qr_valid_{false};
    uint16_t givens_update_count_{0};
    n_square_matrix ZtGZ_;
    n_vector rhs_;
    n_vector p_z_;
    n_vector rhs_lam_;
    n_vector g_k_;

    std::vector<int> W_;
    constraint_vector lambda_full_;

    constraint_matrix A_aug_;
    constraint_vector b_aug_;
};

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
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
qp_result<Scalar, N> solve_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& d,
    const Eigen::Matrix<Scalar, Meq, N>& A_eq,
    const Eigen::Vector<Scalar, Meq>& b_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
    const Eigen::Vector<Scalar, Mineq>& b_ineq,
    const Eigen::Vector<Scalar, N>& x0,
    const argmin::qp_options& opts = {})
{
    const int n = G.rows();
    const int m_eq = A_eq.rows();
    const int m_ineq = A_ineq.rows();
    const int m_total = m_eq + m_ineq;
    const int max_iter = static_cast<int>(opts.max_iterations.value_or(200));
    const auto tol = static_cast<Scalar>(opts.tolerance.value_or(1e-12));

    // Assemble full constraint matrix: [A_eq; A_ineq] x >= [b_eq; b_ineq]
    // (Equality constraints treated as a_i^T x = b_i, always active.)
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_full(m_total, n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_full(m_total);
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

    auto W = initial_working_set<Scalar, N>(x0, A_full, b_full, m_eq, tol);
    Eigen::Vector<Scalar, N> x = x0;

    for(int iter = 0; iter < max_iter; ++iter)
    {
        Eigen::Vector<Scalar, N> g_k = (G * x + d).eval();
        auto A_W = extract_working_rows(A_full, W);

        auto [p, lambda_W] = solve_equality_qp<Scalar, N>(G, g_k, A_W);

        if(p.norm() < tol)
        {
            // At stationary point for current working set -- check multipliers
            int drop = most_negative_multiplier(lambda_W, W, m_eq, tol);
            if(drop == -1)
            {
                // All inequality multipliers non-negative: optimal
                auto lambda_full = build_full_lambda(lambda_W, W, m_total);
                qp_result<Scalar, N> res;
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
            auto [alpha, blocking_idx] = blocking_step_length<Scalar, N>(
                x, p, A_full, b_full, W, m_eq);
            x += alpha * p;

            if(alpha < Scalar(1) && blocking_idx >= 0)
                W.push_back(blocking_idx);
        }
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> zero_lam =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(static_cast<int>(W.size()));
    auto lambda_full = build_full_lambda(zero_lam, W, m_total);
    qp_result<Scalar, N> res;
    res.x = x;
    res.lambda = lambda_full;
    res.status = qp_status::max_iterations;
    res.iterations = max_iter;
    return res;
}

// Convenience overload with box constraints.
//
// Converts box bounds l <= x <= u to inequality constraints:
//   I*x >= l    (lower bounds)
//  -I*x >= -u   (upper bounds)
// and appends them to A_ineq/b_ineq.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
qp_result<Scalar, N> solve_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& d,
    const Eigen::Matrix<Scalar, Meq, N>& A_eq,
    const Eigen::Vector<Scalar, Meq>& b_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
    const Eigen::Vector<Scalar, Mineq>& b_ineq,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const Eigen::Vector<Scalar, N>& x0,
    const argmin::qp_options& opts = {})
{
    const int n = G.rows();
    const int m_ineq = A_ineq.rows();
    const int m_box = 2 * n;

    // Build augmented inequality system: [A_ineq; I; -I] x >= [b_ineq; l; -u]
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_aug(m_ineq + m_box, n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_aug(m_ineq + m_box);

    if(m_ineq > 0)
    {
        A_aug.topRows(m_ineq) = A_ineq;
        b_aug.head(m_ineq) = b_ineq;
    }

    A_aug.middleRows(m_ineq, n) = Eigen::Matrix<Scalar, N, N>::Identity(n, n);
    b_aug.segment(m_ineq, n) = lower;

    A_aug.bottomRows(n) = -Eigen::Matrix<Scalar, N, N>::Identity(n, n);
    b_aug.tail(n) = -upper;

    return solve_qp(G, d, A_eq, b_eq, A_aug, b_aug, x0, opts);
}

}

#endif
