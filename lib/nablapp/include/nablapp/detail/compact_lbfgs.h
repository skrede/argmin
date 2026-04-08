#ifndef HPP_GUARD_NABLAPP_DETAIL_COMPACT_LBFGS_H
#define HPP_GUARD_NABLAPP_DETAIL_COMPACT_LBFGS_H

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

// Compact L-BFGS representation per N&W Section 9.2, eq. 9.15.
//
// Represents B_k = theta*I - W*M^{-1}*W^T where W = [theta*S, Y] and M is
// the 2m x 2m middle matrix. Also provides two_loop_recursion(g) computing
// H_k*g (N&W Algorithm 9.1) using the same (s,y) storage.
//
// Provides both H*g (two-loop recursion) and B*v products plus reduced
// Hessians needed by L-BFGS-B subspace minimization.
//
// MaxHistory is the compile-time history buffer capacity (number of (s,y)
// pairs). All history-dimension storage is fixed-size when MaxHistory is
// known at compile time, eliminating per-iteration heap allocation.
//
// STS_, L_, D_ are maintained incrementally in push() at O(n*k) per push
// instead of O(n*k^2) full recompute. STS_ stores the unscaled S^T*S;
// theta_ scaling is applied only in recompute_middle().
//
// Reference: N&W Section 9.2 (compact representation, eq. 9.15),
//            N&W Algorithm 9.1 (two-loop recursion),
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).
//            Incremental STS/L/D update: standard rank-1 dot-product
//            maintenance (see N&W eq. 7.17 for analogous BFGS increments).

template <typename Scalar = double, int N = nablapp::dynamic_dimension, int MaxHistory = 7>
class compact_lbfgs
{
    static_assert(MaxHistory > 0, "MaxHistory must be positive");

public:
    compact_lbfgs() = default;

    // Add a new (s, y) pair with damped curvature update.
    // Instead of silently rejecting pairs with small s^T y, clamps to
    // max(s^T y, eps * ||s||^2) to preserve curvature information from
    // near-orthogonal pairs. Only truly degenerate (zero-step) pairs
    // are rejected.
    // Reference: N&W Section 9.1 (curvature condition), Powell (damped BFGS).
    void push(const Eigen::Vector<Scalar, N>& s, const Eigen::Vector<Scalar, N>& y)
    {
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        Scalar sTs = s.squaredNorm();

        // Reject truly degenerate pairs (zero step)
        if(sTs < eps * eps) return;

        // Reject negative curvature (sTy <= 0). Negative curvature from
        // Lagrangian evaluations in SQP solvers poisons the BFGS approximation.
        // Damped clamping only applies to small-but-positive curvature.
        Scalar sTy = s.dot(y);
        if(sTy <= Scalar(0)) return;

        // Damped: clamp small positive s^T y upward to preserve curvature
        // from near-orthogonal pairs (common at bound constraints).
        Scalar sTy_effective = std::max(sTy, eps * sTs);

        const int n = s.size();

        // Lazy row-dimension initialization on first push (when N is dynamic)
        if(count_ == 0 && S_.rows() == 0)
        {
            S_.setZero(n, MaxHistory);
            Y_.setZero(n, MaxHistory);
        }

        if(count_ < MaxHistory)
        {
            S_.col(count_) = s;
            Y_.col(count_) = y;
            rho_[count_] = Scalar(1) / sTy_effective;
            ++count_;

            // Incremental STS_, L_, D_ for the new column.
            // Cost: O(n*k) dot products for the new column.
            update_new_column();
        }
        else
        {
            for(int j = 0; j < MaxHistory - 1; ++j)
            {
                S_.col(j) = S_.col(j + 1);
                Y_.col(j) = Y_.col(j + 1);
                rho_[j] = rho_[j + 1];
            }
            S_.col(MaxHistory - 1) = s;
            Y_.col(MaxHistory - 1) = y;
            rho_[MaxHistory - 1] = Scalar(1) / sTy_effective;

            // Shift STS_, L_, D_ to match column eviction, then compute new
            // column entries. Cost: O(k^2) element shifts + O(n*k) dots.
            // The .eval() calls are mandatory: without them Eigen lazy
            // evaluation corrupts overlapping source/destination regions.
            shift_evict_and_update();
        }

        // Initial Hessian scaling: theta = y^T y / s^T y (N&W eq. 9.6, inverse of gamma_0).
        // gamma_0 = s^T y / y^T y scales H_0 to approximate curvature magnitude.
        theta_ = y.squaredNorm() / sTy_effective;
        theta_ = std::clamp(theta_, Scalar(1e-10), Scalar(1e10));

        recompute_middle();
    }

    // Compute B*v = theta*v - W * M^{-1} * W^T * v.
    // W = [theta*S, Y], M = [[theta*S^T*S, L], [L^T, -D]].
    // Reference: N&W Section 9.2, eq. 9.15.
    Eigen::Vector<Scalar, N> multiply(const Eigen::Vector<Scalar, N>& v) const
    {
        if(count_ == 0) return (theta_ * v).eval();

        const int k = count_;
        auto S = S_.leftCols(k);
        auto Y = Y_.leftCols(k);

        // W^T * v (2k x 1). noalias() safe: Wv segments are distinct from
        // all source operands (S, Y, v).
        Eigen::Vector<Scalar, 2 * MaxHistory> Wv;
        Wv.head(k).noalias() = theta_ * (S.transpose() * v);
        Wv.segment(MaxHistory, k).noalias() = Y.transpose() * v;

        // Solve M * z = W^T * v using pre-allocated middle matrix
        auto M = middle_.template topLeftCorner<2 * MaxHistory, 2 * MaxHistory>();

        // Build the active submatrix for the LDLT solve
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                       0, 2 * MaxHistory, 2 * MaxHistory> M_active(2 * k, 2 * k);
        M_active.topLeftCorner(k, k) = middle_.topLeftCorner(k, k);
        M_active.block(0, k, k, k) = middle_.block(0, MaxHistory, k, k);
        M_active.block(k, 0, k, k) = middle_.block(MaxHistory, 0, k, k);
        M_active.block(k, k, k, k) = middle_.block(MaxHistory, MaxHistory, k, k);

        Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                  0, 2 * MaxHistory, 2 * MaxHistory>> ldlt(M_active);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Wv_active(2 * k);
        Wv_active.head(k) = Wv.head(k);
        Wv_active.tail(k) = Wv.segment(MaxHistory, k);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> z = ldlt.solve(Wv_active);

        // B*v = theta*v - W*z. noalias() safe: Wz is a fresh variable,
        // distinct from S, Y, z.
        Eigen::Vector<Scalar, N> Wz(v.size());
        Wz.noalias() = theta_ * (S * z.head(k));
        Wz.noalias() += Y * z.tail(k);
        return (theta_ * v - Wz).eval();
    }

    // Build reduced Hessian Z^T * B * Z for free variable subspace.
    // Z selects rows in free_indices from the identity matrix.
    // B_FF = theta * I_F - W_F * M^{-1} * W_F^T.
    //
    // LDLT stays dynamic because free-variable count varies per call.
    // Reference: N&W Section 16.6 (subspace minimization).
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced_hessian(
        const std::vector<int>& free_indices) const
    {
        using dyn_matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        const int nf = static_cast<int>(free_indices.size());
        if(nf == 0) return dyn_matrix{};

        if(count_ == 0)
            return (theta_ * dyn_matrix::Identity(nf, nf)).eval();

        const int k = count_;

        dyn_matrix WF(nf, 2 * k);
        for(int j = 0; j < nf; ++j)
        {
            int idx = free_indices[j];
            WF.row(j).head(k) = theta_ * S_.row(idx).head(k);
            WF.row(j).tail(k) = Y_.row(idx).head(k);
        }

        // Build active M submatrix
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                       0, 2 * MaxHistory, 2 * MaxHistory> M_active(2 * k, 2 * k);
        M_active.topLeftCorner(k, k) = middle_.topLeftCorner(k, k);
        M_active.block(0, k, k, k) = middle_.block(0, MaxHistory, k, k);
        M_active.block(k, 0, k, k) = middle_.block(MaxHistory, 0, k, k);
        M_active.block(k, k, k, k) = middle_.block(MaxHistory, MaxHistory, k, k);

        Eigen::LDLT<dyn_matrix> ldlt(M_active);
        dyn_matrix MiWFt = ldlt.solve(WF.transpose());

        dyn_matrix BFF = theta_ * dyn_matrix::Identity(nf, nf)
                       - WF * MiWFt;
        return BFF;
    }

    // Build full dense Hessian B = theta*I - W * M^{-1} * W^T.
    // Single LDLT factorization of 2k x 2k active submatrix instead of n
    // separate factorizations from n multiply() calls. Returns by value
    // (NRVO eliminates copy).
    // Reference: N&W Section 9.2, eq. 9.15 (compact representation).
    Eigen::Matrix<Scalar, N, N> dense_hessian(int n) const
    {
        Eigen::Matrix<Scalar, N, N> B;
        if constexpr(N != Eigen::Dynamic)
            B.setIdentity();
        else
            B.setIdentity(n, n);
        B *= theta_;

        if(count_ == 0) return B;

        const int k = count_;
        auto S = S_.leftCols(k);
        auto Y = Y_.leftCols(k);

        // W = [theta*S, Y], size n x 2k
        Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0, N, 2 * MaxHistory> W;
        if constexpr(N != Eigen::Dynamic)
            W.resize(N, 2 * k);
        else
            W.resize(n, 2 * k);
        W.leftCols(k).noalias() = theta_ * S;
        W.rightCols(k) = Y;

        // Extract active 2k x 2k submatrix from middle_ (Pitfall 2: blocks
        // are at MaxHistory offsets, not contiguous 2k x 2k)
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                       0, 2 * MaxHistory, 2 * MaxHistory> M_active(2 * k, 2 * k);
        M_active.topLeftCorner(k, k) = middle_.topLeftCorner(k, k);
        M_active.block(0, k, k, k) = middle_.block(0, MaxHistory, k, k);
        M_active.block(k, 0, k, k) = middle_.block(MaxHistory, 0, k, k);
        M_active.block(k, k, k, k) = middle_.block(MaxHistory, MaxHistory, k, k);

        // Solve M * Z = W^T => Z = M^{-1} * W^T, size 2k x n
        Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                  0, 2 * MaxHistory, 2 * MaxHistory>> ldlt(M_active);
        auto Z = ldlt.solve(W.transpose()).eval();

        // B = theta*I - W * Z
        B.noalias() -= W * Z;

        return B;
    }

    // Two-loop recursion computing H_k * g (N&W Algorithm 9.1).
    // Returns H*g without negation; caller negates for search direction.
    Eigen::Vector<Scalar, N> two_loop_recursion(const Eigen::Vector<Scalar, N>& g) const
    {
        if(count_ == 0) return g;

        const int k = count_;
        Eigen::Vector<Scalar, N> q = g;

        for(int j = k - 1; j >= 0; --j)
        {
            alpha_[j] = rho_[j] * S_.col(j).dot(q);
            q -= alpha_[j] * Y_.col(j);
        }

        Scalar gamma = Scalar(1) / theta_;
        Eigen::Vector<Scalar, N> r = (gamma * q).eval();

        for(int j = 0; j < k; ++j)
        {
            Scalar beta = rho_[j] * Y_.col(j).dot(r);
            r += S_.col(j) * (alpha_[j] - beta);
        }

        return r;
    }

    Scalar theta() const { return theta_; }

    void reset()
    {
        count_ = 0;
        theta_ = Scalar(1);
    }

    int size() const { return count_; }
    static constexpr int capacity() { return MaxHistory; }

private:
    // Incrementally update STS_, L_, D_ for the newly appended column.
    // Called after count_ has been incremented and S_/Y_ columns written.
    // Cost: O(n*k) dot products for the new column.
    void update_new_column()
    {
        const int k = count_;
        const int col = k - 1;

        for(int j = 0; j < k; ++j)
        {
            Scalar dot = S_.col(j).dot(S_.col(col));
            STS_(j, col) = dot;
            STS_(col, j) = dot;
        }

        for(int j = 0; j < col; ++j)
            L_(col, j) = S_.col(col).dot(Y_.col(j));
        L_(col, col) = Scalar(0);

        D_[col] = S_.col(col).dot(Y_.col(col));
    }

    // Shift STS_, L_, D_ after column eviction (row/col 0 removed, remainder
    // shifted up-left), then compute new entries for the last column.
    // Cost: O(k^2) element shifts + O(n*k) dot products.
    void shift_evict_and_update()
    {
        const int k = MaxHistory;

        // Shift cached matrices: remove row/col 0, move remainder up-left.
        // .eval() is mandatory — Eigen lazy evaluation aliases overlapping
        // source/destination regions in the same matrix.
        STS_.template topLeftCorner<MaxHistory - 1, MaxHistory - 1>() =
            STS_.template bottomRightCorner<MaxHistory - 1, MaxHistory - 1>().eval();
        L_.template topLeftCorner<MaxHistory - 1, MaxHistory - 1>() =
            L_.template bottomRightCorner<MaxHistory - 1, MaxHistory - 1>().eval();
        D_.template head<MaxHistory - 1>() = D_.template tail<MaxHistory - 1>().eval();

        // Compute new column/row entries for position k-1
        for(int j = 0; j < k; ++j)
        {
            Scalar dot = S_.col(j).dot(S_.col(k - 1));
            STS_(j, k - 1) = dot;
            STS_(k - 1, j) = dot;
        }

        for(int j = 0; j < k - 1; ++j)
            L_(k - 1, j) = S_.col(k - 1).dot(Y_.col(j));
        L_(k - 1, k - 1) = Scalar(0);

        D_[k - 1] = S_.col(k - 1).dot(Y_.col(k - 1));
    }

    // Middle matrix M stored in block layout:
    //   [0..MaxHistory-1, 0..MaxHistory-1]        = theta * S^T * S
    //   [0..MaxHistory-1, MaxHistory..2*MaxHistory] = L
    //   [MaxHistory.., 0..MaxHistory-1]            = L^T
    //   [MaxHistory.., MaxHistory..]               = -D
    // Assemble middle matrix M from incrementally maintained STS_, L_, D_.
    // Cost: O(k^2) block copies instead of O(n*k^2) S^T*S recomputation.
    void recompute_middle()
    {
        const int k = count_;

        middle_.template topLeftCorner<MaxHistory, MaxHistory>()
            .topLeftCorner(k, k) = theta_ * STS_.topLeftCorner(k, k);
        middle_.template block<MaxHistory, MaxHistory>(0, MaxHistory)
            .topLeftCorner(k, k) = L_.topLeftCorner(k, k);
        middle_.template block<MaxHistory, MaxHistory>(MaxHistory, 0)
            .topLeftCorner(k, k) = L_.topLeftCorner(k, k).transpose();
        middle_.template block<MaxHistory, MaxHistory>(MaxHistory, MaxHistory)
            .topLeftCorner(k, k) = (-D_.head(k)).asDiagonal();
    }

    Eigen::Matrix<Scalar, N, MaxHistory> S_{};
    Eigen::Matrix<Scalar, N, MaxHistory> Y_{};
    Eigen::Matrix<Scalar, MaxHistory, MaxHistory> STS_{};  // cached unscaled S^T*S
    Eigen::Matrix<Scalar, MaxHistory, MaxHistory> L_{};
    Eigen::Vector<Scalar, MaxHistory> D_{};
    Eigen::Matrix<Scalar, 2 * MaxHistory, 2 * MaxHistory> middle_{};
    Eigen::Vector<Scalar, MaxHistory> rho_{};
    mutable Eigen::Vector<Scalar, MaxHistory> alpha_{};
    Scalar theta_{Scalar(1)};
    int count_{0};
};

}

#endif
