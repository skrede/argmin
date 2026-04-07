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
// Reference: N&W Section 9.2 (compact representation, eq. 9.15),
//            N&W Algorithm 9.1 (two-loop recursion),
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

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
        }

        // Initial Hessian scaling: theta = y^T y / s^T y (N&W eq. 9.6, inverse of gamma_0).
        // gamma_0 = s^T y / y^T y scales H_0 to approximate curvature magnitude.
        theta_ = y.squaredNorm() / sTy_effective;
        theta_ = std::clamp(theta_, Scalar(1e-10), Scalar(1e10));

        recompute_LD();
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

        // W^T * v (2k x 1)
        Eigen::Vector<Scalar, 2 * MaxHistory> Wv;
        Wv.head(k) = theta_ * (S.transpose() * v);
        Wv.segment(MaxHistory, k) = Y.transpose() * v;

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

        // B*v = theta*v - W*z
        Eigen::Vector<Scalar, N> Wz = (theta_ * (S * z.head(k)) + Y * z.tail(k)).eval();
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
    void recompute_LD()
    {
        const int k = count_;
        for(int i = 0; i < k; ++i)
        {
            D_[i] = S_.col(i).dot(Y_.col(i));
            for(int j = 0; j < i; ++j)
                L_(i, j) = S_.col(i).dot(Y_.col(j));
            for(int j = i; j < k; ++j)
                L_(i, j) = Scalar(0);
        }
    }

    // Middle matrix M stored in block layout:
    //   [0..MaxHistory-1, 0..MaxHistory-1]        = theta * S^T * S
    //   [0..MaxHistory-1, MaxHistory..2*MaxHistory] = L
    //   [MaxHistory.., 0..MaxHistory-1]            = L^T
    //   [MaxHistory.., MaxHistory..]               = -D
    void recompute_middle()
    {
        const int k = count_;
        auto S = S_.leftCols(k);

        middle_.template topLeftCorner<MaxHistory, MaxHistory>()
            .topLeftCorner(k, k).noalias() = theta_ * (S.transpose() * S);
        middle_.template block<MaxHistory, MaxHistory>(0, MaxHistory)
            .topLeftCorner(k, k) = L_.topLeftCorner(k, k);
        middle_.template block<MaxHistory, MaxHistory>(MaxHistory, 0)
            .topLeftCorner(k, k) = L_.topLeftCorner(k, k).transpose();
        middle_.template block<MaxHistory, MaxHistory>(MaxHistory, MaxHistory)
            .topLeftCorner(k, k) = (-D_.head(k)).asDiagonal();
    }

    Eigen::Matrix<Scalar, N, MaxHistory> S_{};
    Eigen::Matrix<Scalar, N, MaxHistory> Y_{};
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
