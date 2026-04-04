#ifndef HPP_GUARD_NABLAPP_DETAIL_COMPACT_LBFGS_H
#define HPP_GUARD_NABLAPP_DETAIL_COMPACT_LBFGS_H

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <vector>

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
// Reference: N&W Section 9.2 (compact representation, eq. 9.15),
//            N&W Algorithm 9.1 (two-loop recursion),
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class compact_lbfgs
{
public:
    explicit compact_lbfgs(int m = 10) : capacity_(m)
    {
        L_.setZero(capacity_, capacity_);
        D_.setZero(capacity_);
        middle_.setZero(2 * capacity_, 2 * capacity_);
        alpha_.resize(capacity_);
    }

    // Add a new (s, y) pair. Rejects pairs with s^T y <= 0 (curvature guard).
    void push(const Eigen::Vector<Scalar, N>& s, const Eigen::Vector<Scalar, N>& y)
    {
        Scalar sTy = s.dot(y);
        if(sTy <= Scalar(0)) return;

        const int n = s.size();

        // Lazy allocation on first push
        if(count_ == 0 && S_.cols() == 0)
        {
            S_.setZero(n, capacity_);
            Y_.setZero(n, capacity_);
            rho_.resize(capacity_);
        }

        if(count_ < capacity_)
        {
            S_.col(count_) = s;
            Y_.col(count_) = y;
            rho_[count_] = Scalar(1) / sTy;
            ++count_;
        }
        else
        {
            // Shift columns left by 1, overwrite last
            for(int j = 0; j < capacity_ - 1; ++j)
            {
                S_.col(j) = S_.col(j + 1);
                Y_.col(j) = Y_.col(j + 1);
                rho_[j] = rho_[j + 1];
            }
            S_.col(capacity_ - 1) = s;
            Y_.col(capacity_ - 1) = y;
            rho_[capacity_ - 1] = Scalar(1) / sTy;
        }

        // Update theta = y^T y / s^T y (N&W eq. 9.6, inverse of gamma)
        theta_ = y.squaredNorm() / sTy;
        theta_ = std::clamp(theta_, Scalar(1e-10), Scalar(1e10));

        // Recompute L, D, and middle matrix from active columns
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
        Eigen::VectorX<Scalar> Wv(2 * k);
        Wv.head(k) = theta_ * (S.transpose() * v);
        Wv.tail(k) = Y.transpose() * v;

        // Solve M * z = W^T * v using pre-allocated middle matrix
        auto M = middle_.topLeftCorner(2 * k, 2 * k);
        Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(M);
        Eigen::VectorX<Scalar> z = ldlt.solve(Wv);

        // B*v = theta*v - W*z
        Eigen::Vector<Scalar, N> Wz = (theta_ * (S * z.head(k)) + Y * z.tail(k)).eval();
        return (theta_ * v - Wz).eval();
    }

    // Build reduced Hessian Z^T * B * Z for free variable subspace.
    // Z selects rows in free_indices from the identity matrix.
    // B_FF = theta * I_F - W_F * M^{-1} * W_F^T.
    // Reference: N&W Section 16.6 (subspace minimization).
    Eigen::MatrixX<Scalar> reduced_hessian(const std::vector<int>& free_indices) const
    {
        const int nf = static_cast<int>(free_indices.size());
        if(nf == 0) return Eigen::MatrixX<Scalar>{};

        if(count_ == 0)
            return (theta_ * Eigen::MatrixX<Scalar>::Identity(nf, nf)).eval();

        const int k = count_;

        // Extract W_F (nf x 2k): rows of W restricted to free_indices
        Eigen::MatrixX<Scalar> WF(nf, 2 * k);
        for(int j = 0; j < nf; ++j)
        {
            int idx = free_indices[j];
            WF.row(j).head(k) = theta_ * S_.row(idx).head(k);
            WF.row(j).tail(k) = Y_.row(idx).head(k);
        }

        // M^{-1} * W_F^T using pre-allocated middle matrix
        auto M = middle_.topLeftCorner(2 * k, 2 * k);
        Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(M);
        Eigen::MatrixX<Scalar> MiWFt = ldlt.solve(WF.transpose());

        // B_FF = theta * I_F - W_F * M^{-1} * W_F^T
        Eigen::MatrixX<Scalar> BFF = theta_ * Eigen::MatrixX<Scalar>::Identity(nf, nf)
                                   - WF * MiWFt;
        return BFF;
    }

    // Two-loop recursion computing H_k * g (N&W Algorithm 9.1).
    // Returns H*g without negation; caller negates for search direction.
    // Uses the same (s,y) storage as the compact form.
    Eigen::Vector<Scalar, N> two_loop_recursion(const Eigen::Vector<Scalar, N>& g) const
    {
        if(count_ == 0) return g;

        const int k = count_;
        Eigen::Vector<Scalar, N> q = g;

        // Backward loop (N&W Algorithm 9.1, step 1)
        for(int j = k - 1; j >= 0; --j)
        {
            alpha_[j] = rho_[j] * S_.col(j).dot(q);
            q -= alpha_[j] * Y_.col(j);
        }

        // Initial Hessian scaling: gamma = s^T y / y^T y = 1/theta
        Scalar gamma = Scalar(1) / theta_;
        Eigen::Vector<Scalar, N> r = (gamma * q).eval();

        // Forward loop (N&W Algorithm 9.1, step 3)
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
        // Keep allocated storage, just reset count
    }

    int size() const { return count_; }
    int capacity() const { return capacity_; }

private:
    // Recompute L_ (strictly lower triangular) and D_ (diagonal) from active columns.
    // L_{i,j} = s_i^T y_j for i > j; D_i = s_i^T y_i.
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

    // Recompute the 2k x 2k middle matrix M = [[theta*S^T*S, L], [L^T, -D]]
    // into the pre-allocated middle_ member.
    void recompute_middle()
    {
        const int k = count_;
        auto S = S_.leftCols(k);

        middle_.topLeftCorner(k, k).noalias() = theta_ * (S.transpose() * S);
        middle_.block(0, k, k, k) = L_.topLeftCorner(k, k);
        middle_.block(k, 0, k, k) = L_.topLeftCorner(k, k).transpose();
        middle_.block(k, k, k, k) = (-D_.head(k)).asDiagonal();
    }

    Eigen::Matrix<Scalar, N, Eigen::Dynamic> S_;  // n x capacity_ (displacement vectors)
    Eigen::Matrix<Scalar, N, Eigen::Dynamic> Y_;  // n x capacity_ (gradient differences)
    Eigen::MatrixX<Scalar> L_;                     // capacity_ x capacity_ (strictly lower triangular)
    Eigen::VectorX<Scalar> D_;                     // capacity_ (diagonal: s_i^T y_i)
    Eigen::MatrixX<Scalar> middle_;                // 2*capacity_ x 2*capacity_ (pre-allocated)
    std::vector<Scalar> rho_;                      // 1 / (s_i^T y_i) for two-loop recursion
    mutable std::vector<Scalar> alpha_;            // pre-allocated for two_loop_recursion
    Scalar theta_{Scalar(1)};
    int count_{0};
    int capacity_;
};

}

#endif
