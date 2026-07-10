#ifndef HPP_GUARD_ARGMIN_DETAIL_COMPACT_LBFGS_H
#define HPP_GUARD_ARGMIN_DETAIL_COMPACT_LBFGS_H

#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

// Factorization strategy for the (structurally indefinite) 2m x 2m middle
// matrix M = [[theta S^T S, L], [L^T, -D]] appearing in the compact L-BFGS
// representation. Eigen::LDLT is out of contract here (it targets positive
// or negative semidefinite matrices); a near-zero -D pivot with large L
// coupling drives element growth and corrupts B*v. Both strategies below
// are in-contract for this indefinite-but-invertible system:
//
//   partial_piv_lu — dense PartialPivLU of the assembled M. Row pivoting
//                    bounds element growth for a general invertible matrix.
//   bns            — Byrd-Nocedal-Schnabel 1994 structured inverse: form
//                    the SPD Schur complement theta S^T S + L D^{-1} L^T
//                    (well-defined because the curvature clamp keeps every
//                    D diagonal strictly positive), Cholesky-factor it, and
//                    back-substitute for the -D block. Uses only in-contract
//                    Cholesky plus a diagonal solve.
//
// Reference: Byrd, Nocedal, Schnabel (1994), "Representations of quasi-Newton
//            matrices and their use in limited memory methods", Math. Prog.
//            63, 129-156 (middle-matrix inverse); Byrd, Lu, Nocedal, Zhu
//            1995 (L-BFGS-B reference code, middle-matrix factorization).
enum class lbfgs_middle_solve { partial_piv_lu, bns };

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

// Default factorization is partial_piv_lu: an empirical A/B of B*v against a
// long-double reference on near-degenerate curvature sequences (see
// compact_lbfgs_test.cpp) shows PartialPivLU stays at ~1e-16 relative while
// the BNS Schur-complement route degrades to ~1e-1 once a D diagonal reaches
// the clamp floor, because BNS forms D^{-1} explicitly and amplifies the tiny
// pivot. PartialPivLU never inverts D and pivots for stability, so it is the
// in-contract choice of equal cost with strictly better accuracy here.
template <typename Scalar = double, int N = argmin::dynamic_dimension, int MaxHistory = 7,
          lbfgs_middle_solve Factorization = lbfgs_middle_solve::partial_piv_lu>
class compact_lbfgs
{
    static_assert(MaxHistory > 0, "MaxHistory must be positive");

    using dyn_matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Max-bounded workspace types. The 2k middle system is bounded by
    // 2 * MaxHistory regardless of N; the nf/n axes are bounded by N (dynamic
    // when N is dynamic, exactly the subspace_minimization.h precedent).
    using middle_matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, 2 * MaxHistory, 2 * MaxHistory>;
    using middle_vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 2 * MaxHistory, 1>;
    using middle_panel = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, 2 * MaxHistory, N>;
    using reduced_matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, N, N>;

public:
    compact_lbfgs()
    {
        STS_.setZero();
        L_.setZero();
        D_.setZero();
        middle_.setZero();
    }

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

        const Eigen::Index n = s.size();

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

        // Curvature-clamp consistency: update_new_column / shift_evict_and_update
        // recompute the freshly stored D diagonal as the RAW s^T y, whereas
        // rho_ and theta_ use the clamped s^T y (max(s^T y, eps ||s||^2)).
        // Overwrite the newest D entry (always at index count_ - 1 on both the
        // append and the evict-and-shift paths) with the same clamped value so
        // the -D block of the middle matrix, rho_, and theta_ agree. Invariant:
        // no D diagonal entry is below the clamp floor post-push, which keeps
        // the BNS Schur complement theta S^T S + L D^{-1} L^T positive definite.
        D_[count_ - 1] = sTy_effective;

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

        // W^T * v assembled directly into the hoisted 2k stacked rhs buffer
        // [theta S^T v; Y^T v]. mrhs_ is middle-dimension max-bounded
        // (2*MaxHistory), so the resize never touches the heap. noalias() safe:
        // the head/tail segments are distinct from all source operands.
        mrhs_.resize(2 * k);
        mrhs_.head(k).noalias() = theta_ * (S.transpose() * v);
        mrhs_.tail(k).noalias() = Y.transpose() * v;

        // Solve M * z = W^T * v with the hoisted middle-matrix factorization.
        mz_.resize(2 * k);
        solve_middle_into(mrhs_, mz_);

        // B*v = theta*v - W*z accumulated in the hoisted N-dimension buffer.
        // noalias() safe: Wz_ is distinct from S, Y, mz_.
        Wz_.resize(v.size());
        Wz_.noalias() = theta_ * (S * mz_.head(k));
        Wz_.noalias() += Y * mz_.tail(k);
        return (theta_ * v - Wz_).eval();
    }

    // Build reduced Hessian Z^T * B * Z for free variable subspace.
    // Z selects rows in free_indices from the identity matrix.
    // B_FF = theta * I_F - W_F * M^{-1} * W_F^T.
    //
    // The free-variable count nf varies per call but is bounded by N, and the
    // 2k middle system is bounded by 2*MaxHistory, so every buffer here is a
    // max-bounded type (the subspace_minimization free-Hessian precedent): at a
    // compile-time N the reduced Hessian, the WF panel, and the middle solve
    // are all stack-resident and the bound-active path allocates nothing.
    // Reference: N&W Section 16.6 (subspace minimization).
    reduced_matrix reduced_hessian(const std::vector<int>& free_indices) const
    {
        const int nf = static_cast<int>(free_indices.size());
        reduced_matrix BFF;
        if(nf == 0) return BFF;

        BFF.resize(nf, nf);
        BFF.setIdentity();
        BFF *= theta_;
        if(count_ == 0) return BFF;

        const int k = count_;

        // Build W_F^T (2k x nf) directly in the middle-panel buffer (rows
        // bounded by 2*MaxHistory, cols by N). Storing the transpose keeps the
        // bounded row axis on the 2*MaxHistory side, which stays a valid
        // column-major max-bounded shape even when N == 1.
        WFt_.resize(2 * k, nf);
        for(int j = 0; j < nf; ++j)
        {
            int idx = free_indices[static_cast<std::size_t>(j)];
            WFt_.col(j).head(k) = theta_ * S_.row(idx).head(k).transpose();
            WFt_.col(j).tail(k) = Y_.row(idx).head(k).transpose();
        }

        MiWt_.resize(2 * k, nf);
        solve_middle_into(WFt_, MiWt_);

        // B_FF = theta*I_F - W_F * M^{-1} * W_F^T = theta*I_F - (W_F^T)^T * MiWt.
        BFF.noalias() -= WFt_.transpose() * MiWt_;
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

        // Solve M * Z = W^T => Z = M^{-1} * W^T, size 2k x n, into the hoisted
        // middle-panel buffer (rows bounded by 2*MaxHistory, cols by N).
        MiWt_.resize(2 * k, n);
        solve_middle_into(W.transpose(), MiWt_);

        // B = theta*I - W * Z
        B.noalias() -= W * MiWt_;

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

    // Smallest curvature s^T y currently stored on the middle-matrix -D
    // diagonal. Post-push invariant: this is never below the clamp floor
    // (max(s^T y, eps ||s||^2) is stored, so every entry is > 0), which keeps
    // the -D block nonsingular and any downstream D^{-1} well-defined. Returns
    // theta_ when the history is empty (no curvature stored yet).
    Scalar min_stored_curvature() const
    {
        if(count_ == 0) return theta_;
        return D_.head(count_).minCoeff();
    }

    void reset()
    {
        count_ = 0;
        theta_ = Scalar(1);
        STS_.setZero();
        L_.setZero();
        D_.setZero();
        middle_.setZero();
    }

    int size() const { return count_; }
    static constexpr int capacity() { return MaxHistory; }

private:
    // Solve M * Z = rhs for the 2k x 2k active middle matrix
    //   M = [[theta S^T S, L], [L^T, -D]],
    // writing the solution into the caller-provided max-bounded destination.
    // rhs is stacked [r1 (k rows); r2 (k rows)] and may carry multiple columns.
    // M is symmetric indefinite; the strategy is chosen at compile time by the
    // Factorization template parameter. The partial_piv_lu path reuses the
    // member M_active_ buffer and the member PartialPivLU factorization so no
    // heap traffic occurs on the bound-active hot path.
    //
    // Reference: Byrd, Nocedal, Schnabel 1994 (structured middle-matrix
    //            inverse); N&W Section 9.2, eq. 9.15.
    template <typename Rhs, typename Dst>
    void solve_middle_into(const Eigen::MatrixBase<Rhs>& rhs,
                           Eigen::MatrixBase<Dst>& dst) const
    {
        const int k = count_;

        if constexpr(Factorization == lbfgs_middle_solve::bns)
        {
            // BNS structured solve. With M z = [r1; r2]:
            //   z2 = D^{-1} (L^T z1 - r2)
            //   (theta S^T S + L D^{-1} L^T) z1 = r1 + L D^{-1} r2.
            // The Schur complement is SPD because every D diagonal is kept
            // strictly positive by the curvature clamp, so an in-contract
            // Cholesky (LLT) factorizes it. Only a diagonal solve remains for
            // the -D block. This never forms an indefinite factorization.
            const Eigen::Vector<Scalar, Eigen::Dynamic> Dinv =
                D_.head(k).cwiseInverse();
            const dyn_matrix L = L_.topLeftCorner(k, k);
            const dyn_matrix r1 = rhs.topRows(k);
            const dyn_matrix r2 = rhs.bottomRows(k);
            const dyn_matrix Dinv_r2 = Dinv.asDiagonal() * r2;

            dyn_matrix schur = theta_ * STS_.topLeftCorner(k, k);
            schur.noalias() += L * (Dinv.asDiagonal() * L.transpose());

            Eigen::LLT<dyn_matrix> llt(schur);
            const dyn_matrix z1 = llt.solve(r1 + L * Dinv_r2);
            const dyn_matrix z2 = Dinv.asDiagonal() * (L.transpose() * z1 - r2);

            dst.topRows(k) = z1;
            dst.bottomRows(k) = z2;
        }
        else
        {
            // PartialPivLU of the assembled 2k x 2k middle matrix. Row
            // pivoting bounds element growth for a general invertible system
            // (in contract for the indefinite M, unlike Eigen::LDLT). The
            // buffer and the factorization are members: at a compile-time N
            // (and always for the 2*MaxHistory middle system) the resize and
            // the factorization stay stack-resident.
            M_active_.resize(2 * k, 2 * k);
            M_active_.topLeftCorner(k, k) = middle_.topLeftCorner(k, k);
            M_active_.block(0, k, k, k) = middle_.block(0, MaxHistory, k, k);
            M_active_.block(k, 0, k, k) = middle_.block(MaxHistory, 0, k, k);
            M_active_.block(k, k, k, k) = middle_.block(MaxHistory, MaxHistory, k, k);

            lu_.compute(M_active_);
            dst = lu_.solve(rhs);
        }
    }

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

    // Hoisted middle-solve workspace (mutable: the products are const methods).
    mutable middle_matrix M_active_{};
    mutable Eigen::PartialPivLU<middle_matrix> lu_{};
    mutable middle_vector mrhs_{};        // multiply(): stacked W^T v
    mutable middle_vector mz_{};          // multiply(): M^{-1} W^T v
    mutable Eigen::Vector<Scalar, N> Wz_{};  // multiply(): W * z accumulation
    mutable middle_panel WFt_{};          // reduced_hessian(): free-variable panel W_F^T
    mutable middle_panel MiWt_{};         // reduced_/dense_hessian(): M^{-1} W^T

    Scalar theta_{Scalar(1)};
    int count_{0};
};

}

#endif
