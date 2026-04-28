#ifndef HPP_GUARD_NABLAPP_DETAIL_ADAPTIVE_BFGS_H
#define HPP_GUARD_NABLAPP_DETAIL_ADAPTIVE_BFGS_H

// Adaptive-theta dense BFGS Hessian approximation.
//
// Maintains a dense B (n x n) that equals the compact L-BFGS operator
// with memory M evaluated at the current iterate: B = theta * I plus
// successive raw BFGS rank-2 updates for each of the M most recent
// (s, y) pairs, with theta = y^T y / s^T y recomputed on every push()
// from the latest pair (Shanno 1978 / N&W eq. 6.20). The identity
// floor of B therefore tracks local curvature magnitude rather than
// being frozen at iteration 0.
//
// The updates are applied **without** Powell damping, matching the
// compact L-BFGS representation B = theta * I - W M^-1 W^T exactly
// (N&W Section 9.2 Theorem 9.2 shows the two are equivalent for raw
// S, Y history). SQP robustness against small or negative s^T y
// comes from three guards: (1) pairs with s^T y <= 0 are rejected
// outright, (2) small-but-positive s^T y is clamped to max(s^T y,
// eps * ||s||^2) in the effective rho before the update, and
// (3) theta is clamped to [1e-10, 1e10] to reject pathological
// scalings from near-degenerate curvature.
//
// Rationale: dense BFGS with a one-time initial scaling (Shanno
// applied only at iteration 0) is NOT equivalent to L-BFGS with
// adaptive theta -- on SQP problems where the Lagrangian curvature
// shifts significantly across the trajectory (e.g. Hock-Schittkowski
// HS026, whose objective Hessian vanishes at the constrained
// optimum) the one-time-scaled dense BFGS drifts and produces QP
// directions dominated by null-space exploration instead of descent.
// Rebuilding from a fresh theta every step prevents that drift at
// the cost of O(M * n^2) work per push.
//
// Cost: O(M * n^2) per push (M rank-2 updates). For M = 10 and
// n = 200 this is ~4e5 flops per step, comparable to
// compact_lbfgs::dense_hessian at matching M. The class is
// allocation-free once constructed at a fixed dimension.
//
// Reference: N&W eq. 6.20 (initial Hessian scaling, Shanno 1978),
//            N&W eq. 8.19 (direct BFGS update),
//            N&W Procedure 18.2, eq. 18.22-18.24 (Powell damping),
//            N&W Section 7.2 (limited-memory BFGS),
//            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3.

#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double,
          int N = nablapp::dynamic_dimension,
          int MaxHistory = 10>
class adaptive_bfgs
{
public:
    explicit adaptive_bfgs(int n = (N == nablapp::dynamic_dimension ? 0 : N))
        : B_(Eigen::Matrix<Scalar, N, N>::Identity(n, n))
        , theta_(Scalar(1))
        , S_(Eigen::Matrix<Scalar, N, MaxHistory>::Zero(n, MaxHistory))
        , Y_(Eigen::Matrix<Scalar, N, MaxHistory>::Zero(n, MaxHistory))
        , STY_(Eigen::Matrix<Scalar, MaxHistory, MaxHistory>::Zero())
        , STS_(Eigen::Matrix<Scalar, MaxHistory, MaxHistory>::Zero())
        , sTy_(Eigen::Matrix<Scalar, MaxHistory, 1>::Zero())
        , head_(0)
        , count_(0)
    {}

    // Record a curvature pair and rebuild B with adaptive theta.
    //
    // Pairs with s^T y <= 0 are rejected outright: the SQP Lagrangian
    // gradient difference can easily have non-positive curvature when
    // the constraint-Hessian contribution (A_{k+1} - A_k)^T lambda
    // dominates the objective curvature, and feeding such a pair into
    // the BFGS formula distorts B rather than improving it. Pairs
    // with small-but-positive s^T y are preserved by clamping the
    // effective sTy to max(s^T y, eps * ||s||^2) (same policy as
    // compact_lbfgs::push), which keeps near-orthogonal pairs from
    // contributing near-infinite rank-2 terms.
    //
    // On success the pair is stored in the ring buffer (evicting the
    // oldest if full), theta is updated from the latest (unclamped)
    // pair, and the dense B is rebuilt from theta * I augmented with
    // raw rank-2 updates across all stored pairs.
    void push(const Eigen::Vector<Scalar, N>& s,
              const Eigen::Vector<Scalar, N>& y)
    {
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

        const Scalar sTs = s.squaredNorm();
        if(!(sTs > eps * eps)) return;

        const Scalar sTy = s.dot(y);
        if(sTy <= Scalar(0)) return;

        const Scalar sTy_effective = std::max(sTy, eps * sTs);

        const Scalar yTy = y.squaredNorm();
        Scalar new_theta = yTy / sTy_effective;
        if(!std::isfinite(new_theta)) return;
        new_theta = std::clamp(new_theta, Scalar(1e-10), Scalar(1e10));

        // FIFO layout (no ring buffer) to keep the S^T S / S^T Y
        // bookkeeping trivial: on overflow the oldest column is
        // shifted out and the new pair goes at column (MaxHistory-1).
        // Matches compact_lbfgs::push storage layout so the compact
        // L-BFGS formula below applies directly.
        if(count_ == MaxHistory)
        {
            for(int j = 0; j < MaxHistory - 1; ++j)
            {
                S_.col(j) = S_.col(j + 1);
                Y_.col(j) = Y_.col(j + 1);
                sTy_[j] = sTy_[j + 1];
            }
            for(int i = 0; i < MaxHistory - 1; ++i)
                for(int j = 0; j < MaxHistory - 1; ++j)
                {
                    STS_(i, j) = STS_(i + 1, j + 1);
                    STY_(i, j) = STY_(i + 1, j + 1);
                }
            S_.col(MaxHistory - 1) = s;
            Y_.col(MaxHistory - 1) = y;
            sTy_[MaxHistory - 1] = sTy;
        }
        else
        {
            S_.col(count_) = s;
            Y_.col(count_) = y;
            sTy_[count_] = sTy;
            ++count_;
        }

        // Update new column entries of STS_ and STY_ (triangular, row/col
        // at index k-1 where k = count_). Older entries are unchanged.
        const int k = count_;
        const int col = k - 1;
        for(int i = 0; i < k; ++i)
        {
            STS_(i, col) = S_.col(i).dot(S_.col(col));
            STS_(col, i) = STS_(i, col);
            STY_(i, col) = S_.col(i).dot(Y_.col(col));
            STY_(col, i) = S_.col(col).dot(Y_.col(i));
        }

        theta_ = new_theta;
        rebuild_();
    }

    const Eigen::Matrix<Scalar, N, N>& hessian() const { return B_; }

    Eigen::Vector<Scalar, N> multiply(const Eigen::Vector<Scalar, N>& v) const
    {
        return (B_ * v).eval();
    }

    void reset()
    {
        const int n = static_cast<int>(B_.rows());
        B_ = Eigen::Matrix<Scalar, N, N>::Identity(n, n);
        theta_ = Scalar(1);
        STS_.setZero();
        STY_.setZero();
        sTy_.setZero();
        head_ = 0;
        count_ = 0;
    }

    int dimension() const { return static_cast<int>(B_.rows()); }
    int history_size() const { return count_; }
    Scalar theta() const { return theta_; }

private:
    // Rebuild B via the Byrd-Nocedal-Schnabel 1994 compact L-BFGS
    // representation (N&W Theorem 9.2, eq. 9.15):
    //
    //   B = theta * I - W * M^{-1} * W^T
    //
    // with
    //   W = [theta * S,  Y]                        (n x 2k)
    //   M = [[theta * S^T S,  L     ],
    //        [L^T,            -D    ]]              (2k x 2k)
    //   L = strictly lower triangle of S^T Y
    //   D = diag(s_i^T y_i)
    //
    // This is the operator produced by applying raw BFGS updates
    // (s_0, y_0), ..., (s_{k-1}, y_{k-1}) to the baseline theta * I,
    // and is numerically more stable than iterating rank-2 updates
    // when theta rescales relative to prior pairs' curvature.
    void rebuild_()
    {
        const int n = static_cast<int>(B_.rows());
        B_.setIdentity(n, n);
        B_ *= theta_;

        const int k = count_;
        if(k == 0) return;

        // Build M (2k x 2k).
        M_.resize(2 * k, 2 * k);
        M_.topLeftCorner(k, k).noalias() = theta_ * STS_.topLeftCorner(k, k);
        // L = strictly lower triangle of STY
        for(int i = 0; i < k; ++i)
            for(int j = 0; j < k; ++j)
                M_(i, k + j) = (i > j) ? STY_(i, j) : Scalar(0);
        M_.bottomLeftCorner(k, k) = M_.topRightCorner(k, k).transpose();
        M_.bottomRightCorner(k, k).setZero();
        for(int i = 0; i < k; ++i)
            M_(k + i, k + i) = -sTy_[i];

        // Build W (n x 2k).
        W_.resize(n, 2 * k);
        W_.leftCols(k).noalias() = theta_ * S_.leftCols(k);
        W_.rightCols(k) = Y_.leftCols(k);

        // Solve M * Z = W^T  ->  Z = M^{-1} W^T   (2k x n).
        // M is symmetric indefinite (it has a -D diagonal block in the
        // lower-right); Eigen::LDLT copes with this for the small 2k x 2k
        // sizes used here (k <= MaxHistory). Match compact_lbfgs's
        // tolerance by not checking ldlt.info().
        ldlt_.compute(M_);
        Z_.resize(2 * k, n);
        Z_ = W_.transpose();
        ldlt_.solveInPlace(Z_);

        // B <- theta * I - W Z
        B_.noalias() -= W_ * Z_;
    }

    Eigen::Matrix<Scalar, N, N> B_;
    Scalar theta_;
    Eigen::Matrix<Scalar, N, MaxHistory> S_;
    Eigen::Matrix<Scalar, N, MaxHistory> Y_;
    Eigen::Matrix<Scalar, MaxHistory, MaxHistory> STS_;
    Eigen::Matrix<Scalar, MaxHistory, MaxHistory> STY_;
    Eigen::Matrix<Scalar, MaxHistory, 1> sTy_;

    // M_, W_, Z_ have Dynamic runtime extents bounded at compile time by
    // (2 * MaxHistory) on the history axis, which lets Eigen route the
    // resize / factorization / solve through stack-allocated scratch
    // instead of per-push heap allocation. ldlt_ is hoisted to a member
    // so its internal m_matrix copy reuses the same fixed-max storage.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                   2 * MaxHistory, 2 * MaxHistory> M_;
    Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0, N, 2 * MaxHistory> W_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N, 0, 2 * MaxHistory, N> Z_;
    Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                               2 * MaxHistory, 2 * MaxHistory>> ldlt_;
    int head_;
    int count_;
};

}

#endif
