#ifndef HPP_GUARD_NABLAPP_DETAIL_DENSE_LDL_BFGS_H
#define HPP_GUARD_NABLAPP_DETAIL_DENSE_LDL_BFGS_H

// Packed LDL^T BFGS Hessian approximation with Powell-damped rank-1
// updates.
//
// Maintains a single Hessian B = L * D * L^T, where L is unit lower
// triangular and D is positive diagonal, packed column-by-column in
// an n*(n+1)/2 array (column j stores D[j] in slot 0 then L[j+1, j]
// .. L[n-1, j] in slots 1..n-1-j). On each push() of a curvature
// pair (s, y) the class computes B*s, applies Powell damping per
// N&W eq. 18.22-18.24 to ensure positive curvature, and applies
// two Fletcher-Powell rank-1 LDL updates: positive 1/(s^T u) * u *
// u^T (with u the damped y), then negative -1/(s^T B s) * (B s) *
// (B s)^T. Net effect: B_new = B - (Bs)(Bs)^T / s^T B s + u u^T /
// s^T u, the classical direct BFGS formula (N&W eq. 8.19) with a
// damping safeguard against indefinite Lagrangian curvature.
//
// Cost: O(n^2) per push (two rank-1 updates plus one B*s multiply
// against the packed factors). The classical compact L-BFGS rebuild
// in adaptive_bfgs scales as O(M * n^2) where M is the kept history
// length, making the LDL form M-times cheaper at typical M = 10
// while preserving the entire BFGS history (the LDL absorbs every
// rank-1 contribution into a single dense factor).
//
// Lazy hessian(): consumers that need the dense B (e.g. SQP
// policies that pass B to active_set_qp_solver) call hessian(),
// which materializes B = L * D * L^T into a cached buffer on the
// first call after each push. SQP policies that consume the
// Cholesky factor of B directly (kraft_lsq_qp_solver in the LSQ
// cast min ||E p - f||^2) call factor_to_E_and_f() instead, which
// writes E = sqrt(D) * L^T and f = -D^{-1/2} L^{-1} g without ever
// forming B -- skipping the O(n^3 / 3) LLT(B) the QP solver would
// otherwise run on every solve.
//
// Theta scaling: NLopt's slsqpb_ initializes the LDL factors with
// L = I, D = I (so B_0 = I) and never rescales -- the rank-1
// updates carry curvature into the factor automatically. This
// drops the per-push theta = y^T y / s^T y rescale that
// adaptive_bfgs runs (Shanno 1978 / N&W eq. 6.20). The trade is
// stability of B across iterations (no spurious resets) versus
// adaptivity at SQP problems with rapidly shifting Lagrangian
// curvature; HS026-class problems are the canary -- validate
// outer-iter counts there before declaring a regression.
//
// References:
//   Fletcher, R. & Powell, M.J.D. (1974). On the modification of
//     LDL' factorization. Math. Computation 28, 1067-1078.
//   Kraft, D. (1988). DFVLR-FB 88-28 Section 2.2.3 (BFGS update);
//     slsqpb_ subroutine (NLopt slsqp.c lines 2127-2183).
//   N&W 2e Procedure 18.2, eq. 18.22-18.24 (Powell damping);
//     eq. 8.19 (direct BFGS update); Section 7.2 (limited memory).

#include "nablapp/types.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cstdint>

namespace nablapp::detail
{

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class dense_ldl_bfgs
{
public:
    explicit dense_ldl_bfgs(int n = (N == nablapp::dynamic_dimension ? 0 : N))
        : n_(n)
        , L_packed_(packed_size_(n))
        , B_(Eigen::Matrix<Scalar, N, N>::Identity(n, n))
        , Bs_(Eigen::Vector<Scalar, N>::Zero(n))
        , u_workspace_(Eigen::Vector<Scalar, N>::Zero(n))
        , v_workspace_(Eigen::Vector<Scalar, N>::Zero(n))
        , w_workspace_(Eigen::Vector<Scalar, N>::Zero(n))
        , B_dirty_(false)
    {
        reset();
    }

    // Clear the stored history: L = I, D = I, B = I.
    void reset()
    {
        // Pack L = I, D = I into the column-major flat array.
        // Column j layout: slot col_start_(j) holds D[j] = 1, slots
        // col_start_(j) + 1 .. + (n - 1 - j) hold L[j+1, j] .. L[n-1, j] = 0.
        std::size_t idx = 0;
        for(int j = 0; j < n_; ++j)
        {
            L_packed_[idx] = Scalar(1);
            ++idx;
            for(int i = j + 1; i < n_; ++i)
            {
                L_packed_[idx] = Scalar(0);
                ++idx;
            }
        }
        B_.setIdentity(n_, n_);
        B_dirty_ = false;
        updates_since_reset_ = 0;
    }

    // Apply a Powell-damped BFGS update for the curvature pair (s, y).
    //
    // Steps (NLopt slsqp.c slsqpb_ block at L260):
    //   1. v = B * s via packed L, D.
    //   2. h1 = s^T y, h2 = s^T v.
    //   3. If h1 < 0.2 * h2 (Powell damping condition):
    //        h4 = (h2 - 0.2 * h2) / (h2 - h1)
    //        u  = h4 * y + (1 - h4) * v        (damped y)
    //        h1 = 0.2 * h2 (algebraically guaranteed = s^T u after damping)
    //      else u = y unchanged, h1 = s^T y.
    //   4. ldl_update(u, +1 / h1)
    //   5. ldl_update(v, -1 / h2)
    //
    // Pairs with s near-zero (||s|| < eps) or with non-positive damped
    // curvature (h2 <= 0) are rejected. The damping always produces
    // s^T u >= 0.2 * s^T B s > 0 when h2 > 0, so the positive update
    // is well-defined; the negative update relies on h2 > 0 (B is SPD
    // by induction).
    void push(const Eigen::Vector<Scalar, N>& s,
              const Eigen::Vector<Scalar, N>& y)
    {
        using std::isfinite;
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

        if(n_ <= 0) return;

        const Scalar sTs = s.squaredNorm();
        if(!(sTs > eps * eps)) return;

        // Shanno initial-Hessian rescale (N&W eq. 6.20). On the first
        // accepted (s, y) pair after construction or reset, replace
        // B = I with B = (y^T y / s^T y) * I before applying the rank-1
        // updates. This adapts the Hessian baseline to the local
        // curvature magnitude and is the canonical SQP/BFGS warm-start
        // (Kraft 1988 §2.2.3, NLopt sets it explicitly only at
        // iteration 0; subsequent pushes evolve B incrementally without
        // rescaling). On near-degenerate Lagrangian curvature problems
        // (HS026 family) the initial scaling determines whether B
        // tracks the small-magnitude objective Hessian or stays stuck
        // at the identity baseline.
        if(updates_since_reset_ == 0)
        {
            const Scalar sTy_init = s.dot(y);
            if(sTy_init > Scalar(0))
            {
                const Scalar yTy_init = y.squaredNorm();
                const Scalar theta = yTy_init / sTy_init;
                if(isfinite(theta) && theta > Scalar(0))
                {
                    const Scalar theta_clamped = std::clamp(
                        theta, Scalar(1e-10), Scalar(1e10));
                    // L stays identity (all sub-diagonals already 0);
                    // scale only the stored diagonals D[i] = 1 -> theta.
                    for(int i = 0; i < n_; ++i)
                    {
                        L_packed_[col_start_(i)] = theta_clamped;
                    }
                }
            }
        }

        // v = B * s using packed factors. Three-pass:
        //   pass 1: v = L^T * s  (upper sweep over packed sub-diags)
        //   pass 2: v = D * v    (scale by stored diagonals)
        //   pass 3: v = L * v    (lower sweep)
        multiply_in_place_(s, v_workspace_);
        const Scalar h2 = s.dot(v_workspace_);
        if(!(h2 > Scalar(0))) return;  // B no longer SPD or s in null space

        const Scalar h1_raw = s.dot(y);

        // Powell damping (N&W eq. 18.22-18.24).
        Scalar h1;
        const Scalar h3 = Scalar(0.2) * h2;
        if(h1_raw < h3)
        {
            const Scalar h4 = (h2 - h3) / (h2 - h1_raw);
            // u = h4 * y + (1 - h4) * v_workspace_
            u_workspace_.head(n_).noalias() = h4 * y.head(n_)
                                            + (Scalar(1) - h4) * v_workspace_.head(n_);
            h1 = h3;
        }
        else
        {
            u_workspace_.head(n_) = y.head(n_);
            h1 = h1_raw;
        }
        if(!(h1 > Scalar(0))) return;

        // Positive rank-1 update: B += (1/h1) * u * u^T.
        ldl_update_(Scalar(1) / h1, u_workspace_);

        // Negative rank-1 update: B -= (1/h2) * v * v^T (with v = B*s
        // computed BEFORE the positive update, as in NLopt).
        ldl_update_(-Scalar(1) / h2, v_workspace_);

        ++updates_since_reset_;
        B_dirty_ = true;
    }

    // Lazy materialization of the dense Hessian B = L * D * L^T.
    //
    // First call after a push() rebuilds B in O(n^3); subsequent
    // calls are O(1). Used by SQP policies that hand B off to a QP
    // solver expecting a dense matrix (active_set_qp / solve_qp);
    // policies that consume the Cholesky factor directly should call
    // factor_to_E_and_f() to skip the materialization entirely.
    const Eigen::Matrix<Scalar, N, N>& hessian() const
    {
        if(B_dirty_)
        {
            materialize_B_();
            B_dirty_ = false;
        }
        return B_;
    }

    // Compute B * v using the packed factors directly. Cost O(n^2).
    Eigen::Vector<Scalar, N> multiply(const Eigen::Vector<Scalar, N>& v) const
    {
        Eigen::Vector<Scalar, N> out(n_);
        multiply_in_place_(v, out);
        return out;
    }

    // Write the upper-triangular Cholesky-like factor E = sqrt(D) * L^T
    // into E_out and the QP RHS f = -D^{-1/2} * L^{-1} * g into f_out.
    //
    // The kraft_lsq_qp_solver casts the QP min 0.5 p^T B p + g^T p as
    // the least-squares min ||E p - f||^2 with E^T E = B and E^T f = -g.
    // For B = L D L^T we get E = sqrt(D) * L^T directly (upper
    // triangular, with diagonal sqrt(D[i]) and off-diagonals
    // sqrt(D[i]) * L[j, i] for i < j). The RHS solves L * sqrt(D) * f
    // = -g, i.e. forward-solve in L for L^{-1} (-g), then divide by
    // sqrt(D[i]).
    //
    // Cost: O(n^2). Replaces the LLT(B) factorization that
    // kraft_lsq_qp_solver runs internally on every solve when given a
    // dense B (Eigen::LLT is O(n^3 / 3)).
    void factor_to_E_and_f(Eigen::Matrix<Scalar, N, N>& E_out,
                           const Eigen::Vector<Scalar, N>& g,
                           Eigen::Vector<Scalar, N>& f_out) const
    {
        using std::sqrt;

        if(E_out.rows() != n_ || E_out.cols() != n_) E_out.resize(n_, n_);
        if(f_out.size() != n_) f_out.resize(n_);

        E_out.setZero();
        // Forward-solve L * f_tmp = -g, where L is unit lower (diagonal
        // entries are stored as D in the packed array but treated as 1
        // for the L solve). f_tmp[i] = -g[i] - sum_{j<i} L[i, j] *
        // f_tmp[j].
        for(int i = 0; i < n_; ++i)
        {
            Scalar acc = -g[i];
            for(int j = 0; j < i; ++j)
            {
                acc -= L_at_(i, j) * f_out[j];
            }
            f_out[i] = acc;
        }

        // Build E = sqrt(D) * L^T row-by-row, and finalize f = D^{-1/2}
        // * f_tmp. E[i, j] = sqrt(D[i]) * L[j, i] for j >= i (with
        // L[i, i] = 1).
        for(int i = 0; i < n_; ++i)
        {
            const Scalar di = D_at_(i);
            const Scalar sqrt_di = sqrt(di);
            E_out(i, i) = sqrt_di;
            for(int j = i + 1; j < n_; ++j)
            {
                E_out(i, j) = sqrt_di * L_at_(j, i);
            }
            f_out[i] /= sqrt_di;
        }
    }

    int dimension() const { return n_; }

private:
    static std::size_t packed_size_(int n) noexcept
    {
        return static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1) / 2;
    }

    // Index of the diagonal entry of column j in the packed array.
    // Column 0 starts at 0, column 1 at n, column 2 at n + (n-1), etc.
    // Closed form: j * n - j*(j-1)/2.
    std::size_t col_start_(int j) const noexcept
    {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(n_)
             - static_cast<std::size_t>(j) * static_cast<std::size_t>(j - 1) / 2;
    }

    // L[i, j] for i > j (sub-diagonal entry).
    Scalar L_at_(int i, int j) const noexcept
    {
        return L_packed_[col_start_(j) + static_cast<std::size_t>(i - j)];
    }

    // D[i] (stored as the diagonal slot of column i in packed L).
    Scalar D_at_(int i) const noexcept
    {
        return L_packed_[col_start_(i)];
    }

    // out = B * v using the three-pass L * D * L^T multiply (matches
    // NLopt slsqpb_ at L260-L320).
    void multiply_in_place_(const Eigen::Vector<Scalar, N>& v,
                            Eigen::Vector<Scalar, N>& out) const
    {
        if(out.size() != n_) out.resize(n_);

        // Pass 1: out = L^T * v. out[i] = v[i] + sum_{j > i} L[j, i] * v[j].
        for(int i = 0; i < n_; ++i)
        {
            Scalar acc = v[i];
            for(int j = i + 1; j < n_; ++j)
            {
                acc += L_at_(j, i) * v[j];
            }
            out[i] = acc;
        }

        // Pass 2: out = D * out.
        for(int i = 0; i < n_; ++i)
        {
            out[i] *= D_at_(i);
        }

        // Pass 3: out = L * out. out[i] += sum_{j < i} L[i, j] * out[j].
        // Sweep top-down with a temporary so we don't read after writing.
        for(int i = n_ - 1; i >= 0; --i)
        {
            Scalar acc = Scalar(0);
            for(int j = 0; j < i; ++j)
            {
                acc += L_at_(i, j) * out[j];
            }
            out[i] += acc;
        }
    }

    // Materialize B = L * D * L^T into B_ (lower triangle written, then
    // mirrored across the diagonal). Cost O(n^3 / 3).
    void materialize_B_() const
    {
        B_.setZero(n_, n_);
        // B[i, k] = sum_j L[i, j] * D[j] * L[k, j], for i >= k.
        // Compute lower triangle with the j loop innermost.
        for(int i = 0; i < n_; ++i)
        {
            for(int k = 0; k <= i; ++k)
            {
                Scalar acc = Scalar(0);
                // j ranges 0..min(i, k) = k since k <= i.
                for(int j = 0; j <= k; ++j)
                {
                    const Scalar lij = (i == j) ? Scalar(1) : L_at_(i, j);
                    const Scalar lkj = (k == j) ? Scalar(1) : L_at_(k, j);
                    acc += lij * D_at_(j) * lkj;
                }
                B_(i, k) = acc;
                if(i != k) B_(k, i) = acc;
            }
        }
    }

    // Fletcher-Powell rank-1 update: B := B + sigma * z * z^T, applied
    // to the packed L D L^T factors in-place. z is destroyed; for
    // sigma < 0 the workspace w_workspace_ is also overwritten. Mirrors
    // NLopt's ldl_ exactly (slsqp.c L1437) but in 0-indexed form.
    void ldl_update_(Scalar sigma, Eigen::Vector<Scalar, N>& z)
    {
        if(sigma == Scalar(0)) return;

        constexpr Scalar epmach = std::numeric_limits<Scalar>::epsilon();
        Scalar t = Scalar(1) / sigma;
        std::size_t ij = 0;

        if(sigma < Scalar(0))
        {
            // Negative-update preparation: factor out the negative
            // rank-1 contribution from the running diagonal scratch
            // t before the forward sweep, so the sub-diagonal updates
            // see the correctly-rescaled denominator.
            w_workspace_.head(n_) = z.head(n_);
            for(int i = 0; i < n_; ++i)
            {
                const Scalar v = w_workspace_[i];
                t += v * v / L_packed_[ij];
                for(int j = i + 1; j < n_; ++j)
                {
                    ++ij;
                    w_workspace_[j] -= v * L_packed_[ij];
                }
                ++ij;
            }
            if(t >= Scalar(0))
            {
                // Numerical safeguard: t collapsed to non-negative,
                // which would make D[i] hit zero or flip sign. Floor
                // it at machine epsilon (matches NLopt slsqp.c:1514).
                t = epmach / sigma;
            }
            // Reverse sweep to fill w_workspace_ with the per-column
            // running denominators consumed by the update phase.
            for(int i = 0; i < n_; ++i)
            {
                const int jcol = n_ - 1 - i;
                ij -= static_cast<std::size_t>(i + 1);
                const Scalar u = w_workspace_[jcol];
                w_workspace_[jcol] = t;
                t -= u * u / L_packed_[ij];
            }
        }

        // Update phase: walk forward through columns applying the
        // rank-1 to each (D[i], L sub-diag) in turn.
        ij = 0;
        for(int i = 0; i < n_; ++i)
        {
            const Scalar v = z[i];
            const Scalar delta = v / L_packed_[ij];
            const Scalar tp = (sigma < Scalar(0)) ? w_workspace_[i] : t + delta * v;
            const Scalar alpha = tp / t;
            L_packed_[ij] = alpha * L_packed_[ij];
            if(i == n_ - 1) break;
            const Scalar beta = delta / tp;
            if(alpha > Scalar(4))
            {
                // Numerically robust branch from Fletcher-Powell:
                // when alpha is large, swap the order of accumulation
                // to keep the L update bounded.
                const Scalar gamma_ = t / tp;
                for(int j = i + 1; j < n_; ++j)
                {
                    ++ij;
                    const Scalar u = L_packed_[ij];
                    L_packed_[ij] = gamma_ * u + beta * z[j];
                    z[j] -= v * u;
                }
            }
            else
            {
                for(int j = i + 1; j < n_; ++j)
                {
                    ++ij;
                    z[j] -= v * L_packed_[ij];
                    L_packed_[ij] += beta * z[j];
                }
            }
            ++ij;
            t = tp;
        }
    }

    int n_;
    std::vector<Scalar> L_packed_;
    mutable Eigen::Matrix<Scalar, N, N> B_;
    Eigen::Vector<Scalar, N> Bs_;
    Eigen::Vector<Scalar, N> u_workspace_;
    Eigen::Vector<Scalar, N> v_workspace_;
    Eigen::Vector<Scalar, N> w_workspace_;
    mutable bool B_dirty_;
    // Counts pushes since the last reset / construction. Used to gate
    // the Shanno initial-Hessian rescale (N&W eq. 6.20) to fire only
    // on the first accepted curvature pair after each reset.
    std::uint32_t updates_since_reset_{0};
};

}

#endif
