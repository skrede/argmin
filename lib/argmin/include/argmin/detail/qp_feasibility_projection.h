#ifndef HPP_GUARD_ARGMIN_DETAIL_QP_FEASIBILITY_PROJECTION_H
#define HPP_GUARD_ARGMIN_DETAIL_QP_FEASIBILITY_PROJECTION_H

#include "argmin/detail/ldp.h"
#include "argmin/detail/nnls.h"
#include "argmin/detail/householder_apply.h"

#include <Eigen/QR>
#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>
#include <algorithm>

namespace argmin::detail
{

enum class feasibility_projection { moved, unchanged, inconsistent };

template <typename DerivedA, typename DerivedB, typename DerivedX, typename Scalar>
bool violates_inequality_rows(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b,
                              const Eigen::MatrixBase<DerivedX>& x, std::int32_t m_eq, std::int32_t m_total, Scalar tol)
{
    for(std::int32_t i = m_eq; i < m_total; ++i)
        if(std::isfinite(b[i]) && A.row(i).dot(x) - b[i] < -tol)
            return true;
    return false;
}

// Moves a point that satisfies the equality rows A_eq x = b_eq to the nearest point that also satisfies every finite
// inequality row A_i x >= b_i: the least-distance program over the null space of the equality rows, solved through
// its dual NNLS (Lawson and Hanson, Solving Least Squares Problems, 1974, chapter 23). The equality rows enter through
// a rank-revealing factorization, so a dependent row is dropped rather than divided by. inconsistent needs a checked
// proof that no point meets every row within verify_tol (1 + |a_i| |x| + |b_i|); a failed solve leaves x unchanged.
template <typename Scalar>
class qp_feasibility_projection
{
    using matrix = Eigen::MatrixX<Scalar>;
    using vector = Eigen::VectorX<Scalar>;

public:
    qp_feasibility_projection(std::int32_t n, std::int32_t max_rows)
        : G_(max_rows, n)
        , h_(max_rows)
        , pad_(n, n)
        , row_(n)
        , move_(n)
        , scaled_(n)
        , nnls_A_(n + 1, max_rows)
        , nnls_b_(n + 1)
        , nnls_x_(max_rows)
        , nnls_w_(max_rows)
        , lambda_(max_rows)
        , allowance_(max_rows)
        , cut_(max_rows)
        , nnls_ws_()
        , qr_(n, n)
    {
        nnls_ws_.ensure(n + 1, max_rows);
        nnls_ws_.r_buf.resize(n + 1);
    }

    qp_feasibility_projection() = default;

    // A row left out for its slack can bind after the move, so a move that violates one is solved again over every row.
    template <typename DerivedA, typename DerivedB, typename DerivedX>
    feasibility_projection project(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, Eigen::MatrixBase<DerivedX>& x,
                                   std::int32_t m_eq, std::int32_t m_total, Scalar tol, Scalar verify_tol)
    {
        const Eigen::Index rank = factor_equalities(A, m_eq);
        feasibility_projection outcome = attempt(A, b, x, m_eq, m_total, rank, verify_tol, far_slack);
        if(outcome == feasibility_projection::moved && cut_row_violated(A, b, x, m_eq, tol))
            outcome = attempt(A, b, x, m_eq, m_total, rank, verify_tol, std::numeric_limits<Scalar>::infinity());
        if(outcome == feasibility_projection::moved)
            x += move_;
        return outcome;
    }

private:
    matrix G_;
    vector h_;
    matrix pad_;
    vector row_;
    vector move_;
    vector scaled_;
    matrix nnls_A_;
    vector nnls_b_;
    vector nnls_x_;
    vector nnls_w_;
    vector lambda_;
    vector allowance_;
    Eigen::ArrayX<bool> cut_;
    nnls_workspace<Scalar> nnls_ws_;
    static constexpr Scalar far_slack{100};
    Eigen::ColPivHouseholderQR<matrix> qr_;

    // The factor of A_eq^T is padded to n x n with zero columns, which add no rank, so its storage keeps one shape;
    // the prescribed threshold is the epsilon * m_eq the unpadded factor would use.
    template <typename DerivedA>
    Eigen::Index factor_equalities(const Eigen::MatrixBase<DerivedA>& A, std::int32_t m_eq)
    {
        if(m_eq == 0)
            return 0;
        pad_.setZero();
        pad_.leftCols(m_eq) = A.topRows(m_eq).transpose();
        qr_.setThreshold(std::numeric_limits<Scalar>::epsilon() * Scalar(m_eq));
        qr_.compute(pad_);
        return qr_.rank();
    }

    template <typename DerivedA, typename DerivedB, typename DerivedX>
    feasibility_projection attempt(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, const Eigen::MatrixBase<DerivedX>& x,
                                   std::int32_t m_eq, std::int32_t m_total, Eigen::Index rank, Scalar verify_tol, Scalar cut)
    {
        if(!stack_rows(A, b, x, m_eq, m_total, rank, verify_tol))
            return feasibility_projection::inconsistent;
        const Scalar scale = h_.maxCoeff();
        if(!(scale > Scalar(0)))
            return feasibility_projection::unchanged;
        scale_rows(scale, cut);
        if(ldp<Scalar, Eigen::Dynamic, Eigen::Dynamic>(G_, h_, move_, lambda_, nnls_A_, nnls_b_, nnls_x_, nnls_w_, nnls_ws_,
                                                         static_cast<std::int32_t>(G_.rows()), static_cast<std::int32_t>(G_.cols())) != 1)
            return certifies_inconsistency() ? feasibility_projection::inconsistent : feasibility_projection::unchanged;
        move_ *= scale;
        if(rank > 0)
            apply_householder_q_on_the_left(qr_, rank, false, move_, scaled_);
        return feasibility_projection::moved;
    }

    // Every inequality slot keeps its row so the program has one shape; an absent or left-out row is a zero row, 0 >= 0.
    template <typename DerivedA, typename DerivedB, typename DerivedX>
    bool stack_rows(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, const Eigen::MatrixBase<DerivedX>& x,
                    std::int32_t m_eq, std::int32_t m_total, Eigen::Index rank, Scalar verify_tol)
    {
        G_.setZero();
        h_.setZero();
        allowance_.setZero();
        for(std::int32_t i = m_eq; i < m_total; ++i)
            if(std::isfinite(b[i]) && !stack_row(A, b, x, i, i - m_eq, rank, verify_tol))
                return false;
        return true;
    }

    // A row whose restriction to the null space vanishes is fixed by the equality rows, so it either holds within its
    // allowance or makes the rows inconsistent.
    template <typename DerivedA, typename DerivedB, typename DerivedX>
    bool stack_row(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, const Eigen::MatrixBase<DerivedX>& x,
                   std::int32_t i, Eigen::Index k, Eigen::Index rank, Scalar verify_tol)
    {
        row_ = A.row(i).transpose();
        if(rank > 0)
            apply_householder_q_on_the_left(qr_, rank, true, row_, scaled_);
        row_.head(rank).setZero();
        const Scalar norm = row_.norm();
        const Scalar violation = b[i] - A.row(i).dot(x);
        const Scalar allowance = verify_tol * (Scalar(1) + A.row(i).cwiseAbs().dot(x.cwiseAbs()) + std::abs(b[i]));
        if(norm <= std::numeric_limits<Scalar>::epsilon() * Scalar(row_.size()) * A.row(i).norm())
            return violation <= allowance;
        G_.row(k) = row_.transpose() / norm;
        h_[k] = violation / norm;
        allowance_[k] = allowance / norm;
        return true;
    }

    // Dividing by the largest violation conditions the dual NNLS whatever the data scale. A row with more than cut times
    // that slack binds only on a move that long and would loosen the NNLS tolerance, the largest row sum, so it is left out.
    void scale_rows(Scalar scale, Scalar cut)
    {
        h_ /= scale;
        allowance_ /= scale;
        cut_ = h_.array() < -cut;
        G_.array() = cut_.replicate(1, G_.cols()).select(Scalar(0), G_.array());
        h_.array() = cut_.select(Scalar(0), h_.array());
    }

    // A u >= 0 with G^T u = 0 and h^T u > allowance^T u proves that every point violates some row beyond its allowance
    // (Farkas' lemma). u is from a failed solve, so G^T u must vanish within the NNLS's own max(m, n) epsilon |G| |u| band.
    bool certifies_inconsistency()
    {
        const auto u = nnls_x_.head(G_.rows());
        row_.noalias() = G_.transpose() * u;
        const Scalar band = Scalar(std::max(G_.rows(), G_.cols() + 1)) * std::numeric_limits<Scalar>::epsilon() * G_.norm();
        return u.minCoeff() >= Scalar(0) && row_.norm() <= band * u.norm() && h_.dot(u) > allowance_.dot(u);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedX>
    bool cut_row_violated(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, const Eigen::MatrixBase<DerivedX>& x,
                          std::int32_t m_eq, Scalar tol) const
    {
        for(Eigen::Index k = 0; k < cut_.size(); ++k)
            if(cut_[k] && A.row(m_eq + k).dot(x + move_) - b[m_eq + k] < -tol)
                return true;
        return false;
    }
};

}

#endif
