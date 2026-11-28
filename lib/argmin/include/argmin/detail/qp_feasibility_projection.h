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
#include <optional>

namespace argmin::detail
{

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
// a rank-revealing factorization, so a dependent row is dropped rather than divided by. project expects a point for
// which violates_inequality_rows holds and returns false when the rows have no common point.
template <typename Scalar>
class qp_feasibility_projection
{
    using matrix = Eigen::MatrixX<Scalar>;
    using vector = Eigen::VectorX<Scalar>;

    // A row satisfied by more than this multiple of the largest violation is left out: it binds only on a move that
    // long, and its right-hand side would loosen the NNLS tolerance, which scales with the largest row sum.
    static constexpr Scalar far_slack{100};

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
        , nnls_ws_()
        , qr_(n, n)
    {
        nnls_ws_.ensure(n + 1, max_rows);
        nnls_ws_.r_buf.resize(n + 1);
    }

    qp_feasibility_projection() = default;

    template <typename DerivedA, typename DerivedB, typename DerivedX>
    bool project(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, Eigen::MatrixBase<DerivedX>& x,
                 std::int32_t m_eq, std::int32_t m_total, Scalar tol)
    {
        const Eigen::Index rank = factor_equalities(A, m_eq);
        const std::optional<Scalar> scale = stack_rows(A, b, x, m_eq, m_total, rank, tol);
        if(!scale)
            return false;
        if(ldp<Scalar, Eigen::Dynamic, Eigen::Dynamic>(G_, h_, move_, lambda_, nnls_A_, nnls_b_, nnls_x_, nnls_w_, nnls_ws_,
                                                         static_cast<std::int32_t>(G_.rows()),
                                                         static_cast<std::int32_t>(G_.cols())) != 1)
            return false;
        move_ *= *scale;
        if(rank > 0)
            apply_householder_q_on_the_left(qr_, rank, false, move_, scaled_);
        x += move_;
        return true;
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
    nnls_workspace<Scalar> nnls_ws_;
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

    // Every inequality slot keeps its row so the program has one shape; an absent or left-out row is a zero row,
    // 0 >= 0. The rows are normalized and the right-hand sides divided by the largest violation, which conditions the
    // dual NNLS whatever the scale of the data; the returned factor undoes that division.
    template <typename DerivedA, typename DerivedB, typename DerivedX>
    std::optional<Scalar> stack_rows(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b,
                                     const Eigen::MatrixBase<DerivedX>& x, std::int32_t m_eq, std::int32_t m_total,
                                     Eigen::Index rank, Scalar tol)
    {
        G_.setZero();
        h_.setZero();
        for(std::int32_t i = m_eq; i < m_total; ++i)
            if(std::isfinite(b[i]) && !stack_row(A, b, x, i, i - m_eq, rank, tol))
                return std::nullopt;
        const Scalar scale = h_.maxCoeff();
        h_ /= scale;
        for(Eigen::Index k = 0; k < h_.size(); ++k)
            if(h_[k] < -far_slack)
            {
                G_.row(k).setZero();
                h_[k] = Scalar(0);
            }
        return scale;
    }

    // A row whose restriction to the null space vanishes is fixed by the equality rows, so it either holds or makes
    // the rows inconsistent.
    template <typename DerivedA, typename DerivedB, typename DerivedX>
    bool stack_row(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& b, const Eigen::MatrixBase<DerivedX>& x,
                   std::int32_t i, Eigen::Index k, Eigen::Index rank, Scalar tol)
    {
        row_ = A.row(i).transpose();
        if(rank > 0)
            apply_householder_q_on_the_left(qr_, rank, true, row_, scaled_);
        row_.head(rank).setZero();
        const Scalar norm = row_.norm();
        const Scalar violation = b[i] - A.row(i).dot(x);
        if(norm <= std::numeric_limits<Scalar>::epsilon() * Scalar(row_.size()) * A.row(i).norm())
            return violation <= tol;
        G_.row(k) = row_.transpose() / norm;
        h_[k] = violation / norm;
        return true;
    }
};

}

#endif
