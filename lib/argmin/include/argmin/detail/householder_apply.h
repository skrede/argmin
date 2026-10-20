#ifndef HPP_GUARD_ARGMIN_DETAIL_HOUSEHOLDER_APPLY_H
#define HPP_GUARD_ARGMIN_DETAIL_HOUSEHOLDER_APPLY_H

#include <Eigen/Core>

#include <cassert>
#include <type_traits>

namespace argmin::detail
{

// MatrixBase::applyHouseholderOnTheLeft on one column with Eigen's operations in Eigen's order, so the
// result is bit-identical; Eigen materializes tau * essential in a heap temporary at dynamic size, here
// it goes into the caller's scaled storage.
template <typename Dst, typename Essential, typename Scalar, typename Scaled>
void apply_householder_on_the_left(Dst&& dst, const Essential& essential, Scalar tau, Scaled& scaled)
{
    if(dst.rows() == 1)
    {
        dst *= Scalar(1) - tau;
        return;
    }
    if(tau == Scalar(0))
        return;
    using row_type = typename Eigen::internal::plain_row_type<typename std::decay_t<Dst>::PlainObject>::type;
    assert(dst.cols() == 1);
    Scalar w;
    Eigen::Map<row_type> tmp(&w, dst.cols());
    const Eigen::Index tail = dst.rows() - 1;
    auto bottom = dst.bottomRows(tail);
    tmp.noalias() = essential.adjoint() * bottom;
    tmp += dst.row(0);
    dst.row(0) -= tau * tmp;
    scaled.head(tail) = tau * essential;
    bottom.noalias() -= scaled.head(tail) * tmp;
}

// HouseholderSequence::applyThisOnTheLeft of qr.householderQ(), or of its adjoint, truncated to length
// reflectors, on a column vector: the unblocked path Eigen takes for a single column.
template <typename QR, typename Dst, typename Scaled>
void apply_householder_q_on_the_left(const QR& qr, Eigen::Index length, bool adjoint, Dst&& dst, Scaled& scaled)
{
    static_assert(std::decay_t<Dst>::ColsAtCompileTime == 1);
    using vectors_type = std::decay_t<decltype(qr.matrixQR())>;
    const Eigen::Index rows = qr.matrixQR().rows();
    for(Eigen::Index k = 0; k < length; ++k)
    {
        const Eigen::Index actual_k = adjoint ? k : length - k - 1;
        const Eigen::Block<const vectors_type, Eigen::Dynamic, 1> essential(qr.matrixQR(), actual_k + 1, actual_k,
                                                                            rows - actual_k - 1, 1);
        const auto coeff = qr.hCoeffs().coeff(actual_k);
        apply_householder_on_the_left(dst.bottomRightCorner(rows - actual_k, dst.cols()), essential,
                                      adjoint ? coeff : Eigen::numext::conj(coeff), scaled);
    }
}

// ColPivHouseholderQR::solve in the order of Eigen's _solve_impl, overwriting the right-hand side c
// instead of copying it.
template <typename QR, typename Rhs, typename Dst, typename Scaled>
void col_piv_qr_solve_in_place(const QR& qr, Rhs&& c, Dst&& dst, Scaled& scaled)
{
    const Eigen::Index pivots = qr.nonzeroPivots();
    if(pivots == 0)
    {
        dst.setZero();
        return;
    }
    apply_householder_q_on_the_left(qr, pivots, true, c, scaled);
    qr.matrixQR().topLeftCorner(pivots, pivots).template triangularView<Eigen::Upper>()
        .solveInPlace(c.topRows(pivots));
    const auto& perm = qr.colsPermutation().indices();
    for(Eigen::Index i = 0; i < pivots; ++i)
        dst.row(perm.coeff(i)) = c.row(i);
    for(Eigen::Index i = pivots; i < qr.cols(); ++i)
        dst.row(perm.coeff(i)).setZero();
}

}

#endif
