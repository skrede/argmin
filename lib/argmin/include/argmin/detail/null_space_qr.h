#ifndef HPP_GUARD_ARGMIN_DETAIL_NULL_SPACE_QR_H
#define HPP_GUARD_ARGMIN_DETAIL_NULL_SPACE_QR_H

#include "argmin/detail/householder_apply.h"

#include <Eigen/QR>
#include <Eigen/Core>

namespace argmin::detail
{

template <typename Scalar>
struct null_space_qr_workspace
{
    Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr;
    Eigen::VectorX<Scalar> projected;
    Eigen::VectorX<Scalar> scaled;

    void resize(Eigen::Index n, Eigen::Index m)
    {
        qr = Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>>(n, m);
        projected.resize(n);
        scaled.resize(n);
    }
};

// Q1 = H_0 ... H_{r-1} restricted to its first r columns, since every later reflector fixes e_0 .. e_{r-1}.
template <typename Scalar, int N>
void project_off_range_qr(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, N>>& A, Eigen::Ref<Eigen::Vector<Scalar, N>> x, null_space_qr_workspace<Scalar>& ws)
{
    ws.qr.compute(A.transpose());
    const Eigen::Index r = ws.qr.rank();
    if(r == 0)
        return;
    ws.projected = x;
    apply_householder_q_on_the_left(ws.qr, r, true, ws.projected, ws.scaled);
    ws.projected.tail(ws.projected.size() - r).setZero();
    apply_householder_q_on_the_left(ws.qr, r, false, ws.projected, ws.scaled);
    x -= ws.projected;
}

}

#endif
