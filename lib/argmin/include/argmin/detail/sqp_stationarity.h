#ifndef HPP_GUARD_ARGMIN_DETAIL_SQP_STATIONARITY_H
#define HPP_GUARD_ARGMIN_DETAIL_SQP_STATIONARITY_H

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/Core>

namespace argmin::detail
{

// The stationarity reports of the line-search SQP states, measured on the projected Lagrangian gradient so an
// iterate resting on an active bound reads as stationary. Where no bound is reached each value is bit-identical to
// its unprojected counterpart.

template <typename Scalar, int N, typename State>
Scalar lagrangian_gradient_norm(State& s)
{
    if(s.n_eq + s.n_ineq == 0)
        return projected_gradient_norm<Scalar, N>(s.g, s.x, s.lower, s.upper);

    Eigen::Matrix<Scalar, Eigen::Dynamic, N>& A = s.bufs.lag_A_buf;
    if(s.n_eq > 0)
        A.topRows(s.n_eq) = s.J_eq;
    if(s.n_ineq > 0)
        A.bottomRows(s.n_ineq) = s.J_ineq;
    return projected_gradient_norm<Scalar, N>(lagrangian_gradient(s.g, A, s.lambda), s.x, s.lower, s.upper);
}

template <typename Scalar, int N, typename State>
Scalar multiplier_gradient_norm(const State& s)
{
    Eigen::Vector<Scalar, N> grad_L = s.g;
    if(s.n_eq > 0)
        grad_L.noalias() -= s.J_eq.transpose() * s.bufs.kkt_lambda_eq_buf;
    if(s.n_ineq > 0)
        grad_L.noalias() -= s.J_ineq.transpose() * s.bufs.kkt_mu_ineq_buf;
    return projected_gradient_norm<Scalar, N>(grad_L, s.x, s.lower, s.upper);
}

template <typename Scalar, int N, typename State>
Scalar box_kkt_residual(const State& s)
{
    return kkt_residual<Scalar, N, Eigen::Dynamic, Eigen::Dynamic>(s.g, s.J_eq, s.J_ineq, s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf,
                                                                   s.c_eq, s.c_ineq, s.x, s.lower, s.upper);
}

// The residual above with the multipliers re-estimated at the current iterate, for a return reached away from the
// step that last estimated them (a restoration, an exhausted retry). The estimate writes only binding entries.
template <typename Scalar, int N, typename State>
Scalar estimated_box_kkt_residual(State& s)
{
    s.bufs.kkt_lambda_eq_buf.setZero(s.n_eq);
    s.bufs.kkt_mu_ineq_buf.setZero(s.n_ineq);
    if(s.n_eq + s.n_ineq > 0)
        compute_kkt_multipliers_active_set<Scalar, N, Eigen::Dynamic, Eigen::Dynamic>(s.g, s.J_eq, s.J_ineq, s.c_ineq, s.x, s.lower, s.upper, s.bufs.kkt_ws, s.bufs.kkt_lambda_eq_buf, s.bufs.kkt_mu_ineq_buf);
    return box_kkt_residual<Scalar, N>(s);
}

}

#endif
