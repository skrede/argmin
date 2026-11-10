#ifndef HPP_GUARD_ARGMIN_DETAIL_KKT_WORKING_SET_H
#define HPP_GUARD_ARGMIN_DETAIL_KKT_WORKING_SET_H

#include <Eigen/QR>
#include <Eigen/Core>

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace argmin::detail
{

// The rows a multiplier estimate may use: every equality, each inequality inside its active band and each box bound
// inside the same band. A row id indexes the stacked [J_eq; J_ineq]; a bound row is the coordinate it fixes.
template <typename Scalar, int N, int Meq, int Mineq>
struct active_rows
{
    Scalar kappa;
    const Eigen::Vector<Scalar, N>& x;
    const Eigen::Vector<Scalar, N>& lower;
    const Eigen::Vector<Scalar, N>& upper;
    const Eigen::Matrix<Scalar, Meq, N>& J_eq;
    const Eigen::Vector<Scalar, Mineq>& c_ineq;
    const Eigen::Matrix<Scalar, Mineq, N>& J_ineq;

    // Active in the row's own units, |c_i| < kappa * max(1, ||J_i||), so rescaling a constraint does not move it in or
    // out of the band; kappa defaults to sqrt(eps), the classical working-set identification band.
    bool active(Eigen::Index i) const { return std::abs(c_ineq[i]) < kappa * std::max(Scalar(1), J_ineq.row(i).norm()); }

    // The same band for a box bound, whose row e_j has unit norm.
    bool on_bound(Eigen::Index j) const { return std::abs(x[j] - lower[j]) < kappa || std::abs(upper[j] - x[j]) < kappa; }

    Eigen::Index count() const { return J_eq.rows() + J_ineq.rows(); }

    Eigen::Index next(Eigen::Index id) const
    {
        while(id < count() && id >= J_eq.rows() && !active(id - J_eq.rows()))
            ++id;
        return id;
    }

    template <typename Row>
    void copy(Eigen::Index id, Row&& row) const
    {
        if(id < J_eq.rows())
            copy_row(J_eq, id, row);
        else
            copy_row(J_ineq, id - J_eq.rows(), row);
    }

    // Eigen rejects a row block of a compile-time zero-row matrix, so that copy is never instantiated.
    template <typename Jacobian, typename Row>
    static void copy_row(const Jacobian& J, Eigen::Index i, Row& row)
    {
        if constexpr(Jacobian::RowsAtCompileTime != 0)
            row = J.row(i);
    }
};

// Caller-owned workspace for compute_kkt_multipliers_active_set, held by the SQP policy state and grow-only, so
// re-estimation at a stable active-set shape performs no allocation. The estimate never factorizes more rows than
// variables, so MaxN bounds both QR axes and every buffer, and fixed N stays in inline storage.
template <typename Scalar, int MaxN = Eigen::Dynamic>
struct kkt_multiplier_workspace
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::ColMajor, MaxN, MaxN> A_active;
    Eigen::ColPivHouseholderQR<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::ColMajor, MaxN, MaxN>> qr;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> grad_f_dyn;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> lambda_active;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> scaled;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1, Eigen::ColMajor, MaxN, 1> row_ids;
};

template <typename Rows, typename Scalar, int MaxN>
void load_rows(const Rows& rows, Eigen::Index count, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    if(ws.A_active.rows() != count || ws.A_active.cols() != n)
        ws.A_active.resize(count, n);
    for(Eigen::Index k = 0; k < count; ++k)
        rows.copy(ws.row_ids[k], ws.A_active.row(k));
    for(Eigen::Index j = 0; j < n; ++j)
        if(rows.on_bound(j))
            ws.A_active.col(j).setZero();
}

template <typename Rows, typename Gradient, typename Scalar, int MaxN>
void load_gradient(const Rows& rows, const Gradient& g, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    if(ws.grad_f_dyn.size() != g.size())
        ws.grad_f_dyn.resize(g.size());
    ws.grad_f_dyn = g;
    for(Eigen::Index j = 0; j < g.size(); ++j)
        if(rows.on_bound(j))
            ws.grad_f_dyn[j] = Scalar(0);
}

template <typename Rows, typename Scalar, int MaxN>
Eigen::Index fill_rows(const Rows& rows, Eigen::Index& id, Eigen::Index count, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    for(; count < ws.row_ids.size() && id < rows.count(); id = rows.next(id + 1))
        ws.row_ids[count++] = id;
    return count;
}

// Keeps, in order, the rows that are pivot columns within the rank of A_active^T.
template <typename Rows, typename Scalar, int MaxN>
Eigen::Index keep_pivot_rows(const Rows& rows, Eigen::Index count, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    load_rows(rows, count, n, ws);
    ws.qr.compute(ws.A_active.transpose());
    const auto pivots = ws.qr.colsPermutation().indices().head(ws.qr.rank());
    using pivot_index = typename std::decay_t<decltype(pivots)>::Scalar;
    Eigen::Index kept = 0;
    for(Eigen::Index k = 0; k < count; ++k)
        if((pivots.array() == static_cast<pivot_index>(k)).any())
            ws.row_ids[kept++] = ws.row_ids[k];
    return kept;
}

// At a degenerate vertex the active rows outnumber the variables; column-pivoted QR reduces them to a maximal linearly
// independent subset, refilling dropped slots in row order. Every such subset spans the same row space, so the
// least-squares stationarity residual is unchanged, and the multipliers are not unique there in any case.
template <typename Rows, typename Scalar, int MaxN>
Eigen::Index select_rows(const Rows& rows, Eigen::Index n, kkt_multiplier_workspace<Scalar, MaxN>& ws)
{
    if(ws.row_ids.size() != n)
        ws.row_ids.resize(n);
    Eigen::Index id = rows.next(0);
    Eigen::Index count = fill_rows(rows, id, 0, ws);
    if(id == rows.count())
        return count;
    while(true)
    {
        count = keep_pivot_rows(rows, count, n, ws);
        if(count == n || id == rows.count())
            return count;
        count = fill_rows(rows, id, count, ws);
    }
}

}

#endif
