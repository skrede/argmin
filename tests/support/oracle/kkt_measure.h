#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_KKT_MEASURE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_KKT_MEASURE_H

#include "oracle/kkt_point.h"

#include <Eigen/QR>
#include <Eigen/Core>

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace argmin::oracle
{

template <typename Scalar>
struct tolerances
{
    tolerances(Scalar stationarity_tol, Scalar feasibility_tol, Scalar ftol_stationarity_tol)
        : feasibility(feasibility_tol)
        , stationarity(stationarity_tol)
        , ftol_stationarity(ftol_stationarity_tol)
    {
    }

    Scalar feasibility;
    Scalar stationarity;
    Scalar ftol_stationarity;
};

template <typename Scalar>
struct kkt_legs
{
    kkt_legs()
        : stationarity(0)
        , bound_violation(0)
        , complementarity(0)
        , dual_sign_breach(0)
        , equality_violation(0)
        , inequality_violation(0)
    {
    }

    Scalar stationarity;
    Scalar bound_violation;
    Scalar complementarity;
    Scalar dual_sign_breach;
    Scalar equality_violation;
    Scalar inequality_violation;
};

// A variable within the activity tolerance of both bounds is effectively fixed, so its multiplier
// has no sign condition.
enum class active_kind : std::uint8_t
{
    equality,
    inequality,
    lower,
    upper,
    fixed
};

template <typename Scalar>
struct active_row
{
    active_kind kind;
    Eigen::Index index;
    Scalar slack;
};

template <typename Scalar>
std::vector<active_row<Scalar>> near_active(const kkt_point<Scalar>& pt, Scalar tol)
{
    std::vector<active_row<Scalar>> rows;
    for(Eigen::Index i = 0; i < pt.c.size(); ++i)
    {
        if(i < pt.n_eq)
            rows.push_back({active_kind::equality, i, Scalar(0)});
        else if(pt.c[i] <= tol)
            rows.push_back({active_kind::inequality, i, pt.c[i]});
    }
    for(Eigen::Index j = 0; j < pt.x.size(); ++j)
    {
        const Scalar to_lower = pt.x[j] - pt.lower[j];
        const Scalar to_upper = pt.upper[j] - pt.x[j];
        if(to_lower <= tol && to_upper <= tol)
            rows.push_back({active_kind::fixed, j, Scalar(0)});
        else if(to_lower <= tol)
            rows.push_back({active_kind::lower, j, to_lower});
        else if(to_upper <= tol)
            rows.push_back({active_kind::upper, j, to_upper});
    }
    return rows;
}

template <typename Scalar>
bool is_constraint_row(const active_row<Scalar>& row)
{
    return row.kind == active_kind::equality || row.kind == active_kind::inequality;
}

// Multipliers are the least-squares fit of grad f onto the active gradients, with the Lagrangian
// L = f - lambda^T c_eq - mu^T c_ineq - sigma^T (x - l, u - x), so mu and sigma must be nonnegative.
template <typename Scalar>
Eigen::VectorX<Scalar> recover_multipliers(const kkt_point<Scalar>& pt, const std::vector<active_row<Scalar>>& rows)
{
    Eigen::MatrixX<Scalar> A = Eigen::MatrixX<Scalar>::Zero(pt.x.size(), static_cast<Eigen::Index>(rows.size()));
    for(std::size_t k = 0; k < rows.size(); ++k)
    {
        const Eigen::Index col = static_cast<Eigen::Index>(k);
        if(is_constraint_row(rows[k]))
            A.col(col) = pt.jacobian.row(rows[k].index).transpose();
        else
            A(rows[k].index, col) = rows[k].kind == active_kind::upper ? Scalar(-1) : Scalar(1);
    }
    if(rows.empty())
        return Eigen::VectorX<Scalar>(0);
    return A.colPivHouseholderQr().solve(pt.g);
}

// Projected-gradient stationarity (Nocedal and Wright 2e, Section 16.7) of the Lagrangian over the
// general constraints; the box is handled by the projection, not by the bound multipliers.
template <typename Scalar>
Scalar projected_stationarity(const kkt_point<Scalar>& pt, const std::vector<active_row<Scalar>>& rows, const Eigen::VectorX<Scalar>& y)
{
    Eigen::VectorX<Scalar> grad_l = pt.g;
    for(std::size_t k = 0; k < rows.size(); ++k)
    {
        if(is_constraint_row(rows[k]))
            grad_l -= y[static_cast<Eigen::Index>(k)] * pt.jacobian.row(rows[k].index).transpose();
    }
    const Eigen::VectorX<Scalar> projected = (pt.x - grad_l).cwiseMax(pt.lower).cwiseMin(pt.upper);
    return (projected - pt.x).template lpNorm<Eigen::Infinity>();
}

template <typename Scalar>
void measure_signs(const std::vector<active_row<Scalar>>& rows, const Eigen::VectorX<Scalar>& y, kkt_legs<Scalar>& legs)
{
    for(std::size_t k = 0; k < rows.size(); ++k)
    {
        if(rows[k].kind == active_kind::equality || rows[k].kind == active_kind::fixed)
            continue;
        const Scalar multiplier = y[static_cast<Eigen::Index>(k)];
        legs.complementarity = std::max(legs.complementarity, std::abs(multiplier * rows[k].slack));
        legs.dual_sign_breach = std::max(legs.dual_sign_breach, std::max(-multiplier, Scalar(0)));
    }
}

template <typename Scalar>
void measure_feasibility(const kkt_point<Scalar>& pt, kkt_legs<Scalar>& legs)
{
    const Eigen::Index n_ineq = pt.c.size() - pt.n_eq;
    legs.equality_violation = pt.c.head(pt.n_eq).template lpNorm<Eigen::Infinity>();
    legs.inequality_violation = (-pt.c.tail(n_ineq)).cwiseMax(Scalar(0)).template lpNorm<Eigen::Infinity>();
    legs.bound_violation = (pt.lower - pt.x).cwiseMax(pt.x - pt.upper).cwiseMax(Scalar(0)).template lpNorm<Eigen::Infinity>();
}

template <typename Problem, typename Scalar, int N>
kkt_legs<Scalar> measure_kkt(const Problem& problem, const Eigen::Vector<Scalar, N>& x, Scalar active_tol)
{
    const kkt_point<Scalar> pt = evaluate_point(problem, x);
    const std::vector<active_row<Scalar>> rows = near_active(pt, active_tol);
    const Eigen::VectorX<Scalar> y = recover_multipliers(pt, rows);
    kkt_legs<Scalar> legs;
    measure_feasibility(pt, legs);
    legs.stationarity = projected_stationarity(pt, rows, y);
    measure_signs(rows, y, legs);
    return legs;
}

}

#endif
