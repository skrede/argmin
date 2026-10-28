#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_QP_ORACLE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_QP_ORACLE_H

#include "oracle/result_oracle.h"

#include "argmin/qp/qp_types.h"

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <utility>
#include <algorithm>
#include <type_traits>

namespace argmin::oracle
{

template <typename Scalar>
struct qp_tolerances
{
    qp_tolerances(Scalar absolute_tol, Scalar relative_tol, Scalar primal_infeasibility_tol, Scalar dual_infeasibility_tol)
        : absolute(absolute_tol)
        , relative(relative_tol)
        , dual_infeasibility(dual_infeasibility_tol)
        , primal_infeasibility(primal_infeasibility_tol)
    {
    }

    Scalar absolute;
    Scalar relative;
    Scalar dual_infeasibility;
    Scalar primal_infeasibility;
};

struct qp_verdict
{
    explicit qp_verdict(std::string failing_leg)
        : ok(failing_leg.empty())
        , failing(std::move(failing_leg))
    {
    }

    bool ok;
    std::string failing;
};

template <typename Scalar>
using qp_vector = std::type_identity_t<Eigen::VectorX<Scalar>>;

template <typename Scalar>
struct qp_point
{
    qp_vector<Scalar> x;
    qp_vector<Scalar> y;
    qp_vector<Scalar> q;
    qp_vector<Scalar> l;
    qp_vector<Scalar> u;
    qp_vector<Scalar> ax;
    qp_vector<Scalar> px;
    qp_vector<Scalar> aty;
};

template <typename Derived>
typename Derived::RealScalar inf_norm(const Eigen::MatrixBase<Derived>& v)
{
    return v.template lpNorm<Eigen::Infinity>();
}

template <typename Scalar>
std::string qp_optimality_failure(const qp_point<Scalar>& p, const qp_tolerances<Scalar>& tol)
{
    const Scalar dual_tol = tol.absolute + tol.relative * std::max({inf_norm(p.px), inf_norm(p.aty), inf_norm(p.q)});
    const Scalar primal_tol = static_cast<Scalar>(safety_factor) * (tol.absolute + tol.relative * inf_norm(p.ax));
    if(!within(inf_norm(p.px + p.q + p.aty), dual_tol))
        return "stationarity";
    if(inf_norm((p.l - p.ax).cwiseMax(p.ax - p.u).cwiseMax(Scalar(0))) > primal_tol)
        return "primal_violation";
    Scalar breach = Scalar(0);
    for(Eigen::Index i = 0; i < p.y.size(); ++i)
    {
        if(p.y[i] > Scalar(0) && p.u[i] - p.ax[i] > primal_tol)
            breach = std::max(breach, p.y[i]);
        if(p.y[i] < Scalar(0) && p.ax[i] - p.l[i] > primal_tol)
            breach = std::max(breach, -p.y[i]);
    }
    return within(breach, dual_tol) ? std::string() : std::string("dual_sign_breach");
}

// Primal infeasibility certificate of Stellato et al. (2020), "OSQP: An operator splitting solver for
// quadratic programs", Section 3.4: A^T y = 0 with u^T max(y, 0) + l^T min(y, 0) < 0, checked on y.
template <typename Scalar>
std::string primal_certificate_failure(const qp_point<Scalar>& p, const qp_tolerances<Scalar>& tol)
{
    const Scalar y_norm = inf_norm(p.y);
    if(!(y_norm > Scalar(0)) || !within(inf_norm(p.aty), tol.primal_infeasibility * y_norm))
        return "certificate_range";
    Scalar support = Scalar(0);
    for(Eigen::Index i = 0; i < p.y.size(); ++i)
        support += p.y[i] > Scalar(0) ? p.u[i] * p.y[i] : (p.y[i] < Scalar(0) ? p.l[i] * p.y[i] : Scalar(0));
    return support < Scalar(0) ? std::string() : std::string("certificate_support");
}

// Dual infeasibility certificate of the same paper and section, checked on x: finalize stores the last
// iterate there, which grows along the unbounded direction, so the test is scale-free in ||x||.
template <typename Scalar>
std::string dual_certificate_failure(const qp_point<Scalar>& p, const qp_tolerances<Scalar>& tol)
{
    const Scalar x_norm = inf_norm(p.x);
    const Scalar cone_tol = static_cast<Scalar>(safety_factor) * tol.dual_infeasibility * x_norm;
    if(!(x_norm > Scalar(0)) || !within(inf_norm(p.px), tol.dual_infeasibility * x_norm))
        return "certificate_curvature";
    if(!(p.q.dot(p.x) < Scalar(0)))
        return "certificate_descent";
    for(Eigen::Index i = 0; i < p.ax.size(); ++i)
    {
        if((std::isfinite(p.u[i]) && p.ax[i] > cone_tol) || (std::isfinite(p.l[i]) && p.ax[i] < -cone_tol))
            return "certificate_recession";
    }
    return {};
}

// solved_inaccurate meets only the solver's relaxed termination test, so like max_iterations it makes
// no claim this oracle can hold the solver to.
template <typename Scalar>
std::string qp_claim_failure(qp_solve_status status, const qp_point<Scalar>& p, const qp_tolerances<Scalar>& tol)
{
    switch(status)
    {
    case qp_solve_status::solved:
        return qp_optimality_failure(p, tol);
    case qp_solve_status::primal_infeasible:
        return primal_certificate_failure(p, tol);
    case qp_solve_status::dual_infeasible:
        return dual_certificate_failure(p, tol);
    case qp_solve_status::solved_inaccurate:
    case qp_solve_status::max_iterations:
        return {};
    }
    return {};
}

template <typename PDerived, typename ADerived, typename Scalar, int N>
qp_verdict judge_qp(const Eigen::EigenBase<PDerived>& P, const qp_vector<Scalar>& q, const Eigen::EigenBase<ADerived>& A, const qp_vector<Scalar>& l,
                    const qp_vector<Scalar>& u, const qp_result<Scalar, N>& result, const qp_tolerances<Scalar>& tol)
{
    if(result.x.size() != q.size() || !result.x.allFinite())
        return qp_verdict("x");
    if(result.y.size() != l.size() || !result.y.allFinite())
        return qp_verdict("y");
    const qp_vector<Scalar> x = result.x;
    const qp_point<Scalar> p{x, result.y, q, l, u, A.derived() * x, P.derived() * x, A.derived().transpose() * result.y};
    return qp_verdict(qp_claim_failure(result.status, p, tol));
}

}

#endif
