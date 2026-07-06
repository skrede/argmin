// Delta-property invariant for the BMAT/ZMAT factored interpolation
// system. The Lagrange cardinal basis must satisfy L_k(x_j) = delta_kj
// at the interpolation nodes: evaluating basis k at node j returns 1 when
// k == j and 0 otherwise. The maximum deviation
//     max_{k,j} |L_k(x_j) - delta_kj|
// must sit at machine-epsilon scale for a correctly assembled system.
//
// The suite covers both a "swap" objective (a negative-axis perturbation
// is better, which triggers the bootstrap's conditional point swap) and a
// "no-swap" objective, and re-checks the invariant after a rank-2 point
// replacement.
//
// Reference: Powell, M. J. D. (2009), "The BOBYQA algorithm for bound
//            constrained optimization without derivatives",
//            DAMTP 2009/NA06, Sections 2-4.

#include "argmin/detail/interpolation_system.h"
#include "argmin/detail/trust_region.h"
#include "argmin/detail/bound_projection.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <cmath>

using argmin::detail::interpolation_system;
using argmin::detail::bootstrap_interpolation_system;
using argmin::detail::compute_lagrange_at;
using argmin::detail::compute_vlag_beta;
using argmin::detail::compute_denom;
using argmin::detail::update_bmat_zmat;
using argmin::detail::update_model_on_replacement;

namespace
{

// max_{k,j} |L_k(x_j) - delta_kj| over all interpolation nodes.
double delta_property_violation(const interpolation_system<double>& sys)
{
    const int m = sys.m_points;
    double worst = 0.0;
    for(int j = 0; j < m; ++j)
    {
        // Node j relative to xbase is exactly column j of xpt.
        Eigen::VectorXd x_j = sys.xpt.col(j).head(sys.xbase.size());
        Eigen::VectorXd lag = compute_lagrange_at(sys, x_j);
        for(int k = 0; k < m; ++k)
        {
            const double target = (k == j) ? 1.0 : 0.0;
            worst = std::max(worst, std::abs(lag[k] - target));
        }
    }
    return worst;
}

}

// A near-machine-epsilon bar: a correctly assembled cardinal basis
// reproduces the Kronecker delta to full double precision.
constexpr double delta_tol = 1e-12;

TEST_CASE("interpolation delta-property holds for the no-swap objective",
          "[interpolation_delta_property]")
{
    const int n = 2;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    // f(x) = sum_i (x_i - 10)^2: moving along +axis is always better, so
    // the positive perturbation point wins and the bootstrap swap never
    // fires. This isolates the assembly from the swap-order defect.
    auto eval = [](const Eigen::VectorXd& x) {
        return (x.array() - 10.0).square().sum();
    };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);

    CHECK(delta_property_violation(sys) < delta_tol);
}

TEST_CASE("interpolation delta-property survives a rank-2 update",
          "[interpolation_delta_property]")
{
    const int n = 2;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    // No-swap objective, so the bootstrap itself is clean; the invariant
    // is re-checked after the rank-2 replacement to exercise the update
    // path in isolation.
    auto eval = [](const Eigen::VectorXd& x) {
        return (x.array() - 10.0).square().sum();
    };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);

    REQUIRE(delta_property_violation(sys) < delta_tol);

    // Replace a non-optimal interpolation point with a fresh node at
    // xopt + d. Pick knew != kopt so the incumbent best is preserved.
    const int m = sys.m_points;
    int knew = (sys.kopt == 0) ? 1 : 0;

    Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
    d[0] = 0.5 * rhobeg;
    d[1] = -0.3 * rhobeg;

    auto [vlag, beta] = compute_vlag_beta(sys, d);

    const int nptm = m - n - 1;
    double alpha = 0.0;
    for(int jj = 0; jj < nptm; ++jj)
        alpha += sys.zmat(knew, jj) * sys.zmat(knew, jj);
    double denom = compute_denom(vlag[knew], alpha, beta);
    REQUIRE(std::abs(denom) > 1e-20);

    Eigen::VectorXd new_x = sys.xopt + d;             // relative to xbase
    Eigen::VectorXd new_x_abs = sys.xbase + new_x;    // absolute coords
    double new_f = eval(new_x_abs);

    update_bmat_zmat(sys, vlag, beta, denom, knew);
    update_model_on_replacement(sys, new_x, new_f, knew, d);

    CHECK(delta_property_violation(sys) < delta_tol);
}

// The geometry (ALTMOV) branch must pick the interpolation point FARTHEST
// from xopt, not index 0. Because the cardinal basis satisfies
// L_k(xopt) == delta_{k,kopt}, the old |L_k(xopt)| score tied every candidate
// at zero and always returned index 0; this checks the distance criterion and
// that an ALTMOV replacement keeps the delta-property at machine-epsilon.
TEST_CASE("bobyqa geometry selects the farthest point and preserves the delta-property",
          "[interpolation_delta_property]")
{
    using argmin::detail::altmov_geometry_step;
    using argmin::detail::project;

    const int n = 2;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    auto eval = [](const Eigen::VectorXd& x) {
        return (x.array() - 10.0).square().sum();
    };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);
    REQUIRE(delta_property_violation(sys) < delta_tol);

    // Push one non-optimal node far from xopt via a valid rank-2 update, so
    // there is an unambiguous farthest point.
    const int m = sys.m_points;
    int kfar = (sys.kopt == 0) ? 1 : 0;
    // Displace AWAY from the descent direction so the far node stays
    // non-optimal (it must not become the new kopt, or it would be excluded
    // from the geometry candidate set).
    Eigen::VectorXd d_far = Eigen::VectorXd::Zero(n);
    d_far[0] = -3.0 * rhobeg;
    d_far[1] = -2.0 * rhobeg;
    {
        auto [vlag, beta] = compute_vlag_beta(sys, d_far);
        const int nptm = m - n - 1;
        double alpha = 0.0;
        for(int jj = 0; jj < nptm; ++jj)
            alpha += sys.zmat(kfar, jj) * sys.zmat(kfar, jj);
        double denom = compute_denom(vlag[kfar], alpha, beta);
        REQUIRE(std::abs(denom) > 1e-20);
        Eigen::VectorXd new_x = sys.xopt + d_far;
        double new_f = eval((sys.xbase + new_x).eval());
        update_bmat_zmat(sys, vlag, beta, denom, kfar);
        update_model_on_replacement(sys, new_x, new_f, kfar, d_far);
    }
    REQUIRE(delta_property_violation(sys) < delta_tol);

    // Farthest-point selection (the exact criterion the policy applies).
    int knew_geo = -1;
    double dist_geo_sq = 0.0;
    for(int i = 0; i < m; ++i)
    {
        if(i == sys.kopt) continue;
        double dsq = (sys.xpt.col(i).head(n) - sys.xopt).squaredNorm();
        if(dsq > dist_geo_sq)
        {
            dist_geo_sq = dsq;
            knew_geo = i;
        }
    }
    // The displaced node is the farthest; the vacuous |L_k(xopt)| rule would
    // instead have returned index 0.
    CHECK(knew_geo == kfar);
    CHECK(knew_geo != -1);

    // Run the ALTMOV geometry step on the farthest point and re-check the
    // delta-property after the geometry replacement.
    const double dist_geo = std::sqrt(dist_geo_sq);
    const double delta = rhobeg;
    const double adelt = std::max(std::min(0.1 * dist_geo, 0.5 * delta), rhobeg);
    Eigen::VectorXd geo_shifted = altmov_geometry_step<double, Eigen::Dynamic>(
        sys, knew_geo, adelt, (lower - sys.xbase).eval(), (upper - sys.xbase).eval());
    Eigen::VectorXd geo_abs = project((sys.xbase + geo_shifted).eval(), lower, upper);
    geo_shifted = geo_abs - sys.xbase;
    Eigen::VectorXd d_geo = geo_shifted - sys.xopt;

    auto [vlag_g, beta_g] = compute_vlag_beta(sys, d_geo);
    const int nptm = m - n - 1;
    double alpha_g = 0.0;
    for(int jj = 0; jj < nptm; ++jj)
        alpha_g += sys.zmat(knew_geo, jj) * sys.zmat(knew_geo, jj);
    double denom_g = compute_denom(vlag_g[knew_geo], alpha_g, beta_g);
    REQUIRE(std::abs(denom_g) > 1e-20);
    double geo_f = eval((sys.xbase + geo_shifted).eval());
    update_bmat_zmat(sys, vlag_g, beta_g, denom_g, knew_geo);
    update_model_on_replacement(sys, geo_shifted, geo_f, knew_geo, d_geo);

    CHECK(delta_property_violation(sys) < delta_tol);
}

// For the swap objective the negative-axis point is better on every axis,
// so the bootstrap swaps fval/xpt on every axis. The swap now happens BEFORE
// the BMAT/ZMAT assembly, so the factored inverse represents the final point
// positions and the cardinal basis keeps the delta-property to full machine
// precision: max_{k,j} |L_k(x_j) - delta_kj| is 0 here (it was 2.0 when the
// swap ran after the BMAT/ZMAT writes).
TEST_CASE("interpolation delta-property holds for the swap-objective bootstrap",
          "[interpolation_delta_property]")
{
    const int n = 2;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    // f(x) = sum_i x_i is increasing along every axis, so the negative
    // perturbation is better on every axis and the swap fires everywhere.
    auto eval = [](const Eigen::VectorXd& x) { return x.sum(); };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);

    CHECK(delta_property_violation(sys) < delta_tol);
}
