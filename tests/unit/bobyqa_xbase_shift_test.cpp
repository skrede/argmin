// Origin-shift invariance for the BMAT/ZMAT factored interpolation system.
//
// shift_xbase re-expresses the whole system about xbase + xopt so that xopt
// becomes the origin. Because the best absolute point does not move, the
// shift is an EXACT re-parameterization: the quadratic model Q evaluated in
// absolute coordinates and every Lagrange function L_k are numerically
// unchanged across the shift, and the cardinal-basis delta-property survives.
//
// These pins catch a partial port (e.g. missing the ZMAT-dependent BMAT
// terms, the symmetric bottom block, or the HQ revision), which would leave
// the model unchanged for the trivial probe but corrupt the factored inverse
// and break poisedness.
//
// Reference: Powell, M. J. D. (2009), "The BOBYQA algorithm for bound
//            constrained optimization without derivatives",
//            DAMTP 2009/NA06, Section 5 (origin shift).

#include "argmin/detail/interpolation_system.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>
#include <cmath>

using argmin::detail::interpolation_system;
using argmin::detail::bootstrap_interpolation_system;
using argmin::detail::compute_lagrange_at;
using argmin::detail::compute_vlag_beta;
using argmin::detail::compute_denom;
using argmin::detail::update_bmat_zmat;
using argmin::detail::update_model_on_replacement;
using argmin::detail::evaluate_interpolation_model;
using argmin::detail::shift_xbase;

namespace
{

// max_{k,j} |L_k(x_j) - delta_kj| over all interpolation nodes.
double delta_property_violation(const interpolation_system<double>& sys)
{
    const int m = sys.m_points;
    double worst = 0.0;
    for(int j = 0; j < m; ++j)
    {
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

// Move the incumbent away from xbase via a valid rank-2 update so xopt has a
// substantive, multi-component magnitude and the shift is a non-trivial test.
void push_incumbent(interpolation_system<double>& sys,
                    const Eigen::VectorXd& d,
                    const std::function<double(const Eigen::VectorXd&)>& eval)
{
    const int n = sys.xbase.size();
    const int m = sys.m_points;
    const int nptm = m - n - 1;
    int knew = (sys.kopt == 0) ? 1 : 0;

    auto [vlag, beta] = compute_vlag_beta(sys, d);
    double alpha = 0.0;
    for(int jj = 0; jj < nptm; ++jj)
        alpha += sys.zmat(knew, jj) * sys.zmat(knew, jj);
    double denom = compute_denom(vlag[knew], alpha, beta);
    REQUIRE(std::abs(denom) > 1e-20);

    Eigen::VectorXd new_x = sys.xopt + d;
    double new_f = eval((sys.xbase + new_x).eval());
    update_bmat_zmat(sys, vlag, beta, denom, knew);
    update_model_on_replacement(sys, new_x, new_f, knew, d);
}

constexpr double delta_tol = 1e-12;

}

TEST_CASE("bobyqa origin shift leaves the model and Lagrange values invariant",
          "[bobyqa][xbase_shift]")
{
    const int n = 3;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    // A curved (non-separable) quadratic so HQ carries off-diagonal mass and
    // the shift's HQ revision is exercised, not just the diagonal.
    auto eval = [](const Eigen::VectorXd& x) {
        double f = 0.0;
        for(int i = 0; i < x.size(); ++i)
            f += (x[i] - 10.0) * (x[i] - 10.0);
        f += 0.5 * (x[0] - 10.0) * (x[1] - 10.0);
        return f;
    };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);
    REQUIRE(delta_property_violation(sys) < delta_tol);

    // Drive the incumbent away from xbase so xopt is a genuine multi-component
    // displacement (toward the descent direction, so it becomes the new kopt).
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
    d[0] = 0.7 * rhobeg;
    d[1] = 0.5 * rhobeg;
    d[2] = 0.3 * rhobeg;
    push_incumbent(sys, d, eval);
    REQUIRE(delta_property_violation(sys) < delta_tol);

    const Eigen::VectorXd xopt_before = sys.xopt;
    const Eigen::VectorXd xbase_before = sys.xbase;
    REQUIRE(xopt_before.norm() > 1e-3);   // the shift is substantive

    // Probe displacements from the optimum. Q(xopt + s) in absolute
    // coordinates and L_k at the same absolute point must be unchanged by the
    // shift. Because the optimum's absolute position is fixed, the same
    // displacement s reproduces the same absolute point before and after.
    std::vector<Eigen::VectorXd> probes;
    {
        Eigen::VectorXd s(n); s << 0.03, -0.05, 0.02; probes.push_back(s);
        Eigen::VectorXd s2(n); s2 << -0.12, 0.08, -0.04; probes.push_back(s2);
        Eigen::VectorXd s3(n); s3 << 0.2, 0.15, 0.11; probes.push_back(s3);
        probes.push_back(Eigen::VectorXd::Zero(n));  // the optimum itself
    }

    std::vector<double> q_before;
    std::vector<Eigen::VectorXd> lag_before;
    for(const auto& s : probes)
    {
        q_before.push_back(evaluate_interpolation_model(sys, s));
        lag_before.push_back(compute_lagrange_at(sys, (sys.xopt + s).eval()));
    }

    shift_xbase(sys);

    // Post-shift state.
    CHECK(sys.xopt.norm() == 0.0);
    CHECK((sys.xbase - (xbase_before + xopt_before)).cwiseAbs().maxCoeff() < 1e-15);
    CHECK(delta_property_violation(sys) < delta_tol);

    // Invariance: same displacement s, same absolute point (xopt is now 0).
    for(std::size_t i = 0; i < probes.size(); ++i)
    {
        const Eigen::VectorXd& s = probes[i];
        double q_after = evaluate_interpolation_model(sys, s);
        Eigen::VectorXd lag_after = compute_lagrange_at(sys, s);
        CHECK(std::abs(q_after - q_before[i]) < 1e-10);
        CHECK((lag_after - lag_before[i]).cwiseAbs().maxCoeff() < 1e-10);
    }
}

TEST_CASE("bobyqa origin shift on a bootstrapped 2D system preserves poisedness",
          "[bobyqa][xbase_shift]")
{
    const int n = 2;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
    const double rhobeg = 0.1;
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, inf);

    // Swap objective (negative-axis point is better on every axis), so the
    // bootstrap swap fires and xopt lands on a perturbation point.
    auto eval = [](const Eigen::VectorXd& x) { return x.sum(); };

    auto sys = bootstrap_interpolation_system<double, Eigen::Dynamic>(
        x0, rhobeg, lower, upper, eval);
    REQUIRE(delta_property_violation(sys) < delta_tol);
    REQUIRE(sys.xopt.norm() > 0.0);

    const Eigen::VectorXd xopt_before = sys.xopt;
    const Eigen::VectorXd xbase_before = sys.xbase;

    // Record Q and L_k at a probe before the shift.
    Eigen::VectorXd s(n); s << 0.04, -0.03;
    double q_before = evaluate_interpolation_model(sys, s);
    Eigen::VectorXd lag_before = compute_lagrange_at(sys, (sys.xopt + s).eval());

    shift_xbase(sys);

    CHECK(sys.xopt.norm() == 0.0);
    CHECK((sys.xbase - (xbase_before + xopt_before)).cwiseAbs().maxCoeff() < 1e-15);
    CHECK(delta_property_violation(sys) < delta_tol);

    double q_after = evaluate_interpolation_model(sys, s);
    Eigen::VectorXd lag_after = compute_lagrange_at(sys, s);
    CHECK(std::abs(q_after - q_before) < 1e-10);
    CHECK((lag_after - lag_before).cwiseAbs().maxCoeff() < 1e-10);
}
