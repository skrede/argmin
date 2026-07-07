#include "argmin/detail/lagrangian.h"
#include "argmin/detail/lsei.h"
#include "argmin/detail/ldp.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using Catch::Approx;

// ---------------------------------------------------------------------------
// Scale-varied substrate witnesses.
//
// Each case constructs a problem whose SCALE (not conditioning) pushes a
// substrate tolerance out of its intended regime. On the pre-relative
// absolute tolerances every assertion below fails; on the scale-relative
// forms they pass. The cases are hand-derived so the pre/post behavior is a
// direct consequence of the tolerance form, not of any solver heuristic.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// lagrangian active-set detection for a constraint scaled to non-unit units.
//
// A single inequality whose gradient is scaled by S = 1e6: J_ineq = [S, 0],
// with the constraint value c_ineq = S * 1e-9 = 1e-3 (a genuinely active
// constraint whose residual in its own natural units is 1e-9). grad_f =
// [2 S, 0] = J_ineq^T * mu with mu = 2. The absolute active-set band
// (|c_ineq| < 1e-8) misses the constraint because |c_ineq| = 1e-3 >> 1e-8, so
// it is dropped from the working set and the recovered multiplier is 0. The
// per-row-relative band |c_ineq[i]| < kappa * max(1, ||J_ineq.row(i)||) scales
// with the constraint gradient magnitude: kappa * 1e6 ~ 1.5e-2 > 1e-3, so the
// constraint is correctly identified as active and mu ~ 2 is recovered.
//
// The problem-relative band kappa * max(1, ||c_ineq||_inf) does NOT green this
// case (||c_ineq||_inf = 1e-3, so the band collapses to kappa < 1e-3); the
// per-row form is required, matching the more principled activity-in-own-units
// reference.
TEST_CASE("substrate scale witness: lagrangian detects a scaled active constraint",
          "[substrate][scale][lagrangian]")
{
    const double S = 1e6;
    Eigen::Vector<double, 2> grad_f;
    grad_f << 2.0 * S, 0.0;

    Eigen::Matrix<double, 0, 2> J_eq;   // no equalities
    Eigen::Matrix<double, 1, 2> J_ineq;
    J_ineq << S, 0.0;
    Eigen::Vector<double, 1> c_ineq;
    c_ineq << S * 1e-9;                  // = 1e-3

    Eigen::Vector<double, Eigen::Dynamic> mu =
        argmin::detail::estimate_multipliers_active_set<double, 2, 0, 1>(
            grad_f, J_eq, J_ineq, c_ineq);

    REQUIRE(mu.size() == 1);
    INFO("recovered mu = " << mu[0]);
    CHECK(mu[0] == Approx(2.0).margin(1e-6));
}

// ---------------------------------------------------------------------------
// LDP feasibility verdict for a well-conditioned, large-||x|| problem.
//
// min ||x||^2 s.t. x >= 1e8 (G = [1], h = [1e8]). The optimum x* = 1e8 is
// trivially feasible, but the LDP dual recovery fac = 1 - h^T u collapses to
// ~1.1e-16 (< eps) because ||x|| is large. The absolute guard
// (rnorm_sq <= eps || fac <= eps) therefore misreads this feasible problem as
// "incompatible" (mode 4). The Farkas-relative form tests the primal-recovery
// residual ||G^T u|| against its natural scale ||G||_inf * ||u|| (which is
// O(1) here, not ~0) and only rejects on fac <= 0 (the true empty-intersection
// certificate), so the feasible large-norm solution is recovered at mode 1.
//
// At h = 1e8, fac = 1.1e-16 > 0 so the solution is recoverable (x ~ 9e7,
// order-correct); at h >= 1e9, fac underflows to exactly 0 and mode 4 remains
// the honest verdict (the recovery would divide by zero).
TEST_CASE("substrate scale witness: LDP accepts a large-norm feasible solution",
          "[substrate][scale][ldp]")
{
    const int n = 1, m = 1;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G(1, 1);
    G << 1.0;
    Eigen::Vector<double, Eigen::Dynamic> h(1);
    h << 1e8;
    Eigen::Vector<double, Eigen::Dynamic> x(1);
    Eigen::Vector<double, Eigen::Dynamic> lambda(1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nnls_A;
    Eigen::Vector<double, Eigen::Dynamic> nnls_b, nnls_x_vec, nnls_w;

    argmin::detail::nnls_workspace<double> nnls_ws;
    const int mode = argmin::detail::ldp<double, Eigen::Dynamic, Eigen::Dynamic>(
        G, h, x, lambda, nnls_A, nnls_b, nnls_x_vec, nnls_w, nnls_ws, m, n);

    INFO("mode = " << mode << "  x = " << x[0]);
    CHECK(mode == 1);
    CHECK(x[0] > 5e7);
}

// ---------------------------------------------------------------------------
// LSEI mode-4 feasibility band at a large-scale determined system.
//
// n = m_eq = 1 so x is fully determined by the equality C x = d (x = 1e10);
// the reduced free dimension is zero and the inequality G x >= h is checked
// only by the explicit mode-4 feasibility band. With h = 1e10 + 1 the residual
// G x - h = -1 is an absolute violation of 1 but a RELATIVE violation of
// 1e-10 (well inside rounding for a 1e10-scale problem). The absolute band
// feas_tol = eps*n*10 ~ 2.2e-15 rejects it (mode 4); the relative band
// feas_tol = tol * max(1, ||h||_inf) accepts it (mode 1) because
// tol * 1e10 >> 1 for the swept tol.
namespace
{
int run_lsei_1d(double d_val, double h_val, double& x_out)
{
    const int n = 1, m_eq = 1, m_ineq = 1;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C(1, 1);
    C << 1.0;
    Eigen::VectorXd d_vec(1);
    d_vec << d_val;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> E(1, 1);
    E << 1.0;
    Eigen::VectorXd f(1);
    f << 0.0;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G(1, 1);
    G << 1.0;
    Eigen::VectorXd h(1);
    h << h_val;
    Eigen::VectorXd x(1);
    Eigen::VectorXd lambda_eq(1), lambda_ineq(1);
    argmin::detail::lsei_workspace<double> ws;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nnls_A;
    Eigen::VectorXd nnls_b, nnls_x_vec, nnls_w;

    argmin::detail::nnls_workspace<double> nnls_ws;
    const int mode = argmin::detail::lsei<double, Eigen::Dynamic>(
        C, d_vec, E, f, G, h, x, lambda_eq, lambda_ineq, ws,
        nnls_A, nnls_b, nnls_x_vec, nnls_w, nnls_ws, n, m_eq, m_ineq);
    x_out = x[0];
    return mode;
}
}

TEST_CASE("substrate scale witness: LSEI mode-4 band accepts a negligible relative violation",
          "[substrate][scale][lsei]")
{
    double x_out = 0.0;
    const int mode = run_lsei_1d(/*d=*/1e10, /*h=*/1e10 + 1.0, x_out);
    INFO("mode = " << mode << "  x = " << x_out);
    CHECK(mode == 1);
    CHECK(x_out == Approx(1e10).epsilon(1e-9));
}
