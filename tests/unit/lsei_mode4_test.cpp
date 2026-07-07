#include "argmin/detail/lsei.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin::detail;

// ---------------------------------------------------------------------------
// LSEI mode-4 routing on the m_eq = n path.
//
// When the equality block fully determines x (m_eq = n, so the reduced free
// dimension n2 = n - m_eq is zero) the inner LSI branch -- the only place the
// inequalities G x >= h are enforced -- is skipped entirely. The determined x
// must therefore be checked against the inequalities explicitly; a violation
// returns mode 4 ("inequalities incompatible"), which the outer solver routes
// into Kraft 1988 DFVLR-FB 88-28 Section 3.4 augmented-QP recovery, instead of
// stepping to an infeasible point mislabeled mode 1 ("success").
//
// Reference: Kraft 1988 Section 3.4 (infeasible-QP recovery); Lawson & Hanson
//            1974, Chapter 23 (LSI mode codes).
// ---------------------------------------------------------------------------

namespace
{
// Drive lsei() at the n = m_eq = 1 shape. The equality C x = d fixes x = d
// (C = 1); the single inequality is G x >= h. Returns the mode code.
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
    x.setZero();
    Eigen::VectorXd lambda_eq(1), lambda_ineq(1);

    lsei_workspace<double> ws;
    ws.resize(n, m_eq, m_ineq);
    Eigen::MatrixXd nnls_A(n + 1, m_ineq);
    Eigen::VectorXd nnls_b(n + 1), nnls_x(m_ineq), nnls_w(m_ineq);
    nnls_workspace<double> nnls_ws;

    int mode = lsei<double, Eigen::Dynamic>(
        C, d_vec, E, f, G, h, x, lambda_eq, lambda_ineq, ws,
        nnls_A, nnls_b, nnls_x, nnls_w, nnls_ws, n, m_eq, m_ineq);
    x_out = x[0];
    return mode;
}
}  // namespace

TEST_CASE("lsei routes an infeasible determined point to mode 4", "[lsei][mode4]")
{
    // C = [1], d = [0], G = [1], h = [1]. The equality fixes x = 0, so the
    // inequality residual is G x - h = 1*0 - 1 = -1 < 0. The entire derivation
    // is that scalar arithmetic: the determined point is infeasible and must
    // route to Kraft Section 3.4 recovery.
    //
    // Pre-fix: the n2 == 0 branch returned mode 1 with x = 0 unchecked.
    // Post-fix: the explicit feasibility check returns mode 4.
    double x = 0.0;
    int mode = run_lsei_1d(/*d=*/0.0, /*h=*/1.0, x);
    CHECK(mode == 4);
    CHECK(x == Approx(0.0).margin(1e-12));  // x still determined by the equality
}

TEST_CASE("lsei accepts a feasible determined point as mode 1", "[lsei][mode4]")
{
    // Companion case guarding against over-rejection: C = [1], d = [2], so the
    // equality fixes x = 2, and G x - h = 1*2 - 1 = 1 >= 0 is feasible. The
    // determined point satisfies the inequality, so mode 1 is correct.
    double x = 0.0;
    int mode = run_lsei_1d(/*d=*/2.0, /*h=*/1.0, x);
    CHECK(mode == 1);
    CHECK(x == Approx(2.0).margin(1e-12));
}
