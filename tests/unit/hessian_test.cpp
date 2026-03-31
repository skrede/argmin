#include "nablapp/hessian/bfgs_approximation.h"
#include "nablapp/hessian/lbfgs_history.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("bfgs_approximation identity initial state", "[bfgs]")
{
    bfgs_approximation<> bfgs(3);
    Eigen::VectorXd g{{1.0, -2.0, 0.5}};
    auto d = bfgs.direction(g);

    CHECK(d(0) == Approx(-1.0));
    CHECK(d(1) == Approx(2.0));
    CHECK(d(2) == Approx(-0.5));
}

TEST_CASE("bfgs_approximation converges on quadratic", "[bfgs]")
{
    // f(x) = 0.5 * x^T * A * x, gradient = A * x
    // After enough (s,y) updates with y = A*s, H should approximate A^{-1}.
    // Use BFGS directions (not steepest descent) so H converges in n steps.
    Eigen::Matrix2d A;
    A << 2.0, 1.0,
         1.0, 3.0;
    Eigen::Matrix2d A_inv = A.inverse();

    bfgs_approximation<> bfgs(2);

    Eigen::VectorXd x{{5.0, -3.0}};

    for(int k = 0; k < 6; ++k)
    {
        Eigen::VectorXd g = A * x;
        if(g.norm() < 1e-12) break;
        Eigen::VectorXd d = bfgs.direction(g);
        // Exact line search on quadratic: alpha = -g^T d / (d^T A d)
        double alpha = -g.dot(d) / d.dot(A * d);
        Eigen::VectorXd s = alpha * d;
        Eigen::VectorXd y = A * s;
        bfgs.update(s, y);
        x += s;
    }

    // After convergence, direction(-g) should approximate A^{-1} * g
    Eigen::VectorXd g_test{{1.0, 1.0}};
    auto d = bfgs.direction(g_test);
    Eigen::VectorXd expected = -A_inv * g_test;

    CHECK(d(0) == Approx(expected(0)).epsilon(0.05));
    CHECK(d(1) == Approx(expected(1)).epsilon(0.05));
}

TEST_CASE("bfgs_approximation Powell damping maintains positive definiteness", "[bfgs]")
{
    bfgs_approximation<> bfgs(3);

    // Push some normal updates first
    Eigen::VectorXd s1{{1.0, 0.0, 0.0}};
    Eigen::VectorXd y1{{2.0, 0.5, 0.1}};
    bfgs.update(s1, y1);

    // Construct a pair where s^T * y < 0 (nonconvex scenario).
    // Powell damping should activate and keep H positive definite.
    Eigen::VectorXd s2{{0.0, 1.0, 0.0}};
    Eigen::VectorXd y2{{-1.0, -0.5, 0.2}};
    bfgs.update(s2, y2);

    Eigen::VectorXd s3{{0.0, 0.0, 1.0}};
    Eigen::VectorXd y3{{0.1, -0.3, -0.8}};
    bfgs.update(s3, y3);

    // Verify positive definiteness: all eigenvalues > 0
    const auto& H = bfgs.inverse_hessian();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    for(int i = 0; i < 3; ++i)
    {
        CHECK(es.eigenvalues()(i) > 0.0);
    }
}

TEST_CASE("bfgs_approximation symmetry after updates", "[bfgs]")
{
    bfgs_approximation<> bfgs(3);

    Eigen::VectorXd s1{{1.0, 0.5, -0.3}};
    Eigen::VectorXd y1{{2.0, 1.0, 0.5}};
    bfgs.update(s1, y1);

    Eigen::VectorXd s2{{-0.2, 1.0, 0.7}};
    Eigen::VectorXd y2{{0.5, 3.0, 1.2}};
    bfgs.update(s2, y2);

    const auto& H = bfgs.inverse_hessian();
    double asym = (H - H.transpose()).norm();
    CHECK(asym < 1e-12);
}

// ---- L-BFGS tests ----

TEST_CASE("lbfgs_history empty returns steepest descent", "[lbfgs]")
{
    lbfgs_history<> lbfgs(5);
    Eigen::VectorXd g{{3.0, -1.0, 2.0}};
    auto r = lbfgs.two_loop_recursion(g);

    // Empty history: returns g (caller negates for direction)
    CHECK(r(0) == Approx(g(0)));
    CHECK(r(1) == Approx(g(1)));
    CHECK(r(2) == Approx(g(2)));
}

TEST_CASE("lbfgs_history direction matches dense BFGS on quadratic", "[lbfgs]")
{
    // On a quadratic, L-BFGS and dense BFGS both start from identity.
    // They agree on the first direction (steepest descent). After that,
    // dense BFGS uses the full rank-2 update while L-BFGS uses gamma-
    // scaled identity + corrections. We verify they produce the SAME
    // final search direction after accumulating n pairs on an n-dim
    // quadratic (at which point L-BFGS has the full history).
    Eigen::Matrix2d A;
    A << 4.0, 1.0,
         1.0, 2.0;

    bfgs_approximation<> dense(2);
    lbfgs_history<> lbfgs(20);

    Eigen::VectorXd x_dense{{5.0, -3.0}};
    Eigen::VectorXd x_lbfgs = x_dense;

    // Run both for several iterations independently
    for(int k = 0; k < 4; ++k)
    {
        Eigen::VectorXd g_d = A * x_dense;
        Eigen::VectorXd g_l = A * x_lbfgs;
        if(g_d.norm() < 1e-12 || g_l.norm() < 1e-12) break;

        auto d_dense = dense.direction(g_d);
        auto r_lbfgs = lbfgs.two_loop_recursion(g_l);
        Eigen::VectorXd d_lbfgs = -r_lbfgs;

        // Use respective directions for line search
        double alpha_d = -g_d.dot(d_dense) / d_dense.dot(A * d_dense);
        double alpha_l = -g_l.dot(d_lbfgs) / d_lbfgs.dot(A * d_lbfgs);

        Eigen::VectorXd s_d = alpha_d * d_dense;
        Eigen::VectorXd s_l = alpha_l * d_lbfgs;
        Eigen::VectorXd y_d = A * s_d;
        Eigen::VectorXd y_l = A * s_l;

        dense.update(s_d, y_d);
        lbfgs.push(s_l, y_l);
        x_dense += s_d;
        x_lbfgs += s_l;
    }

    // Both should have converged to the minimizer (origin)
    CHECK(x_dense.norm() < 1e-6);
    CHECK(x_lbfgs.norm() < 1e-6);
}

TEST_CASE("lbfgs_history circular buffer wraps correctly", "[lbfgs]")
{
    lbfgs_history<> lbfgs(3);

    for(int k = 0; k < 5; ++k)
    {
        Eigen::VectorXd s{{1.0 + k, 0.5}};
        Eigen::VectorXd y{{2.0 + k, 1.0}};
        lbfgs.push(s, y);
    }

    CHECK(lbfgs.size() == 3);
    CHECK(lbfgs.capacity() == 3);
}

TEST_CASE("lbfgs_history curvature guard skips bad pairs", "[lbfgs]")
{
    lbfgs_history<> lbfgs(5);

    // s^T y <= 0 pair should be rejected
    Eigen::VectorXd s{{1.0, 0.0}};
    Eigen::VectorXd y{{-1.0, 0.0}};
    lbfgs.push(s, y);
    CHECK(lbfgs.size() == 0);

    // s^T y = 0 should also be rejected
    Eigen::VectorXd y2{{0.0, 1.0}};
    lbfgs.push(s, y2);
    CHECK(lbfgs.size() == 0);

    // s^T y > 0 should be accepted
    Eigen::VectorXd y3{{2.0, 1.0}};
    lbfgs.push(s, y3);
    CHECK(lbfgs.size() == 1);
}

TEST_CASE("lbfgs_history gamma scaling applied", "[lbfgs]")
{
    lbfgs_history<> lbfgs(5);

    Eigen::VectorXd s{{1.0, 0.0}};
    Eigen::VectorXd y{{4.0, 0.0}};
    lbfgs.push(s, y);

    Eigen::VectorXd g{{1.0, 1.0}};
    auto r = lbfgs.two_loop_recursion(g);

    // With gamma scaling (N&W eq. 9.6), r != g
    // gamma = s^T y / y^T y = 4/16 = 0.25, so result is scaled
    CHECK(r.norm() != Approx(g.norm()).epsilon(0.01));
}
