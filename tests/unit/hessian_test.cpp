#include "nablapp/hessian/bfgs_approximation.h"

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
    Eigen::Matrix2d A;
    A << 2.0, 1.0,
         1.0, 3.0;
    Eigen::Matrix2d A_inv = A.inverse();

    bfgs_approximation<> bfgs(2);

    // Simulate steepest descent steps on the quadratic
    Eigen::VectorXd x{{5.0, -3.0}};
    constexpr double alpha = 0.1;

    for(int k = 0; k < 10; ++k)
    {
        Eigen::VectorXd g = A * x;
        Eigen::VectorXd s = -alpha * g;
        Eigen::VectorXd y = A * s;
        bfgs.update(s, y);
        x += s;
    }

    // After convergence, direction(-g) should approximate A^{-1} * g
    Eigen::VectorXd g_test{{1.0, 1.0}};
    auto d = bfgs.direction(g_test);
    Eigen::VectorXd expected = -A_inv * g_test;

    CHECK(d(0) == Approx(expected(0)).epsilon(0.01));
    CHECK(d(1) == Approx(expected(1)).epsilon(0.01));
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
