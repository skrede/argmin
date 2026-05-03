#include "argmin/derivative/finite_difference.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/booth.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

TEST_CASE("fd_gradient on Rosenbrock", "[finite_difference]")
{
    rosenbrock fn{.a = 1.0, .b = 5.0, .n = 2};
    Eigen::VectorXd g_fd(2);
    Eigen::VectorXd g_analytical(2);

    SECTION("at origin [0, 0]")
    {
        Eigen::VectorXd x{{0.0, 0.0}};
        fd_gradient(fn, x, g_fd);
        fn.gradient(x, g_analytical);

        for(int i = 0; i < 2; ++i)
        {
            CHECK(g_fd(i) == Approx(g_analytical(i)).epsilon(1e-6));
        }
    }

    SECTION("at [1.5, -0.5]")
    {
        Eigen::VectorXd x{{1.5, -0.5}};
        fd_gradient(fn, x, g_fd);
        fn.gradient(x, g_analytical);

        for(int i = 0; i < 2; ++i)
        {
            CHECK(g_fd(i) == Approx(g_analytical(i)).epsilon(1e-6));
        }
    }
}

TEST_CASE("fd_gradient on Booth", "[finite_difference]")
{
    booth fn;
    Eigen::Vector2d x{2.0, 4.0};
    Eigen::Vector2d g_fd;
    Eigen::Vector2d g_analytical;

    fd_gradient(fn, x, g_fd);
    fn.gradient(x, g_analytical);

    for(int i = 0; i < 2; ++i)
    {
        CHECK(g_fd(i) == Approx(g_analytical(i)).epsilon(1e-6));
    }
}

namespace
{
struct half_norm_sq
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 3; }
    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
};
}

TEST_CASE("fd_gradient on quadratic", "[finite_difference]")
{
    half_norm_sq fn;
    Eigen::VectorXd x{{3.0, -2.0, 1.0}};
    Eigen::VectorXd g_fd(3);

    fd_gradient(fn, x, g_fd);

    // Gradient of 0.5*||x||^2 is x.
    CHECK(g_fd(0) == Approx(3.0).epsilon(1e-8));
    CHECK(g_fd(1) == Approx(-2.0).epsilon(1e-8));
    CHECK(g_fd(2) == Approx(1.0).epsilon(1e-8));
}

TEST_CASE("fd_jacobian on vector-valued function", "[finite_difference]")
{
    // F(x) = [x1^2, x1*x2, x2^2]
    // J = [[2*x1, 0], [x2, x1], [0, 2*x2]]
    auto F = [](const Eigen::VectorXd& x, Eigen::VectorXd& out)
    {
        out(0) = x(0) * x(0);
        out(1) = x(0) * x(1);
        out(2) = x(1) * x(1);
    };

    Eigen::VectorXd x{{2.0, 3.0}};
    Eigen::MatrixXd J;
    fd_jacobian(F, x, J, 3);

    CHECK(J(0, 0) == Approx(4.0).epsilon(1e-6));
    CHECK(J(0, 1) == Approx(0.0).margin(1e-6));
    CHECK(J(1, 0) == Approx(3.0).epsilon(1e-6));
    CHECK(J(1, 1) == Approx(2.0).epsilon(1e-6));
    CHECK(J(2, 0) == Approx(0.0).margin(1e-6));
    CHECK(J(2, 1) == Approx(6.0).epsilon(1e-6));
}

TEST_CASE("fd_gradient with float scalar", "[finite_difference]")
{
    booth<float> fn;
    Eigen::Vector2f x{2.0f, 4.0f};
    Eigen::Vector2f g_fd;
    Eigen::Vector2f g_analytical;

    fd_gradient(fn, x, g_fd);
    fn.gradient(x, g_analytical);

    for(int i = 0; i < 2; ++i)
    {
        CHECK(g_fd(i) == Approx(g_analytical(i)).epsilon(1e-3));
    }
}
