#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/booth.h"
#include "argmin/test_functions/beale.h"
#include "argmin/test_functions/ackley.h"
#include "argmin/test_functions/rastrigin.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/himmelblau.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// ------------------------------------------------------------------
// Concept satisfaction (compile-time verification)
// ------------------------------------------------------------------

// double
static_assert(argmin::objective<argmin::rosenbrock<double>>);
static_assert(argmin::objective<argmin::booth<double>>);
static_assert(argmin::objective<argmin::beale<double>>);
static_assert(argmin::objective<argmin::himmelblau<double>>);
static_assert(argmin::objective<argmin::rastrigin<double>>);
static_assert(argmin::objective<argmin::ackley<double>>);

static_assert(argmin::differentiable<argmin::rosenbrock<double>>);
static_assert(argmin::differentiable<argmin::booth<double>>);
static_assert(argmin::differentiable<argmin::beale<double>>);
static_assert(argmin::differentiable<argmin::himmelblau<double>>);
static_assert(argmin::differentiable<argmin::rastrigin<double>>);
static_assert(argmin::differentiable<argmin::ackley<double>>);

// float
static_assert(argmin::objective<argmin::rosenbrock<float>, float>);
static_assert(argmin::objective<argmin::booth<float>, float>);
static_assert(argmin::objective<argmin::beale<float>, float>);
static_assert(argmin::objective<argmin::himmelblau<float>, float>);
static_assert(argmin::objective<argmin::rastrigin<float>, float>);
static_assert(argmin::objective<argmin::ackley<float>, float>);

static_assert(argmin::differentiable<argmin::rosenbrock<float>, float>);
static_assert(argmin::differentiable<argmin::booth<float>, float>);
static_assert(argmin::differentiable<argmin::beale<float>, float>);
static_assert(argmin::differentiable<argmin::himmelblau<float>, float>);
static_assert(argmin::differentiable<argmin::rastrigin<float>, float>);
static_assert(argmin::differentiable<argmin::ackley<float>, float>);

// ------------------------------------------------------------------
// Value and gradient tests at known optima
// ------------------------------------------------------------------

using Catch::Approx;

TEST_CASE("rosenbrock", "[test_functions]")
{
    argmin::rosenbrock<> fn{.a = 1.0, .b = 5.0, .n = 2};
    Eigen::VectorXd x_star{{1.0, 1.0}};
    Eigen::VectorXd g(2);

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }

    SECTION("n-dimensional (n=4)")
    {
        argmin::rosenbrock<> fn4{.a = 1.0, .b = 5.0, .n = 4};
        Eigen::VectorXd x4{{1.0, 1.0, 1.0, 1.0}};
        CHECK(fn4.value(x4) == Approx(0.0).margin(1e-10));
        CHECK(fn4.dimension() == 4);
    }
}

TEST_CASE("booth", "[test_functions]")
{
    argmin::booth fn;
    Eigen::Vector<double, 2> x_star;
    x_star << 1.0, 3.0;
    Eigen::Vector<double, 2> g;

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }
}

TEST_CASE("beale", "[test_functions]")
{
    argmin::beale fn;
    Eigen::Vector<double, 2> x_star;
    x_star << 3.0, 0.5;
    Eigen::Vector<double, 2> g;

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }
}

TEST_CASE("himmelblau", "[test_functions]")
{
    argmin::himmelblau fn;
    Eigen::Vector<double, 2> x_star;
    x_star << 3.0, 2.0;
    Eigen::Vector<double, 2> g;

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }
}

TEST_CASE("rastrigin", "[test_functions]")
{
    argmin::rastrigin<> fn{.n = 2};
    Eigen::VectorXd x_star{{0.0, 0.0}};
    Eigen::VectorXd g(2);

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }
}

TEST_CASE("ackley", "[test_functions]")
{
    argmin::ackley fn;
    Eigen::VectorXd x_star{{0.0, 0.0}};
    Eigen::VectorXd g(2);

    SECTION("value at optimum")
    {
        CHECK(fn.value(x_star) == Approx(0.0).margin(1e-10));
    }

    SECTION("gradient at optimum")
    {
        fn.gradient(x_star, g);
        CHECK(g[0] == Approx(0.0).margin(1e-8));
        CHECK(g[1] == Approx(0.0).margin(1e-8));
    }

    SECTION("dimension")
    {
        CHECK(fn.dimension() == 2);
    }
}
