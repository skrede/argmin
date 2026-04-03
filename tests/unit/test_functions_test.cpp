#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/ackley.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/himmelblau.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// ------------------------------------------------------------------
// Concept satisfaction (compile-time verification)
// ------------------------------------------------------------------

// double
static_assert(nablapp::objective<nablapp::rosenbrock<double>>);
static_assert(nablapp::objective<nablapp::booth<double>>);
static_assert(nablapp::objective<nablapp::beale<double>>);
static_assert(nablapp::objective<nablapp::himmelblau<double>>);
static_assert(nablapp::objective<nablapp::rastrigin<double>>);
static_assert(nablapp::objective<nablapp::ackley<double>>);

static_assert(nablapp::differentiable<nablapp::rosenbrock<double>>);
static_assert(nablapp::differentiable<nablapp::booth<double>>);
static_assert(nablapp::differentiable<nablapp::beale<double>>);
static_assert(nablapp::differentiable<nablapp::himmelblau<double>>);
static_assert(nablapp::differentiable<nablapp::rastrigin<double>>);
static_assert(nablapp::differentiable<nablapp::ackley<double>>);

// float
static_assert(nablapp::objective<nablapp::rosenbrock<float>, float>);
static_assert(nablapp::objective<nablapp::booth<float>, float>);
static_assert(nablapp::objective<nablapp::beale<float>, float>);
static_assert(nablapp::objective<nablapp::himmelblau<float>, float>);
static_assert(nablapp::objective<nablapp::rastrigin<float>, float>);
static_assert(nablapp::objective<nablapp::ackley<float>, float>);

static_assert(nablapp::differentiable<nablapp::rosenbrock<float>, float>);
static_assert(nablapp::differentiable<nablapp::booth<float>, float>);
static_assert(nablapp::differentiable<nablapp::beale<float>, float>);
static_assert(nablapp::differentiable<nablapp::himmelblau<float>, float>);
static_assert(nablapp::differentiable<nablapp::rastrigin<float>, float>);
static_assert(nablapp::differentiable<nablapp::ackley<float>, float>);

// ------------------------------------------------------------------
// Value and gradient tests at known optima
// ------------------------------------------------------------------

using Catch::Approx;

TEST_CASE("rosenbrock", "[test_functions]")
{
    nablapp::rosenbrock fn{.a = 1.0, .b = 5.0, .n = 2};
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
        nablapp::rosenbrock fn4{.a = 1.0, .b = 5.0, .n = 4};
        Eigen::VectorXd x4{{1.0, 1.0, 1.0, 1.0}};
        CHECK(fn4.value(x4) == Approx(0.0).margin(1e-10));
        CHECK(fn4.dimension() == 4);
    }
}

TEST_CASE("booth", "[test_functions]")
{
    nablapp::booth fn;
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
    nablapp::beale fn;
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
    nablapp::himmelblau fn;
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
    nablapp::rastrigin fn{.n = 2};
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
    nablapp::ackley fn;
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
