#include "nablapp/line_search/armijo.h"
#include "nablapp/test_functions/rosenbrock.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("armijo on quadratic", "[line_search][armijo]")
{
    // phi(alpha) = 0.5 * ||x + alpha * d||^2
    // with x = [2, 2], d = [-2, -2] (steepest descent = -x)
    // phi(0) = 4, phi'(0) = x^T d = -8
    // phi(1) = 0 (exact minimum)
    Eigen::VectorXd x{{2.0, 2.0}};
    Eigen::VectorXd d{{-2.0, -2.0}};

    auto phi = [&](double alpha) -> double
    {
        return 0.5 * (x + alpha * d).squaredNorm();
    };

    double phi0 = phi(0.0);
    double dphi0 = x.dot(d); // -8.0

    REQUIRE(phi0 == Approx(4.0));
    REQUIRE(dphi0 == Approx(-8.0));

    auto result = armijo(phi, phi0, dphi0);

    CHECK(result.success);
    CHECK(result.alpha == Approx(1.0).epsilon(1e-6));
    CHECK(result.value == Approx(0.0).margin(1e-10));
    CHECK(result.evaluations > 0);
}

TEST_CASE("armijo on Rosenbrock", "[line_search][armijo]")
{
    rosenbrock fn{.a = 1.0, .b = 5.0, .n = 2};
    Eigen::VectorXd x{{-1.0, 1.0}};

    Eigen::VectorXd g(2);
    fn.gradient(x, g);
    Eigen::VectorXd d = -g; // steepest descent

    auto phi = [&](double alpha) -> double
    {
        return fn.value(x + alpha * d);
    };

    double phi0 = phi(0.0);
    double dphi0 = g.dot(d); // -||g||^2

    REQUIRE(dphi0 < 0.0);

    auto result = armijo(phi, phi0, dphi0);

    CHECK(result.success);
    CHECK(result.value < phi0);
}

TEST_CASE("armijo with zero iterations", "[line_search][armijo]")
{
    auto phi = [](double alpha) -> double { return alpha * alpha; };

    line_search_options<double> opts;
    opts.max_iterations = 0;

    auto result = armijo(phi, 1.0, -2.0, opts);

    CHECK_FALSE(result.success);
}

TEST_CASE("armijo evaluation count", "[line_search][armijo]")
{
    int call_count = 0;
    auto phi = [&](double alpha) -> double
    {
        ++call_count;
        return (alpha - 1.0) * (alpha - 1.0);
    };

    auto result = armijo(phi, 1.0, -2.0);

    CHECK(result.evaluations == call_count);
    CHECK(result.evaluations > 0);
}
