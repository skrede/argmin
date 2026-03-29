#include "nablapp/line_search/armijo.h"
#include "nablapp/line_search/strong_wolfe.h"
#include "nablapp/test_functions/rosenbrock.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

// ---------------------------------------------------------------------------
// Armijo tests
// ---------------------------------------------------------------------------

TEST_CASE("armijo on quadratic", "[line_search][armijo]")
{
    Eigen::VectorXd x{{2.0, 2.0}};
    Eigen::VectorXd d{{-2.0, -2.0}};

    auto phi = [&](double alpha) -> double
    {
        return 0.5 * (x + alpha * d).squaredNorm();
    };

    double phi0 = phi(0.0);
    double dphi0 = x.dot(d);

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
    Eigen::VectorXd d = -g;

    auto phi = [&](double alpha) -> double
    {
        return fn.value(x + alpha * d);
    };

    double phi0 = phi(0.0);
    double dphi0 = g.dot(d);

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

// ---------------------------------------------------------------------------
// Strong Wolfe tests
// ---------------------------------------------------------------------------

TEST_CASE("strong_wolfe on quadratic", "[line_search][strong_wolfe]")
{
    // phi(alpha) = 0.5 * ||x + alpha * d||^2
    // x = [2, 2], d = [-2, -2]
    // phi(0) = 4, dphi(0) = -8
    // Optimal: alpha = 1 where phi(1) = 0
    Eigen::VectorXd x{{2.0, 2.0}};
    Eigen::VectorXd d{{-2.0, -2.0}};

    auto phi = [&](double alpha) -> double
    {
        return 0.5 * (x + alpha * d).squaredNorm();
    };

    auto dphi = [&](double alpha) -> double
    {
        return (x + alpha * d).dot(d);
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    REQUIRE(phi0 == Approx(4.0));
    REQUIRE(dphi0 == Approx(-8.0));

    auto result = strong_wolfe(phi, dphi, phi0, dphi0);

    CHECK(result.success);

    // Verify sufficient decrease (Armijo): phi(alpha) <= phi0 + c1*alpha*dphi0
    line_search_options<double> opts;
    CHECK(result.value <= phi0 + opts.c1 * result.alpha * dphi0);

    // Verify curvature condition: |dphi(alpha)| <= c2*|dphi0|
    double dphi_at_alpha = dphi(result.alpha);
    CHECK(std::abs(dphi_at_alpha) <= opts.c2 * std::abs(dphi0));
}

TEST_CASE("strong_wolfe on Rosenbrock", "[line_search][strong_wolfe]")
{
    rosenbrock fn{.a = 1.0, .b = 5.0, .n = 2};
    Eigen::VectorXd x{{-1.0, 1.0}};

    Eigen::VectorXd g(2);
    fn.gradient(x, g);
    Eigen::VectorXd d = -g;

    auto phi = [&](double alpha) -> double
    {
        return fn.value(x + alpha * d);
    };

    auto dphi = [&](double alpha) -> double
    {
        Eigen::VectorXd g_at(2);
        fn.gradient(x + alpha * d, g_at);
        return g_at.dot(d);
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    REQUIRE(dphi0 < 0.0);

    auto result = strong_wolfe(phi, dphi, phi0, dphi0);

    CHECK(result.success);
    CHECK(result.value < phi0);

    // Verify sufficient decrease
    line_search_options<double> opts;
    CHECK(result.value <= phi0 + opts.c1 * result.alpha * dphi0);

    // Verify curvature condition
    double dphi_at_alpha = dphi(result.alpha);
    CHECK(std::abs(dphi_at_alpha) <= opts.c2 * std::abs(dphi0));
}

TEST_CASE("strong_wolfe sufficient decrease verified", "[line_search][strong_wolfe]")
{
    // Verify on a different problem: phi(alpha) = (alpha - 3)^4
    // phi(0) = 81, dphi(0) = -108
    auto phi = [](double alpha) -> double
    {
        double t = alpha - 3.0;
        return t * t * t * t;
    };

    auto dphi = [](double alpha) -> double
    {
        double t = alpha - 3.0;
        return 4.0 * t * t * t;
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    line_search_options<double> opts;
    opts.max_alpha = 5.0;

    auto result = strong_wolfe(phi, dphi, phi0, dphi0, opts);

    CHECK(result.success);
    CHECK(result.value <= phi0 + opts.c1 * result.alpha * dphi0);
}

TEST_CASE("strong_wolfe curvature condition verified", "[line_search][strong_wolfe]")
{
    // Same quartic as above -- verify curvature condition explicitly
    auto phi = [](double alpha) -> double
    {
        double t = alpha - 3.0;
        return t * t * t * t;
    };

    auto dphi = [](double alpha) -> double
    {
        double t = alpha - 3.0;
        return 4.0 * t * t * t;
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    line_search_options<double> opts;
    opts.max_alpha = 5.0;

    auto result = strong_wolfe(phi, dphi, phi0, dphi0, opts);

    CHECK(result.success);
    double dphi_at_alpha = dphi(result.alpha);
    CHECK(std::abs(dphi_at_alpha) <= opts.c2 * std::abs(dphi0));
}

TEST_CASE("strong_wolfe with insufficient iterations", "[line_search][strong_wolfe]")
{
    // Rosenbrock-derived phi where convergence needs many evals.
    // With budget=0, no evaluation can happen at all.
    auto phi = [](double alpha) -> double
    {
        return (alpha - 2.0) * (alpha - 2.0);
    };

    auto dphi = [](double alpha) -> double
    {
        return 2.0 * (alpha - 2.0);
    };

    line_search_options<double> opts;
    opts.max_iterations = 0;

    auto result = strong_wolfe(phi, dphi, 4.0, -4.0, opts);

    CHECK_FALSE(result.success);
}
