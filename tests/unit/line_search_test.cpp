#include "argmin/line_search/armijo.h"
#include "argmin/line_search/strong_wolfe.h"
#include "argmin/test_functions/rosenbrock.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using Catch::Approx;
using namespace argmin;

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

    line_search_options opts;
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
    line_search_options opts;
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
    line_search_options opts;
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

    line_search_options opts;
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

    line_search_options opts;
    opts.max_alpha = 5.0;

    auto result = strong_wolfe(phi, dphi, phi0, dphi0, opts);

    CHECK(result.success);
    double dphi_at_alpha = dphi(result.alpha);
    CHECK(std::abs(dphi_at_alpha) <= opts.c2 * std::abs(dphi0));
}

TEST_CASE("strong_wolfe interpolation uses fewer evaluations", "[line_search][strong_wolfe]")
{
    // Simple quadratic: phi(a) = (a - 0.5)^2, minimum at a = 0.5.
    // Interpolation should find this in very few evaluations.
    auto phi = [](double alpha) -> double
    {
        return (alpha - 0.5) * (alpha - 0.5);
    };

    auto dphi = [](double alpha) -> double
    {
        return 2.0 * (alpha - 0.5);
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    line_search_options opts;
    opts.max_alpha = 1.0;

    auto result = strong_wolfe(phi, dphi, phi0, dphi0, opts);

    CHECK(result.success);
    CHECK(result.alpha == Approx(0.5).epsilon(1e-6));
    // With interpolation, a quadratic should be solved in very few evals.
    // Pure bisection would take ~50 iterations on [0, 1].
    CHECK(result.evaluations < 10);
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

    double phi0 = 4.0;

    line_search_options opts;
    opts.max_iterations = 0;

    auto result = strong_wolfe(phi, dphi, phi0, -4.0, opts);

    CHECK_FALSE(result.success);
    // Armijo-consistent failure contract: value == phi0, not 0.
    CHECK(result.value == Approx(phi0));
}

TEST_CASE("strong_wolfe bracketing expands from a unit initial step", "[line_search][strong_wolfe]")
{
    // phi(alpha) = (alpha - 50)^2; minimum at alpha = 50.
    // With a large max_alpha, the bracketing phase must start at the
    // unit initial trial (N&W Algorithm 3.5, alpha_1 = 1) and grow
    // geometrically toward max_alpha, rather than jumping straight to
    // max_alpha on the first trial.
    auto phi = [](double alpha) -> double
    {
        double t = alpha - 50.0;
        return t * t;
    };

    auto dphi = [](double alpha) -> double
    {
        return 2.0 * (alpha - 50.0);
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    std::vector<double> evaluated_alphas;
    auto tracked_phi = [&](double alpha) -> double
    {
        evaluated_alphas.push_back(alpha);
        return phi(alpha);
    };

    line_search_options opts;
    opts.max_alpha = 100.0;

    auto result = strong_wolfe(tracked_phi, dphi, phi0, dphi0, opts);

    REQUIRE_FALSE(evaluated_alphas.empty());
    CHECK(evaluated_alphas.front() == Approx(1.0));

    bool intermediate_trial_found = false;
    for(double a : evaluated_alphas)
    {
        if(a > 1.0 && a < opts.max_alpha)
        {
            intermediate_trial_found = true;
            break;
        }
    }
    CHECK(intermediate_trial_found);

    CHECK(result.success);
}

TEST_CASE("strong_wolfe gates a non-finite trial encountered during expansion", "[line_search][strong_wolfe]")
{
    // phi has constant slope -1 on [0, 5) -- never satisfying the default
    // curvature condition -- and is non-finite for alpha >= 5, forcing the
    // bracketing phase's geometric expansion (1, 2, 4, 8, ...) to encounter
    // a non-finite trial. Without the isfinite gate, the non-finite value
    // would silently fail the ordered Armijo/value comparisons (false <
    // anything is false) and be treated as an accepted, decreasing point.
    auto phi = [](double alpha) -> double
    {
        if(alpha >= 5.0)
            return std::numeric_limits<double>::quiet_NaN();
        return -alpha;
    };

    auto dphi = [](double alpha) -> double
    {
        if(alpha >= 5.0)
            return std::numeric_limits<double>::quiet_NaN();
        return -1.0;
    };

    double phi0 = phi(0.0);
    double dphi0 = dphi(0.0);

    line_search_options opts;
    opts.max_alpha = 100.0;

    auto result = strong_wolfe(phi, dphi, phi0, dphi0, opts);

    CHECK_FALSE(result.success);
    // Armijo-consistent failure contract: value == phi0, not 0.
    CHECK(result.value == Approx(phi0));
    // The gate must actually have fired -- this counter is only
    // incremented by the isfinite gate, never by the ordered comparisons.
    CHECK(result.diagnostics.nan_eval_count > 0);
}
