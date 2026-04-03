// EIGEN_RUNTIME_NO_MALLOC must be defined BEFORE any Eigen include
// to enable the runtime malloc-checking machinery.
#define EIGEN_RUNTIME_NO_MALLOC

#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/types.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace nablapp;

TEST_CASE("fixed-dim value+gradient: zero dynamic allocation", "[zero-copy]")
{
    // beale is a fixed 2D problem (problem_dimension = 2).
    // With full N-propagation, all problem-dimension vectors should be
    // stack-allocated Eigen::Vector<double, 2> through value/gradient.
    beale<double> problem;
    Eigen::Vector<double, 2> x;
    x << 0.5, 0.5;
    Eigen::Vector<double, 2> g;

    // Construction is done — guard only the hot path.
    Eigen::internal::set_is_malloc_allowed(false);

    double f = problem.value(x);
    problem.gradient(x, g);

    // Also test detail::project with fixed-dim vectors.
    Eigen::Vector<double, 2> lower;
    lower << -10.0, -10.0;
    Eigen::Vector<double, 2> upper;
    upper << 10.0, 10.0;
    auto projected = detail::project(x, lower, upper);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(f));
    CHECK(std::isfinite(g[0]));
    CHECK(std::isfinite(g[1]));
    CHECK(projected[0] == x[0]);
}

TEST_CASE("fixed-dim booth value+gradient: zero dynamic allocation", "[zero-copy]")
{
    booth<double> problem;
    Eigen::Vector<double, 2> x;
    x << 1.0, 3.0;
    Eigen::Vector<double, 2> g;

    Eigen::internal::set_is_malloc_allowed(false);

    double f = problem.value(x);
    problem.gradient(x, g);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(f < 1e-10);
    CHECK(std::abs(g[0]) < 1e-8);
    CHECK(std::abs(g[1]) < 1e-8);
}
