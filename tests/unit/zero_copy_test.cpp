// EIGEN_RUNTIME_NO_MALLOC must be defined BEFORE any Eigen include
// to enable the runtime malloc-checking machinery.
#define EIGEN_RUNTIME_NO_MALLOC

#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/types.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

using namespace nablapp;

TEST_CASE("fixed-dim value+gradient: zero dynamic allocation", "[zero-copy]")
{
    beale<double> problem;
    Eigen::Vector<double, 2> x;
    x << 0.5, 0.5;
    Eigen::Vector<double, 2> g;

    Eigen::internal::set_is_malloc_allowed(false);

    double f = problem.value(x);
    problem.gradient(x, g);

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

TEST_CASE("fixed-dim solver state types are stack-allocated", "[zero-copy]")
{
    // Verify that basic_solver<lbfgsb_policy<2>, 2> uses fixed-size state
    // vectors, proving end-to-end compile-time dimension propagation.
    using solver_type = basic_solver<lbfgsb_policy<2>, 2>;
    using state_type = typename solver_type::state_type;

    // The state's x and g vectors must be Eigen::Vector<double, 2>,
    // not Eigen::VectorXd (which would be Matrix<double, -1, 1>).
    static_assert(std::is_same_v<decltype(state_type::x),
                                 Eigen::Vector<double, 2>>,
                  "solver state x must be Vector<double, 2>");
    static_assert(std::is_same_v<decltype(state_type::g),
                                 Eigen::Vector<double, 2>>,
                  "solver state g must be Vector<double, 2>");

    // Construct and verify the solver works with the rebound types.
    beale<double> problem;
    Eigen::Vector<double, 2> x0;
    x0 << 0.5, 0.5;
    solver_options opts;
    opts.max_iterations = 3;

    solver_type solver{problem, x0, opts};
    auto result = solver.step();

    CHECK(std::isfinite(result.objective_value));
}
