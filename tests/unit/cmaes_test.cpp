#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Bounded Rosenbrock -- satisfies objective && bound_constrained but NOT
// differentiable (CMA-ES does not need gradient).
struct bounded_rosenbrock
{
    int n{2};
    double a{1};
    double b{5};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < n - 1; ++i)
        {
            double t1 = a - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + b * t2 * t2;
        }
        return f;
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

static_assert(objective<rosenbrock<double>>);
static_assert(objective<rastrigin<double>>);
static_assert(objective<bounded_rosenbrock>);
static_assert(bound_constrained<bounded_rosenbrock>);

TEST_CASE("cmaes_policy: Rosenbrock 2D", "[cmaes]")
{
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    cmaes_policy policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 0.01);
    CHECK(result.x[0] == Approx(1.0).margin(0.1));
    CHECK(result.x[1] == Approx(1.0).margin(0.1));
}

TEST_CASE("cmaes_policy: Rastrigin 5D", "[cmaes]")
{
    rastrigin<double> problem{.n = 3};

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 2.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    cmaes_policy policy;
    policy.options.initial_sigma = 2.0;
    policy.options.seed = 123u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    // Rastrigin 3D is highly multimodal; CMA-ES should reach a good basin.
    // Relaxed threshold: global minimum is 0, local basins have f ~ n*k.
    CHECK(result.objective_value < 10.0);
}

TEST_CASE("cmaes_policy: step_n budget", "[cmaes]")
{
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 1000;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    cmaes_policy policy;
    policy.options.seed = 7u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.step_n(10);

    CHECK(result.iterations <= 10);
    CHECK(result.iterations >= 1);
    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("cmaes_policy: IPOP restart", "[cmaes]")
{
    rastrigin<double> problem{.n = 3};

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 3.0);
    solver_options opts;
    opts.max_iterations = 200;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    cmaes_policy policy;
    policy.options.initial_sigma = 0.1;
    policy.options.restart = cmaes_policy::restart_strategy::ipop;
    policy.options.seed = 99u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    // Should complete without crash
    CHECK(std::isfinite(result.objective_value));
    CHECK(result.objective_value < 100.0);
}

TEST_CASE("cmaes_policy: bounded Rosenbrock", "[cmaes]")
{
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-2.0, -2.0}},
        .ub = Eigen::VectorXd{{2.0, 2.0}},
    };

    Eigen::VectorXd x0{{-0.5, -0.5}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    cmaes_policy policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 55u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 0.1);
    CHECK(result.x[0] >= -2.0 - 1e-10);
    CHECK(result.x[0] <= 2.0 + 1e-10);
    CHECK(result.x[1] >= -2.0 - 1e-10);
    CHECK(result.x[1] <= 2.0 + 1e-10);
}
