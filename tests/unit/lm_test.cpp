#include "nablapp/solver/lm_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Rosenbrock in least-squares residual form (b=5 per project convention).
// r_0 = 1 - x_0, r_1 = sqrt(5)*(x_1 - x_0^2)
// f(x) = 0.5*(r_0^2 + r_1^2), minimum at (1,1), f*=0.
// Provides analytic Jacobian.
struct rosenbrock_ls
{
    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }
};

// Exponential decay fitting: y = a * exp(b * t).
// 5 data points, 2 parameters. Provides analytic Jacobian.
struct exponential_fitting
{
    static constexpr double t[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    static constexpr double y[] = {2.0, 1.2, 0.75, 0.45, 0.28};

    int dimension() const { return 2; }
    int num_residuals() const { return 5; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < 5; ++i)
        {
            double ri = x(0) * std::exp(x(1) * t[i]) - y[i];
            f += ri * ri;
        }
        return 0.5 * f;
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        for(int i = 0; i < 5; ++i)
            r(i) = x(0) * std::exp(x(1) * t[i]) - y[i];
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        for(int i = 0; i < 5; ++i)
        {
            double e = std::exp(x(1) * t[i]);
            J(i, 0) = e;
            J(i, 1) = x(0) * t[i] * e;
        }
    }
};

// Rosenbrock residual form WITHOUT jacobian method -- triggers FD fallback.
struct rosenbrock_ls_no_jac
{
    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }
};

}

// Concept satisfaction checks
static_assert(objective<rosenbrock_ls>);
static_assert(least_squares<rosenbrock_ls>);
static_assert(objective<exponential_fitting>);
static_assert(least_squares<exponential_fitting>);
static_assert(objective<rosenbrock_ls_no_jac>);
static_assert(!least_squares<rosenbrock_ls_no_jac>);

TEST_CASE("lm_policy: Rosenbrock residual form", "[lm]")
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.gradient_tolerance = 1e-12;

    basic_solver solver{lm_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(solver.state().objective_value < 1e-12);
    CHECK(result.x(0) == Approx(1.0).margin(1e-6));
    CHECK(result.x(1) == Approx(1.0).margin(1e-6));
}

TEST_CASE("lm_policy: exponential fitting", "[lm]")
{
    exponential_fitting problem;
    Eigen::VectorXd x0{{1.0, -0.5}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.gradient_tolerance = 1e-10;

    basic_solver solver{lm_policy{}, problem, x0, opts};
    auto result = solver.solve();

    // Data is approximate exponential, so residual won't be zero
    CHECK(solver.state().objective_value < 0.01);
}

TEST_CASE("lm_policy: FD Jacobian fallback", "[lm]")
{
    rosenbrock_ls_no_jac problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.gradient_tolerance = 1e-10;

    basic_solver solver{lm_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(solver.state().objective_value < 1e-8);
    CHECK(result.x(0) == Approx(1.0).margin(1e-4));
    CHECK(result.x(1) == Approx(1.0).margin(1e-4));
}

TEST_CASE("lm_policy: step_n budget", "[lm]")
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 1000;
    opts.gradient_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;
    opts.step_tolerance = 1e-15;

    basic_solver solver{lm_policy{}, problem, x0, opts};
    double f0 = solver.state().objective_value;

    auto result = solver.step_n(5);

    CHECK(std::isfinite(result.objective_value));
    CHECK(result.iterations <= 5);
    CHECK(result.objective_value < f0);
}

TEST_CASE("lm_policy: rejected steps recovery", "[lm]")
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{10.0, 10.0}};

    solver_options opts;
    opts.max_iterations = 500;
    opts.gradient_tolerance = 1e-10;

    basic_solver solver{lm_policy{}, problem, x0, opts};
    auto result = solver.solve();

    // Should converge despite starting far away (some rejected steps expected)
    CHECK(solver.state().objective_value < 1e-8);
    CHECK(result.x(0) == Approx(1.0).margin(1e-4));
    CHECK(result.x(1) == Approx(1.0).margin(1e-4));
}
