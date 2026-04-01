#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Quadratic with equality constraint: min (x1-1)^2+(x2-1)^2 s.t. x1+x2=1.
// Solution: (0.5, 0.5), f* = 0.5.
struct equality_quadratic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0);
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }
};

// Rosenbrock with inequality: min rosenbrock s.t. x0^2 + x1^2 <= 2.
// Inequality convention: c_ineq(x) >= 0, so c_ineq = 2 - x0^2 - x1^2.
struct ineq_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return inner.value(x); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 2.0 - x[0] * x[0] - x[1] * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -2.0 * x[0];
        J(0, 1) = -2.0 * x[1];
    }
};

// Box-constrained Rosenbrock: 0 <= x <= 0.8.
struct box_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return inner.value(x); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 0.0);
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 0.8);
    }
};

// Box-constrained Rosenbrock for liepp-like budget test.
struct liepp_box_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return inner.value(x); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -2.0);
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 2.0);
    }
};

}

// Static concept checks
static_assert(differentiable<equality_quadratic>);
static_assert(constrained<equality_quadratic>);
static_assert(differentiable<ineq_rosenbrock>);
static_assert(constrained<ineq_rosenbrock>);
static_assert(differentiable<box_rosenbrock>);
static_assert(bound_constrained<box_rosenbrock>);

TEST_CASE("kraft_slsqp unconstrained Rosenbrock", "[kraft_slsqp]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 500;

    basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    auto result = solver.solve();

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::ftol_reached));
    CHECK(result.x[0] == Approx(1.0).margin(1e-4));
    CHECK(result.x[1] == Approx(1.0).margin(1e-4));
    CHECK(result.objective_value < 1e-6);
}

TEST_CASE("kraft_slsqp equality constrained", "[kraft_slsqp]")
{
    equality_quadratic problem;
    Eigen::VectorXd x0{{0.0, 1.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.x[0] == Approx(0.5).margin(1e-3));
    CHECK(result.x[1] == Approx(0.5).margin(1e-3));
    // Constraint satisfaction: x0 + x1 = 1
    CHECK(result.x[0] + result.x[1] == Approx(1.0).margin(1e-3));
}

TEST_CASE("kraft_slsqp inequality constrained", "[kraft_slsqp]")
{
    ineq_rosenbrock problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    auto result = solver.solve();

    // Solution is (1,1) which satisfies 1+1=2<=2 (active constraint).
    // Or near it. Objective should be near zero.
    CHECK(result.objective_value < 0.1);
    // Constraint satisfaction: x0^2 + x1^2 <= 2
    CHECK(result.x[0] * result.x[0] + result.x[1] * result.x[1] <= 2.0 + 1e-3);
}

TEST_CASE("kraft_slsqp box constrained", "[kraft_slsqp]")
{
    box_rosenbrock problem;
    Eigen::VectorXd x0{{0.4, 0.4}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 500;
    opts.step_tolerance = 1e-15;
    opts.objective_tolerance = 1e-15;

    basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    auto result = solver.solve();

    // Solution constrained by upper bound (optimum (1,1) excluded).
    // x should be within [0, 0.8].
    CHECK(result.x[0] >= -1e-6);
    CHECK(result.x[0] <= 0.8 + 1e-6);
    CHECK(result.x[1] >= -1e-6);
    CHECK(result.x[1] <= 0.8 + 1e-6);

    // Should improve from starting point
    CHECK(result.objective_value < problem.value(x0));
}

TEST_CASE("kraft_slsqp liepp-like budget", "[kraft_slsqp]")
{
    // Match liepp slsqp_stepper.h configuration:
    // max_iterations=500, gradient_tolerance~1e-8, objective_tolerance~1e-12
    liepp_box_rosenbrock problem;
    Eigen::VectorXd x0{{-1.0, 0.5}};
    solver_options opts;
    opts.gradient_tolerance = 1e-8;
    opts.objective_tolerance = 1e-12;
    opts.step_tolerance = 1e-15;
    opts.max_iterations = 500;

    basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    auto result = solver.solve();

    // Must converge within 500 iterations (liepp budget)
    CHECK(result.iterations <= 500);
    CHECK(result.x[0] == Approx(1.0).margin(1e-3));
    CHECK(result.x[1] == Approx(1.0).margin(1e-3));
    CHECK(result.objective_value < 1e-4);
}

TEST_CASE("kraft_slsqp concept satisfaction", "[kraft_slsqp]")
{
    // kraft_slsqp_policy works with differentiable problems
    static_assert(differentiable<rosenbrock<>>);

    // kraft_slsqp_policy works with constrained problems
    static_assert(constrained<equality_quadratic>);
    static_assert(constrained<ineq_rosenbrock>);

    // kraft_slsqp_policy works with bound_constrained problems
    static_assert(bound_constrained<box_rosenbrock>);

    // kraft_slsqp_policy compiles with basic_solver
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    [[maybe_unused]] basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
    CHECK(true);
}

TEST_CASE("kraft_slsqp step solve step_n", "[kraft_slsqp]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 500;

    SECTION("step returns finite values")
    {
        basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
            CHECK(std::isfinite(sr.step_size));
        }
    }

    SECTION("step_n with budget")
    {
        basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
        auto result = solver.step_n(50);
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.iterations <= 50);
    }

    SECTION("solve runs to convergence")
    {
        basic_solver<kraft_slsqp_policy> solver{problem, x0, opts};
        auto result = solver.solve();
        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached));
    }
}
