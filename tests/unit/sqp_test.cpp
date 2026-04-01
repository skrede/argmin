#include "nablapp/solver/nw_sqp_policy.h"
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

// Rosenbrock with dummy constraints (no actual constraints).
// Satisfies differentiable && constrained with zero constraints.
struct unconstrained_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

    int dimension() const { return inner.dimension(); }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& /*x*/,
                     Eigen::VectorXd& /*c*/) const {}
    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& /*J*/) const {}
};

// min (x0-1)^2 + (x1-1)^2  s.t. x0 + x1 = 1
// Solution: x* = (0.5, 0.5), f* = 0.5
struct equality_constrained_quadratic
{
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

    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }
};

// min Rosenbrock(x)  s.t. x0^2 + x1^2 <= 2
// Reformulate: c(x) = 2 - x0^2 - x1^2 >= 0
// Solution: (1, 1) is inside the circle (1+1=2, boundary), so x*=(1,1).
struct inequality_constrained_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

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

    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        // Actually depends on x, fix:
    }
};

// Proper version with x-dependent Jacobian
struct inequality_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

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

    void constraint_jacobian(const Eigen::VectorXd& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -2.0 * x[0];
        J(0, 1) = -2.0 * x[1];
    }
};

// Box-constrained + constrained Rosenbrock: 0 <= x <= 0.8
// True minimum (1,1) excluded by bounds.
struct box_constrained_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};

    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& /*x*/,
                     Eigen::VectorXd& /*c*/) const {}
    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& /*J*/) const {}

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{0.0, 0.0}};
    }
    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{0.8, 0.8}};
    }
};

}

// --- Concept satisfaction (compile-time) ---
static_assert(differentiable<unconstrained_rosenbrock>);
static_assert(constrained<unconstrained_rosenbrock>);
static_assert(differentiable<equality_constrained_quadratic>);
static_assert(constrained<equality_constrained_quadratic>);
static_assert(differentiable<inequality_rosenbrock>);
static_assert(constrained<inequality_rosenbrock>);
static_assert(differentiable<box_constrained_rosenbrock>);
static_assert(constrained<box_constrained_rosenbrock>);
static_assert(bound_constrained<box_constrained_rosenbrock>);

TEST_CASE("nw_sqp concept satisfaction", "[sqp]")
{
    // Compile-time checks are the static_asserts above.
    // This test case exists so the test runner reports it.
    REQUIRE(true);
}

TEST_CASE("nw_sqp unconstrained Rosenbrock", "[sqp]")
{
    unconstrained_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver<nw_sqp_policy> solver{problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::ftol_reached
           || result.status == solver_status::stalled));
    CHECK(result.x[0] == Approx(1.0).margin(1e-3));
    CHECK(result.x[1] == Approx(1.0).margin(1e-3));
    CHECK(result.objective_value < 1e-4);
}

TEST_CASE("nw_sqp equality constrained", "[sqp]")
{
    equality_constrained_quadratic problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver<nw_sqp_policy> solver{problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::ftol_reached));
    CHECK(result.x[0] == Approx(0.5).margin(1e-3));
    CHECK(result.x[1] == Approx(0.5).margin(1e-3));
    CHECK(result.objective_value == Approx(0.5).margin(1e-3));
}

TEST_CASE("nw_sqp inequality constrained", "[sqp]")
{
    inequality_rosenbrock problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.set_gradient_threshold(1e-4);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver<nw_sqp_policy> solver{problem, x0, opts};
    auto result = solver.solve(opts);

    // The unconstrained minimum (1,1) is on the boundary of x0^2+x1^2<=2
    // so it should be feasible (1+1=2).
    CHECK(result.objective_value < 0.1);
    CHECK(result.x[0] * result.x[0] + result.x[1] * result.x[1]
          <= 2.0 + 1e-4);
}

TEST_CASE("nw_sqp box constrained", "[sqp]")
{
    box_constrained_rosenbrock problem;
    Eigen::VectorXd x0{{0.4, 0.4}};
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver<nw_sqp_policy> solver{problem, x0, opts};
    auto result = solver.solve(opts);

    // Solution must respect bounds [0, 0.8]
    CHECK(result.x[0] >= -1e-10);
    CHECK(result.x[0] <= 0.8 + 1e-10);
    CHECK(result.x[1] >= -1e-10);
    CHECK(result.x[1] <= 0.8 + 1e-10);

    // Must improve from starting point
    CHECK(result.objective_value < problem.value(x0));
}

TEST_CASE("nw_sqp step solve step_n", "[sqp]")
{
    unconstrained_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;

    SECTION("step returns finite values")
    {
        basic_solver<nw_sqp_policy> solver{problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
            CHECK(std::isfinite(sr.step_size));
        }

        auto result = solver.step_n(100);
        CHECK(std::isfinite(result.objective_value));
    }

    SECTION("solve converges from scratch")
    {
        basic_solver<nw_sqp_policy> solver{problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached
               || result.status == solver_status::stalled));
    }
}
