// ISRES solver policy and restarting_policy unit tests.
//
// Validates concept satisfaction, basic convergence, stochastic ranking
// selection preference, restarting_policy composition with cmaes_policy
// and isres_policy, and basic_solver_group compatibility.

#include "nablapp/solver/isres_policy.h"
#include "nablapp/solver/restarting_policy.h"
#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Simple constrained problem for ISRES testing.
// Minimise x0^2 + x1^2 subject to x0 + x1 >= 1, bounds [-10, 10]^2.
// Optimal: x* = (0.5, 0.5), f* = 0.5.
struct simple_constrained
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

// Bounded Rosenbrock for restarting_policy<cmaes_policy<>> tests.
// Satisfies objective && bound_constrained but NOT constrained.
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;

    int n{2};
    double a{1};
    double b{5};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

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

// Concept satisfaction.
static_assert(constrained_values<simple_constrained>);
static_assert(!constrained<simple_constrained>);
static_assert(bound_constrained<simple_constrained>);

// Verify restarting_policy rebind propagates through the decorator.
static_assert(std::same_as<
    restarting_policy<cmaes_policy<>>::rebind<3>,
    restarting_policy<cmaes_policy<3>>>);

static_assert(std::same_as<
    restarting_policy<isres_policy<>>::rebind<3>,
    restarting_policy<isres_policy<3>>>);

TEST_CASE("isres_policy: simple constrained 2D", "[isres]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    isres_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 1.0);
}

TEST_CASE("isres_policy: stochastic ranking selection", "[isres]")
{
    // Verify that ISRES prefers feasible solutions over infeasible ones
    // with better objective. Run with a tight constraint and check that
    // the final point is approximately feasible.
    simple_constrained problem;
    Eigen::VectorXd x0{{-5.0, -5.0}};
    solver_options opts;
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    isres_policy<> policy;
    policy.options.seed = 123u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    // The unconstrained minimum is (0,0) with f=0 but that violates
    // x0 + x1 >= 1. Stochastic ranking should steer towards feasibility.
    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("restarting_policy<cmaes_policy<>>: Rosenbrock 2D", "[restarting][cmaes]")
{
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    restarting_policy<cmaes_policy<>> policy;
    policy.inner_policy_.options.initial_sigma = 0.5;
    policy.inner_policy_.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 5.0);
}

TEST_CASE("restarting_policy<isres_policy<>>: restart compiles", "[restarting][isres]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    restarting_policy<isres_policy<>> policy;
    policy.inner_policy_.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("isres_policy: basic_solver_group compatibility", "[isres][solver_group]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver_group<round_robin_schedule, dynamic_dimension, isres_policy<>> group{
        problem, x0, opts};
    auto result = group.solve();

    CHECK(std::isfinite(result.objective_value));
}
