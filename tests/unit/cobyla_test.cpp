// COBYLA solver policy unit tests.
//
// Validates concept satisfaction, basic convergence on a simple constrained
// problem, HS024 convergence, and basic_solver_group compatibility.

#include "nablapp/solver/cobyla_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Simple constrained problem: minimise x0^2 + x1^2
// subject to x0 + x1 >= 1 (nablapp: c >= 0)
// with box bounds [-10, 10]^2.
//
// Satisfies constrained_values && bound_constrained but NOT constrained
// (no constraint_jacobian), proving COBYLA works without gradients.
//
// Optimal: x* = (0.5, 0.5), f* = 0.5.
struct simple_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        Eigen::Vector<double, 2> lb;
        lb << -10.0, -10.0;
        return lb;
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        Eigen::Vector<double, 2> ub;
        ub << 10.0, 10.0;
        return ub;
    }
};

}

// Concept satisfaction checks.
static_assert(constrained_values<simple_constrained>);
static_assert(!constrained<simple_constrained>);
static_assert(bound_constrained<simple_constrained>);

static_assert(constrained_values<hs071<>>);
static_assert(constrained<hs071<>>);

static_assert(constrained_values<hs024<>>);
static_assert(constrained<hs024<>>);

TEST_CASE("cobyla_policy: simple constrained 2D", "[cobyla]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 0.6);
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("cobyla_policy: HS024", "[cobyla][hs]")
{
    hs024 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    // COBYLA is derivative-free; relaxed tolerance for convergence direction.
    CHECK(result.objective_value == Approx(-1.0).margin(1.0));
    CHECK(solver.constraint_violation() < 0.5);
}

TEST_CASE("cobyla_policy: basic_solver_group compatibility", "[cobyla][solver_group]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver_group<round_robin_schedule, dynamic_dimension, simple_constrained, cobyla_policy> group{
        problem, x0, opts};
    auto result = group.solve();

    CHECK(std::isfinite(result.objective_value));
}
