// Convergence pin for the augmented-Lagrangian / globally-convergent-MMA
// composition on equality-constrained problems.
//
// This composition was impossible before the equality-lifting adapter: the
// moving-asymptotes family is an inequality+box method and rejects
// equalities. The augmented Lagrangian now lifts the equalities into its
// augmented objective (quadratic penalty + Hestenes-Powell multiplier
// update) and passes any inequalities + box straight through to the inner
// solver. The globally convergent variant is the convergence-guaranteed
// inner solver under the standard augmented-Lagrangian assumptions, so it
// is the composition pinned here.

#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

namespace
{

// min  x0^2 + x1^2   s.t.  x0 + x1 - 1 = 0,   x in [-10, 10]^2.
// Hand-computed KKT point: x* = (0.5, 0.5), f* = 0.5, lambda* = 1.
struct convex_eq
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c[0] = x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }

    [[nodiscard]] Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-10.0, -10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{10.0, 10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{2.0, -3.0};
    }
};

// min  x0^2 + x1^2   s.t.  x0 + x1 - 1 = 0 (equality),  x0 - 0.7 >= 0 (ineq),
//                          x in [-10, 10]^2.
// The equality is solved by the outer augmented Lagrangian; the inequality
// is handled natively by the inner solver. With the inequality active the
// KKT point is x* = (0.7, 0.3), f* = 0.58.
struct convex_eq_ineq
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    // Equality row first, then the inequality row (argmin convention:
    // c_ineq >= 0 is feasible).
    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c[0] = x[0] + x[1] - 1.0;
        c[1] = x[0] - 0.7;
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J(0, 0) = 1.0; J(0, 1) = 1.0;
        J(1, 0) = 1.0; J(1, 1) = 0.0;
    }

    [[nodiscard]] Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-10.0, -10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{10.0, 10.0};
    }

    [[nodiscard]] Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{2.0, -3.0};
    }
};

}

TEST_CASE("auglag_gcmma_equality_test: converges on a pure equality problem",
          "[auglag][gcmma][equality]")
{
    constexpr int Dm = convex_eq::problem_dimension;
    convex_eq problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 80;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-5);

    step_budget_solver solver{augmented_lagrangian_policy<gcmma_policy<Dm>, Dm>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(0.5).margin(1e-4));
    CHECK(solver.constraint_violation() < 1e-5);
    CHECK(solver.state().x[0] == Approx(0.5).margin(1e-3));
    CHECK(solver.state().x[1] == Approx(0.5).margin(1e-3));
}

TEST_CASE("auglag_gcmma_equality_test: equality via AL, inequality via the inner solver",
          "[auglag][gcmma][equality]")
{
    constexpr int Dm = convex_eq_ineq::problem_dimension;
    convex_eq_ineq problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-5);

    step_budget_solver solver{augmented_lagrangian_policy<gcmma_policy<Dm>, Dm>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    // Inequality active at the optimum: x* = (0.7, 0.3), f* = 0.58.
    CHECK(result.objective_value == Approx(0.58).margin(2e-3));
    CHECK(solver.constraint_violation() < 1e-4);
    CHECK(solver.state().x[0] == Approx(0.7).margin(5e-3));
    CHECK(solver.state().x[1] == Approx(0.3).margin(5e-3));
}
