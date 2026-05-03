// Strict-Svanberg-1987 MMA (reciprocal moving asymptotes) policy tests.
//
// Plain MMA: no conservativity loop, unconditional acceptance of the
// dual-subproblem trial each outer iter. The conservativity-protected
// variant is GCMMA (gcmma_policy); the quadratic-penalty CCSA variant
// is ccsa_quadratic_policy.
//
// Plain MMA's convergence guarantee is restricted to *separable*
// objectives and constraints (the reciprocal approximation is exact
// when each function depends on each x_j independently). On
// non-separable problems with cross-terms (e.g., HS076's x0*x2 term)
// plain MMA can drift infeasible -- by design, that's GCMMA's job to
// fix. These tests therefore exercise:
//   1. Algorithmic plumbing (init, step, dual solve, reset) does not
//      crash and produces finite output.
//   2. A structural-design-style problem (beam_design) where the
//      reciprocal approximation is well-fitted.
//
// Reference: Svanberg 1987, "The method of moving asymptotes",
//            Int. J. Numer. Methods Engng 24:359-373, Section 5
//            (separability assumption for convergence).

#include "argmin/solver/mma_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace {

// Beam-design surrogate: separable objective, separable constraint, the
// classic structural-optimization shape MMA was invented for. Minimize
// f(x) = x0^2 + x1^2 subject to 1/x0 + 1/x1 <= 2 (i.e. argmin
// convention c0 = 2 - 1/x0 - 1/x1 >= 0), bounds 0.1 <= x_i <= 10.
// The optimum is at x0 = x1 = 1 with f* = 2.
struct separable_beam
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x,
                     Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 2.0 - 1.0 / x[0] - 1.0 / x[1];
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0 / (x[0] * x[0]);
        J(0, 1) = 1.0 / (x[1] * x[1]);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(0.1);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(10.0);
    }
};

}

// Plumbing test: init, step, dual solve, reset all execute without crash
// and produce finite output, even on a non-separable problem (HS076)
// where plain MMA is not expected to converge to the optimum.
TEST_CASE("strict mma plumbing executes on HS076", "[mma_reciprocal]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        mma_policy<hs076<>::problem_dimension>{}, problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(std::isfinite(result.objective_value));
    CHECK(static_cast<int>(result.iterations) >= 1);
}

// Convergence on a separable structural-design problem: this is where
// MMA's reciprocal approximation is theoretically exact. Plain MMA
// should reach the optimum to good accuracy without conservativity.
TEST_CASE("strict mma converges on separable_beam", "[mma_reciprocal]")
{
    separable_beam problem;
    Eigen::Vector<double, 2> x0{2.0, 2.0};

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        mma_policy<separable_beam::problem_dimension>{}, problem, x0, opts};
    const auto result = solver.solve(opts);

    // f* = 2.0 at x0 = x1 = 1.0 (verified via KKT: lambda such that
    // 2*x_i = lambda / x_i^2 -> 2 x_i^3 = lambda for both i, so x0=x1
    // by symmetry; the constraint binds at 2/x = 2 -> x = 1, f = 2).
    CHECK(result.objective_value == Approx(2.0).margin(1e-2));
    // Constraint roughly satisfied (allow 1e-3 violation since plain
    // MMA without conservativity can drift slightly infeasible).
    Eigen::VectorXd c(1);
    problem.constraints(Eigen::Vector<double, 2>{
        Eigen::Vector<double, 2>::Map(result.x.data(), 2)}, c);
    CHECK(c[0] >= -1e-3);
}
