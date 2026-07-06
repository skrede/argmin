// BOBYQA bound-safety invariant: the objective is NEVER evaluated outside the
// box [xl, xu]. This is a core BOBYQA guarantee -- objectives defined only on
// the box (log, sqrt, reciprocal) trap or return NaN if probed outside it, so
// the bootstrap must place its 2n coordinate perturbations inside the bounds
// even when x0 sits within one interpolation radius of a bound (the F7 case) or
// the box is too tight for the requested radius.
//
// The instrumented objective records any out-of-bounds probe; the test then
// asserts none occurred.
//
// Reference: Powell, M. J. D. (2009), "The BOBYQA algorithm for bound
//            constrained optimization without derivatives", DAMTP 2009/NA06.

#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/options.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace argmin;

namespace
{

// Quadratic with an interior minimum at (10, -3); coordinate 1 is unbounded,
// coordinate 0 has a wide finite range -- exercises mixed-bound scaling.
struct mixed_bound_problem
{
    static constexpr int problem_dimension = dynamic_dimension;
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const
    {
        double a = x[0] - 10.0;
        double b = x[1] + 3.0;
        return a * a + b * b;
    }
    Eigen::VectorXd lower_bounds() const
    {
        Eigen::VectorXd lb(2);
        lb << -100.0, -std::numeric_limits<double>::infinity();
        return lb;
    }
    Eigen::VectorXd upper_bounds() const
    {
        Eigen::VectorXd ub(2);
        ub << 100.0, std::numeric_limits<double>::infinity();
        return ub;
    }
};

// A quadratic that is only "defined" on the box: any evaluation strictly
// outside [lb, ub] (beyond a rounding tolerance) is recorded as a domain
// breach. mutable through a pointer so const value() can record.
struct boxed_objective
{
    static constexpr int problem_dimension = dynamic_dimension;
    int n{2};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;
    Eigen::VectorXd center;
    int* breaches{nullptr};
    double tol{1e-12};

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        for(int i = 0; i < n; ++i)
        {
            if(x[i] < lb[i] - tol || x[i] > ub[i] + tol)
            {
                if(breaches) ++(*breaches);
            }
        }
        return (x - center).squaredNorm();
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

TEST_CASE("bobyqa never evaluates the objective outside the box with a near-bound start",
          "[bobyqa][bounds]")
{
    int breaches = 0;

    boxed_objective problem;
    problem.n = 2;
    problem.lb = Eigen::VectorXd(2);
    problem.ub = Eigen::VectorXd(2);
    problem.lb << 0.0, 0.0;
    problem.ub << 1.0, 1.0;
    problem.center = Eigen::VectorXd(2);
    problem.center << 0.5, 0.5;
    problem.breaches = &breaches;

    // x0 sits a hair inside the lower bound on both coordinates -- well within
    // one auto interpolation radius (0.1 * range = 0.1) of the bound. A naive
    // +rhobeg perturbation on x0[0] = 1e-6 would step to 0.1000001 (fine) but
    // x0 near the UPPER bound would overshoot; place one coordinate near each.
    Eigen::VectorXd x0(2);
    x0 << 1e-6, 1.0 - 1e-6;

    solver_options opts;
    opts.max_iterations = 300;

    basic_solver solver{bobyqa_policy<dynamic_dimension>{}, problem, x0, opts};
    auto result = solver.solve();

    INFO("out-of-bounds evaluations: " << breaches);
    CHECK(breaches == 0);
    // Sanity: it still makes progress toward the interior minimum (0.5, 0.5).
    CHECK(result.objective_value < 0.5);
}

TEST_CASE("bobyqa repairs a box tighter than twice the requested radius",
          "[bobyqa][bounds]")
{
    int breaches = 0;

    boxed_objective problem;
    problem.n = 2;
    problem.lb = Eigen::VectorXd(2);
    problem.ub = Eigen::VectorXd(2);
    problem.lb << 0.0, 0.0;
    problem.ub << 1.0, 1.0;
    problem.center = Eigen::VectorXd(2);
    problem.center << 0.5, 0.5;
    problem.breaches = &breaches;

    Eigen::VectorXd x0(2);
    x0 << 0.5, 0.5;

    // Request an initial radius (0.8) larger than half the box range (0.5), so
    // an unrepaired +/- rhobeg perturbation from the center would step outside
    // [0, 1]. The clamp must shrink it to <= 0.5 * range.
    bobyqa_policy<dynamic_dimension>::options_type popts;
    popts.initial_trust_radius = 0.8;

    solver_options opts;
    opts.max_iterations = 300;

    basic_solver solver{bobyqa_policy<dynamic_dimension>{popts}, problem, x0, opts};
    auto result = solver.solve();

    INFO("out-of-bounds evaluations: " << breaches);
    CHECK(breaches == 0);
    CHECK(result.objective_value < 1e-6);
}

// Coherent mixed finite/infinite bound scaling: an unbounded coordinate must
// not be suppressed just because another coordinate has a wide finite range.
// It shares the reference scale of the widest bounded variable instead of
// collapsing to 1 / max_range.
TEST_CASE("bobyqa scales mixed finite/infinite bounds coherently", "[bobyqa][bounds]")
{
    mixed_bound_problem problem;
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;
    solver_options opts;

    bobyqa_policy<dynamic_dimension> pol;
    auto s = pol.init(problem, x0, opts);

    // The bounded variable (range 200) and the unbounded variable share the
    // reference scale, so both normalize to 1.0 -- the unbounded coordinate is
    // NOT suppressed to 1/200.
    CHECK(s.scale[0] == Approx(1.0));
    CHECK(s.scale[1] == Approx(1.0));

    // And it converges to the interior optimum on the unbounded coordinate.
    opts.max_iterations = 400;
    basic_solver solver{bobyqa_policy<dynamic_dimension>{}, problem, x0, opts};
    auto result = solver.solve();
    CHECK(result.objective_value < 1e-6);
}
