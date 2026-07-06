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

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace argmin;

namespace
{

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
