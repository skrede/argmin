#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/hock_schittkowski.h"
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
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;

    basic_solver solver{kraft_slsqp_policy<rosenbrock<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

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
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{kraft_slsqp_policy<equality_quadratic::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

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
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{kraft_slsqp_policy<ineq_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

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
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{kraft_slsqp_policy<box_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

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
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-15);
    opts.max_iterations = 500;

    basic_solver solver{kraft_slsqp_policy<liepp_box_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

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

    [[maybe_unused]] basic_solver solver{kraft_slsqp_policy<>{}, problem, x0, opts};
    CHECK(true);
}

TEST_CASE("kraft_slsqp step solve step_n", "[kraft_slsqp]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;

    SECTION("step returns finite values")
    {
        basic_solver solver{kraft_slsqp_policy<rosenbrock<>::problem_dimension>{}, problem, x0, opts};

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
        basic_solver solver{kraft_slsqp_policy<rosenbrock<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.step_n(50);
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.iterations <= 50);
    }

    SECTION("solve runs to convergence")
    {
        basic_solver solver{kraft_slsqp_policy<rosenbrock<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached));
    }
}

TEST_CASE("kraft_slsqp converges on HS007 equality", "[kraft_slsqp]")
{
    hs007 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{kraft_slsqp_policy<hs007<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-std::sqrt(3.0)).margin(0.01));
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("kraft_slsqp converges on HS039 equality", "[kraft_slsqp]")
{
    hs039 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{kraft_slsqp_policy<hs039<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(0.01));
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("kraft_slsqp converges on HS040 equality", "[kraft_slsqp]")
{
    hs040 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{kraft_slsqp_policy<hs040<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-0.25).margin(0.01));
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("kraft_slsqp converges on HS026 (regression guard)", "[kraft_slsqp][regression]")
{
    hs026 problem;
    auto x0 = problem.initial_point();

    // Baseline (post-phase30, bench defaults): 20 iters, acc 2.90e-07.
    // Post-phase31 regressed to 12 iters, acc 1.57e-04 (premature
    // ftol at primal_eq = 2.15e-03). The Full E-measure fix blocks
    // that premature ftol; the solver continues past iter 12 and
    // reaches the correct neighborhood of the optimum.
    //
    // Note on iter count: with Philosophy-B-leak closed, the stale-
    // multiplier / QP-at-box-bound tail-drift (deferred to Phase
    // 31.2 per DIAGNOSIS.md) prevents ftol from firing tightly
    // within 20 iters here -- the solver runs past the baseline
    // iter count and descends slowly toward the optimum. The
    // important invariant is objective accuracy, not the iter count.
    //
    // Reference: N&W 2e Definition 12.1 full E-measure closure.
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{kraft_slsqp_policy<hs026<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS026 optimum: f* = 0 at (1, 1, 1).
    // Measured post-31.1 accuracy: 2.90e-07. Threshold 1e-6 is ~3x
    // the measured value -- tight enough to detect a ~10x degradation
    // from the current implementation while still giving numerical
    // headroom for run-to-run floating-point noise.
    //
    // No lower bound on iterations: the current iter count of ~20 is
    // inflated by the BFGS tail-drift documented at kraft_slsqp_test
    // line 380-385 and deferred to a follow-up plan. When that fix
    // lands, the iter count is expected to drop, and that is NOT a
    // regression -- a regression guard must not require slow
    // convergence. The upper bound alone protects against premature
    // termination at a non-optimum iterate.
    CHECK(result.objective_value < 1e-6);
    CHECK(solver.constraint_violation() < 1e-3);
    CHECK(result.iterations <= 50);
}

TEST_CASE("kraft_slsqp HS071 mixed constraints (regression guard)",
          "[kraft_slsqp][regression]")
{
    // HS071: n=4, 1 equality (x1^2+x2^2+x3^2+x4^2 = 40), 1 inequality
    // (x1*x2*x3*x4 >= 25), 1 <= xi <= 5.
    // x0 = (1, 5, 5, 1), f* = 17.0140173 at (1, 4.7430, 3.8211, 1.3794).
    //
    // Reference: NLopt LD_SLSQP converges in ~6 iterations to 1e-10.
    // nablapp's kraft_slsqp reaches the same iteration-count order
    // (~8 iters on the benchmark, ~50 iters to reach the tight
    // margin) with the Kraft LSQ/LSEI QP solver and dense BFGS
    // introduced in plan 24.1-02. The tight 1e-4 margin that NLopt
    // achieves is deferred; this test locks in the current
    // nablapp baseline: HS071 must not diverge and must stay near
    // the optimum with a modest feasibility violation.
    //
    // Plan 24.1-02 requirement KRAFT-QP-03 (HS071 without feasibility
    // restoration). The gap to NLopt's tight 1e-10 precision is
    // tracked for a follow-up plan that will tighten the BFGS
    // restart heuristic against NLopt slsqp.c lines 1571+.
    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-10);

    basic_solver solver{kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(17.0140173).margin(0.1));
    CHECK(solver.constraint_violation() < 0.05);
    CHECK(result.x[0] >= 1.0 - 1e-6);
    CHECK(result.x[0] <= 5.0 + 1e-6);
    CHECK(result.x[1] >= 1.0 - 1e-6);
    CHECK(result.x[1] <= 5.0 + 1e-6);
}

// kraft_slsqp populates step_result::kkt_residual on every accepted
// step via detail::kkt_residual. Gradient-aware constrained policies
// must carry a KKT residual so objective_tolerance_criterion gates
// stationarity on the true KKT error rather than the heterogeneous
// gradient_norm proxy.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
TEST_CASE("kraft_slsqp populates kkt_residual", "[kraft_slsqp][kkt]")
{
    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-6);

    basic_solver solver{kraft_slsqp_policy<hs071<>::problem_dimension>{},
                        problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 10; ++i)
    {
        auto sr = solver.step();
        if(sr.kkt_residual.has_value())
        {
            populated = true;
            CHECK(sr.kkt_residual.value() >= 0.0);
        }
        if(sr.policy_status)
            break;
    }
    CHECK(populated);
}

// HS006 regression guard locking the Phase 31.1 closure.
//
// Baseline (post-phase30): 7 iters, acc 9.24e-08. Post-phase31
// regressed to 6 iters, acc 2.16e-06. The Full E-measure fix keeps
// ftol from firing prematurely; the healthy trajectory lands at
// f < 1e-5 in [6, 8] iters.
//
// Reference: N&W 2e Definition 12.1 full E-measure closure.
TEST_CASE("kraft_slsqp HS006 accuracy guard",
          "[kraft_slsqp][regression]")
{
    hs006 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{kraft_slsqp_policy<hs006<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    // Baseline harness: 9 iters under 1e-15 thresholds; bench
    // defaults land similar or tighter. Full E-measure blocks
    // premature ftol that post-phase31 let fire at iter 6.
    // Measured post-31.1 accuracy: 9.24e-08. Threshold 1e-6 is ~10x
    // the measured value -- tight enough to catch a ~10x degradation
    // while tolerating normal floating-point noise.
    CHECK(result.objective_value < 1e-6);
    CHECK(result.iterations >= 6);
    CHECK(result.iterations <= 12);
}
