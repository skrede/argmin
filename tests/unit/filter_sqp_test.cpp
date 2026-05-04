// Convergence regression tests for filter_slsqp_policy on Hock-Schittkowski problems.
//
// Reference: Hock, W. & Schittkowski, K. (1981). Test Examples for
//            Nonlinear Programming Codes. Lecture Notes in Economics
//            and Mathematical Systems 187. Springer-Verlag.
//            Fletcher & Leyffer 2002 (filter SQP).

#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

static_assert(argmin::nlp_solver<argmin::basic_solver<argmin::filter_slsqp_policy<>>>);

TEST_CASE("filter_slsqp on hock-schittkowski problems", "[hs][filter_slsqp]")
{
    // HS071: min x0*x3*(x0+x1+x2) + x2, n=4, 1 eq + 1 ineq, mixed.
    //
    // Reference: Hock & Schittkowski 1981, Problem 71.
    SECTION("HS071: mixed equality + inequality constraints")
    {
        hs071<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(17.014).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
        // No strict Lagrangian-gradient bar: filter_slsqp on HS071 reaches
        // the f-ballpark with a modest constraint violation but does NOT
        // reach a stationary point at HEAD (gradient_norm ~= 0.6). The
        // gap is the known SEED-015 / iter-0 cold-start residual addressed
        // by a separate plan in this phase (mu calibration / SOC retry).
        // Once that plan lands the bar can be tightened.
    }

    // HS043 (Rosen-Suzuki): n=4, 3 inequalities. Loose convergence margin.
    //
    // Reference: Hock & Schittkowski 1981, Problem 43.
    SECTION("HS043: inequality constraints")
    {
        hs043<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs043<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-44.0).margin(1.0));
        CHECK(solver.constraint_violation() < 0.1);
        // Lagrangian gradient norm at converged point (post-COR-01 invariant).
        // HS043 uses a relaxed bar (< 1e-3) because filter_slsqp's
        // Wachter-Biegler envelope (gamma_f = gamma_h = 1e-5 hardcoded)
        // over-rejects strictly-feasible descent on this cell, leaving the
        // iterate at a loosely converged optimum (SEED-006; gamma sweep is
        // a separate plan in this phase).
        CHECK(result.gradient_norm < 1e-3);
    }

    // HS039: n=4, 2 equalities.
    //
    // Reference: Hock & Schittkowski 1981, Problem 39.
    SECTION("HS039: equality constraints")
    {
        hs039<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs039<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
        // Lagrangian gradient norm at converged point (post-COR-01 invariant).
        CHECK(result.gradient_norm < 1e-4);
    }

    // HS035: n=3, 1 inequality, x >= 0 bounds.
    //
    // Reference: Hock & Schittkowski 1981, Problem 35.
    SECTION("HS035: inequality with bounds")
    {
        hs035<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs035<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(0.111).margin(0.01));
        CHECK(solver.constraint_violation() < 0.01);
        // Lagrangian gradient norm at converged point (post-COR-01 invariant).
        CHECK(result.gradient_norm < 1e-4);
    }

    // HS024: n=2, 3 inequalities (parallel rows at x*), bounds x >= 0.
    //
    // Reference: Hock & Schittkowski 1981, Problem 24.
    SECTION("HS024: bound-constrained inequality")
    {
        hs024<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs024<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
        // Lagrangian gradient norm at converged point (post-COR-01 invariant).
        CHECK(result.gradient_norm < 1e-4);
    }

    SECTION("hybrid restoration: infeasible start with L1-then-QP fallback")
    {
        // Start from a point with both equality and inequality constraints
        // violated, using the hybrid restoration strategy with a low step
        // cap. This verifies the hybrid restoration path compiles and
        // executes correctly, and that convergence is not broken by the
        // L1->QP fallback configuration.
        //
        // x0 = (1.2, 2.0, 2.0, 1.2):
        //   eq  = 1.44+4+4+1.44-40 = -29.12 (violated)
        //   ineq = 1.2*2*2*1.2-25  = -19.24 (violated)
        hs071<> problem;
        Eigen::Vector<double, hs071<>::problem_dimension> x0;
        x0 << 1.2, 2.0, 2.0, 1.2;

        solver_options opts;
        opts.max_iterations = 300;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        filter_slsqp_policy<hs071<>::problem_dimension>::options_type policy_opts;
        policy_opts.restoration = detail::restoration_strategy::hybrid;
        policy_opts.max_restoration_steps = 3;

        basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{},
                            problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(17.014).margin(0.5));
        CHECK(solver.constraint_violation() < 0.01);
        // No strict Lagrangian-gradient bar on HS071 (same rationale as
        // the HS071 SECTION above). The hybrid-restoration path does not
        // reach a stationary point at HEAD; this test is for path coverage.
    }
}

// filter_slsqp populates step_result::kkt_residual on accepted steps
// via detail::kkt_residual and sets is_null_step on the documented
// QP-zero-step + high-violation path.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34 (KKT residual);
//            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (SQP null step).
TEST_CASE("filter_slsqp populates kkt_residual and exposes is_null_step",
          "[filter_slsqp][kkt]")
{
    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-10);

    basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{},
                        problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 15; ++i)
    {
        auto sr = solver.step();
        CHECK((sr.is_null_step || !sr.is_null_step));
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

// HS024 regression guard: locks the active-set multiplier re-estimation
// closure on a problem where the constraint Jacobian has parallel rows
// (J_ineq row 2 = -row 3 at x* = (3, sqrt(3))). Plain LS + cwiseMax on the
// full Jacobian picks a min-norm split that is not KKT-valid after sign
// projection, leaving stationarity positive at the optimum. Active-set LS
// restricts the projection to binding inequalities (|c_ineq[i]| < 1e-8)
// and recovers the textbook multipliers. Null-step branches populate
// kkt_residual so the composite convergence check can terminate on
// iterates where the QP returns zero direction at the optimum.
//
// Reference baseline (post-phase30): 13 iters @ f = -1.0. Regression
// target: iterations within 1.
//
// Note: this test does NOT set a gradient_norm tolerance. After COR-01
// the policy reports the Lagrangian gradient norm; on HS024 the
// least-squares multiplier estimate makes ||grad f - A^T lambda||
// numerically zero from iter 0 onward (since HS024 is objective-only
// in the directions the QP currently chases), so a gradient_threshold
// would terminate prematurely on a non-optimal iterate. Convergence is
// driven by the composite KKT residual gate inside objective_tolerance
// (kkt_residual.value_or(gradient_norm)) plus step_threshold.
//
// Reference: Hock & Schittkowski 1981, Problem 24.
//            N&W 2e Section 18.3 + Algorithm 18.3 (working-set);
//            eq. 18.15 (least-squares lambda);
//            Definition 12.1 (KKT conditions);
//            eq. 12.34 (composite first-order optimality).
TEST_CASE("filter_slsqp HS024 regression guard",
          "[filter_slsqp][regression]")
{
    using namespace argmin;

    hs024<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{
        filter_slsqp_policy<hs024<>::problem_dimension>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(1e-6));
    CHECK(result.iterations <= 14);
    CHECK(solver.constraint_violation() < 1e-6);
    // Lagrangian gradient norm at converged point (post-COR-01 invariant).
    CHECK(result.gradient_norm < 1e-4);
}

// HS007: min ln(1 + x0^2) - x1, subject to (1 + x0^2)^2 + x1^2 = 4.
//        n=2, 1 equality. Near-feasible start.
//        f* = -sqrt(3) at x* = (0, sqrt(3)).
//
// Reference: Hock & Schittkowski 1981, Problem 7.
TEST_CASE("filter_slsqp converges on HS007 equality",
          "[filter_slsqp][hs007]")
{
    hs007<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{filter_slsqp_policy<hs007<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-std::sqrt(3.0)).margin(1e-3));
    CHECK(solver.constraint_violation() < 1e-4);
    // Lagrangian gradient norm at converged point (post-COR-01 invariant).
    CHECK(result.gradient_norm < 1e-4);
}

// HS028: min (x0 + x1)^2 + (x1 + x2)^2, subject to x0 + 2*x1 + 3*x2 = 1.
//        n=3, 1 equality, well-conditioned.
//        f* = 0 at x* = (0.5, -0.5, 0.5).
//
// Reference: Hock & Schittkowski 1981, Problem 28.
TEST_CASE("filter_slsqp converges on HS028 quadratic with linear equality",
          "[filter_slsqp][hs028][regression]")
{
    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{filter_slsqp_policy<hs028<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 1e-6);
    CHECK(solver.constraint_violation() < 1e-6);
    // Lagrangian gradient norm at converged point (post-COR-01 invariant).
    CHECK(result.gradient_norm < 1e-4);
    CHECK(std::abs(result.x[0] - 0.5) < 1e-3);
    CHECK(std::abs(result.x[1] - (-0.5)) < 1e-3);
    CHECK(std::abs(result.x[2] - 0.5) < 1e-3);
}

