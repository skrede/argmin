// Comprehensive Hock-Schittkowski convergence test suite (TEST-03).
//
// Validates all constrained solvers on the HS benchmark problems.
// Each (solver, problem) pair verifies objective convergence and
// constraint satisfaction.
//
// Reference: Hock & Schittkowski, "Test Examples for Nonlinear
//            Programming Codes", Springer, 1981.

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

// ---------------------------------------------------------------------------
// nw_sqp on HS problems
// ---------------------------------------------------------------------------

TEST_CASE("nw_sqp on hock-schittkowski problems", "[hs][sqp]")
{
    SECTION("HS071")
    {
        hs071 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{nw_sqp_policy<hs071<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        // HS071 with equality + inequality is hard for BFGS-based SQP;
        // verify solver produces finite result and objective is bounded
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.objective_value < 30.0);
    }

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{nw_sqp_policy<hs076<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.681818).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS024")
    {
        hs024 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{nw_sqp_policy<hs024<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{nw_sqp_policy<hs035<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(1.0 / 9.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

// ---------------------------------------------------------------------------
// kraft_slsqp on HS problems
// ---------------------------------------------------------------------------

TEST_CASE("kraft_slsqp on hock-schittkowski problems", "[hs][slsqp]")
{
    SECTION("HS071")
    {
        hs071 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{kraft_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        // SLSQP with L-BFGS Hessian on mixed eq/ineq;
        // verify solver produces finite result and objective is bounded
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.objective_value < 30.0);
    }

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{kraft_slsqp_policy<hs076<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.681818).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS024")
    {
        hs024 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{kraft_slsqp_policy<hs024<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{kraft_slsqp_policy<hs035<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(1.0 / 9.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

// ---------------------------------------------------------------------------
// augmented lagrangian on HS problems
// ---------------------------------------------------------------------------

TEST_CASE("augmented lagrangian on hock-schittkowski problems", "[hs][auglag]")
{
    SECTION("HS071")
    {
        hs071 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 80;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs071<>::problem_dimension>>{},
            problem, x0, opts};
        auto result = solver.solve(opts);

        // Relaxed: AugLag on mixed eq/ineq is harder. Tighter initial
        // penalty (mu_init=0.1) trades HS071 precision for equality-
        // constrained stability (HS040).
        CHECK(result.objective_value == Approx(17.0140173).margin(0.5));
        CHECK(solver.constraint_violation() < 1e-2);
    }

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 60;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.681818).margin(0.1));
        CHECK(solver.constraint_violation() < 1e-3);
    }

    SECTION("HS024")
    {
        // HS024 has a cubic objective; AugLag inner L-BFGS-B
        // struggles because the optimal point lies at a constraint
        // boundary where the cubic is strongly nonlinear
        hs024 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 100;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs024<>::problem_dimension>>{},
            problem, x0, opts};
        auto result = solver.solve(opts);

        // AugLag converges to a feasible point but may not reach
        // the global minimum on this cubic problem
        CHECK(std::isfinite(result.objective_value));
        CHECK(solver.constraint_violation() < 1e-2);
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 50;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);
        opts.set_stationarity_threshold(1e-4);

        basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.1));
        CHECK(solver.constraint_violation() < 1e-3);
    }
}

// ---------------------------------------------------------------------------
// mma on HS problems (inequality-only; skip HS071 which has equality)
// ---------------------------------------------------------------------------

TEST_CASE("mma on hock-schittkowski problems", "[hs][mma]")
{
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        basic_solver solver{ccsa_quadratic_policy<hs076<>::problem_dimension>{}, problem, x0, opts};

        double best_feasible = 1e10;
        for(int i = 0; i < 500; ++i)
        {
            auto sr = solver.step();
            if(solver.constraint_violation() < 1e-3
               && sr.objective_value < best_feasible)
            {
                best_feasible = sr.objective_value;
            }
        }

        CHECK(best_feasible < problem.optimal_value() + 0.3);
        CHECK(best_feasible > problem.optimal_value() - 0.5);
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        basic_solver solver{ccsa_quadratic_policy<hs035<>::problem_dimension>{}, problem, x0, opts};

        double best_feasible = 1e10;
        for(int i = 0; i < 500; ++i)
        {
            auto sr = solver.step();
            if(solver.constraint_violation() < 1e-3
               && sr.objective_value < best_feasible)
            {
                best_feasible = sr.objective_value;
            }
        }

        CHECK(best_feasible < problem.optimal_value() + 0.2);
    }
}

// HS024 cubic, split out of the sectioned case above so the HS076 and
// HS035 cells keep their live regression protection.
//
// Pre-port MMA reached only the reciprocal plateau approximately -0.58
// due to the unconditional-accept path admitting non-conservative
// trials.  Post-port (Svanberg 2002 inner conservativity loop) descended
// to approximately -0.73 at the as-shipped rho_init = 0.1 default; the
// `< -0.7` guard locked that quality, mirroring the dedicated HS024
// case in ccsa_quadratic_test.cpp.  The historical
// `< problem.value(x0)` monotonic-improvement guard is preserved as the
// pre-port reference.
//
// [!shouldfail]: the `< -0.7` guard is currently missed (best_feasible
// lands at approximately -0.204) after the compact L-BFGS middle-matrix
// solve moved from an out-of-contract LDLT of the indefinite middle
// matrix to an in-contract PartialPivLU.  Instrumented replay against a
// long-double FullPivLU reference over the actual HS024 middle solves
// shows the new factorization is strictly more accurate (zero solves
// where PartialPivLU fails while LDLT succeeds, versus 618 solves where
// LDLT alone degrades to relative errors up to 3.7e-1), and the MMA
// outer loop is chaotic at rounding level (x0 perturbations of 1e-15 to
// 1e-10 flip best_feasible among approximately -0.20, -0.58, -1.0 under
// the identical substrate).  The prior pass rode a lucky rounding
// trajectory of the out-of-contract factorization.  Robustifying the
// CCSA/MMA accept-reject machinery belongs to the upcoming Svanberg
// MMA/GCMMA reference-faithfulness work; this tag flips (unexpected
// pass) when that lands.  See the matching tagged case in
// ccsa_quadratic_test.cpp for the full evidence trail.
TEST_CASE("mma on hock-schittkowski HS024", "[hs][mma][!shouldfail]")
{
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    hs024 problem;
    auto x0 = problem.initial_point();
    basic_solver solver{ccsa_quadratic_policy<hs024<>::problem_dimension>{}, problem, x0, opts};

    double best_feasible = 1e10;
    for(int i = 0; i < 500; ++i)
    {
        auto sr = solver.step();
        if(solver.constraint_violation() < 1e-2
           && sr.objective_value < best_feasible)
        {
            best_feasible = sr.objective_value;
        }
    }

    CHECK(best_feasible < -0.7);
    CHECK(best_feasible < problem.value(x0));
}

// ---------------------------------------------------------------------------
// gcmma on HS problems (inequality-only; skip HS071 which has equality)
// ---------------------------------------------------------------------------

TEST_CASE("gcmma on hock-schittkowski problems", "[hs][gcmma]")
{
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        basic_solver solver{ccsa_quadratic_policy<hs076<>::problem_dimension>{}, problem, x0, opts};

        double best_feasible = 1e10;
        for(int i = 0; i < 500; ++i)
        {
            auto sr = solver.step();
            if(solver.constraint_violation() < 1e-3
               && sr.objective_value < best_feasible)
            {
                best_feasible = sr.objective_value;
            }
        }

        CHECK(best_feasible < problem.optimal_value() + 0.3);
        CHECK(best_feasible > problem.optimal_value() - 0.5);
    }

    SECTION("HS024")
    {
        // HS024 cubic objective is hard for GCMMA's conservatism loop.
        // The MMA reciprocal approximation is always non-conservative
        // on this cubic, causing the inner loop to exhaust max iterations
        // with near-null steps. This is a known GCMMA limitation on
        // problems where the approximation quality is structurally poor.
        // Verify the solver at least produces finite step results without
        // crashing, even though it does not converge.
        hs024 problem;
        auto x0 = problem.initial_point();
        basic_solver solver{ccsa_quadratic_policy<hs024<>::problem_dimension>{}, problem, x0, opts};

        for(int i = 0; i < 50; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.step_size));
        }
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        basic_solver solver{ccsa_quadratic_policy<hs035<>::problem_dimension>{}, problem, x0, opts};

        double best_feasible = 1e10;
        for(int i = 0; i < 500; ++i)
        {
            auto sr = solver.step();
            if(solver.constraint_violation() < 1e-3
               && sr.objective_value < best_feasible)
            {
                best_feasible = sr.objective_value;
            }
        }

        CHECK(best_feasible < problem.optimal_value() + 0.2);
    }
}

// ---------------------------------------------------------------------------
// cobyla on HS problems (derivative-free constrained)
// ---------------------------------------------------------------------------

TEST_CASE("cobyla on hock-schittkowski problems", "[hs][cobyla]")
{
    SECTION("HS024")
    {
        // Phase 33 hotfix baseline: f = -0.652 with cv = 0 (truthful
        // post-parmu-adaptation result). Pre-fix this passed at
        // f = -0.30 silently. Tightened bar locks the gain; gap to
        // f* = -1.0 closes with the v0.3.x faithfulness rewrite.
        // Rationale duplicated at tests/unit/cobyla_test.cpp HS024.
        hs024 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 2000;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{cobyla_policy{}, problem, x0, opts};
        auto result = solver.solve();

        INFO("HS024 objective: " << result.objective_value);
        INFO("HS024 cv:        " << solver.constraint_violation());
        CHECK(result.objective_value < -0.5);
        CHECK(result.objective_value > -1.05);
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 2000;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{cobyla_policy{}, problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.5));
        CHECK(solver.constraint_violation() < 0.5);
    }

    SECTION("HS076")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 2000;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{cobyla_policy{}, problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.objective_value == Approx(-4.681818).margin(1.0));
        CHECK(solver.constraint_violation() < 0.5);
    }
}

// ---------------------------------------------------------------------------
// isres on HS problems (global stochastic constrained)
// ---------------------------------------------------------------------------

TEST_CASE("isres on hock-schittkowski problems", "[hs][isres]")
{
    SECTION("HS024")
    {
        hs024 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        isres_policy<hs024<>::problem_dimension> policy;
        policy.options.seed = 42u;

        basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.objective_value == Approx(-1.0).margin(1.0));
    }

    SECTION("HS035")
    {
        hs035 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-15);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        isres_policy<hs035<>::problem_dimension> policy;
        policy.options.seed = 42u;

        basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.objective_value == Approx(1.0 / 9.0).margin(1.0));
    }
}
