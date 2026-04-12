#include "nablapp/solver/filter_slsqp_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

static_assert(nablapp::nlp_solver<nablapp::basic_solver<nablapp::filter_slsqp_policy<>>>);

TEST_CASE("filter_slsqp on hock-schittkowski problems", "[hs][filter_slsqp]")
{
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
    }

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
    }

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
    }

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
    }

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
    }
}
