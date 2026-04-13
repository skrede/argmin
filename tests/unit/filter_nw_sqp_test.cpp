// Convergence tests for filter_nw_sqp_policy on Hock-Schittkowski problems.
//
// Validates that the filter-based dense BFGS SQP policy converges on
// standard HS benchmark problems with equality, inequality, and mixed
// constraints.
//
// Reference: Hock & Schittkowski 1981; Fletcher & Leyffer 2002.

#include "nablapp/solver/filter_nw_sqp_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

static_assert(nlp_solver<basic_solver<filter_nw_sqp_policy<>>>,
              "filter_nw_sqp_policy must satisfy nlp_solver concept");

TEST_CASE("filter_nw_sqp on hock-schittkowski problems", "[hs][filter_nw_sqp]")
{
    SECTION("HS071: mixed equality + inequality constraints")
    {
        hs071 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs071<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.status != solver_status::max_iterations);
        CHECK(result.objective_value == Approx(17.014).margin(1.0));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS043: inequality constraints only")
    {
        hs043 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs043<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.status != solver_status::max_iterations);
        CHECK(result.objective_value == Approx(-44.0).margin(1.0));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS039: equality constraints only")
    {
        hs039 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs039<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.status != solver_status::max_iterations);
        CHECK(result.objective_value == Approx(-1.0).margin(1.0));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    SECTION("HS076: inequality + equality constraints")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs076<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.68).margin(0.1));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}
