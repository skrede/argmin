#include "oracle_problems.h"

#include "oracle_tolerances.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <initializer_list>

using namespace argmin;
using namespace argmin::oracle::fixtures;

namespace
{

solver_options<> hs071_options()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);
    return opts;
}

solve_result<double, 4> solve_hs071()
{
    hs071<> problem;
    const solver_options<> opts = hs071_options();
    step_budget_solver solver{kraft_slsqp_policy<double, 4>{}, problem, problem.initial_point(), opts};
    return solver.solve(opts);
}

// A rotation in the (x1, x2) plane keeps x1^2 + x2^2, so the equality row holds while the point leaves
// the optimum along the constraint surface.
solve_result<double, 4> rotated_off_optimum(solve_result<double, 4> result, double distance)
{
    const double radius = std::hypot(result.x[1], result.x[2]);
    const double angle = distance / radius;
    const double x1 = result.x[1];
    const double x2 = result.x[2];
    result.x[1] = std::cos(angle) * x1 - std::sin(angle) * x2;
    result.x[2] = std::sin(angle) * x1 + std::cos(angle) * x2;
    return result;
}

}

TEST_CASE("oracle accepts a real kraft_slsqp solve of hs071", "[oracle]")
{
    const solve_result<double, 4> result = solve_hs071();
    const oracle::verdict<double> v = oracle::judge(hs071<>{}, result, oracle::tolerances_of(hs071_options()));
    CAPTURE(static_cast<int>(result.status), v.failing);
    CHECK(v.claimed != oracle::claim::none);
    CHECK(v.ok);
}

TEST_CASE("the hs071 point kraft_slsqp returns meets the converged claim", "[oracle]")
{
    solve_result<double, 4> result = solve_hs071();
    result.status = solver_status::converged;
    const oracle::verdict<double> v = oracle::judge(hs071<>{}, result, oracle::tolerances_of(hs071_options()));
    CAPTURE(v.failing, v.legs.stationarity, v.legs.equality_violation, v.legs.inequality_violation, v.legs.complementarity, v.legs.dual_sign_breach);
    CHECK(v.ok);
}

TEST_CASE("oracle rejects a converged claim moved off the hs071 optimum along a feasible direction", "[oracle]")
{
    solve_result<double, 4> moved = rotated_off_optimum(solve_hs071(), 1e-2);
    moved.status = solver_status::converged;
    const oracle::verdict<double> v = oracle::judge(hs071<>{}, moved, oracle::tolerances_of(hs071_options()));
    CAPTURE(v.legs.stationarity, v.legs.equality_violation, v.legs.inequality_violation);
    CHECK_FALSE(v.ok);
    CHECK(v.failing == "stationarity");
}

TEST_CASE("oracle rejects a converged claim with an equality row broken by 1e-3", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(equality_quadratic{}, planted(solver_status::converged, vec2(0.5005, 0.5005)), unit_tolerances());
    CAPTURE(v.legs.stationarity, v.legs.equality_violation);
    CHECK(v.failing == "equality_violation");
}

TEST_CASE("oracle rejects a converged claim whose recovered inequality multiplier is negative", "[oracle]")
{
    const oracle::verdict<double> v = oracle::judge(linear_halfline{-1.0}, planted(solver_status::converged, vec1(1.0)), unit_tolerances());
    CAPTURE(v.legs.stationarity, v.legs.dual_sign_breach);
    CHECK(v.failing == "dual_sign_breach");
}

TEST_CASE("running claims an unfinished run", "[oracle]")
{
    CHECK(oracle::claim_of(solver_status::running) == oracle::claim::unfinished);
}

TEST_CASE("converged claims optimality", "[oracle]")
{
    CHECK(oracle::claim_of(solver_status::converged) == oracle::claim::optimality);
}

TEST_CASE("ftol_reached claims optimality at the objective criterion's stationarity", "[oracle]")
{
    CHECK(oracle::claim_of(solver_status::ftol_reached) == oracle::claim::ftol_optimality);
}

TEST_CASE("stalled and xtol_reached claim feasibility only", "[oracle]")
{
    for(const solver_status s : {solver_status::stalled, solver_status::xtol_reached})
        CHECK(oracle::claim_of(s) == oracle::claim::feasibility);
}

TEST_CASE("budget and failure statuses make no optimality claim", "[oracle]")
{
    for(const solver_status s : {solver_status::max_iterations, solver_status::budget_exhausted, solver_status::diverged, solver_status::maxeval_reached,
                                 solver_status::roundoff_limited, solver_status::trust_region_step_rejected, solver_status::objective_stalled,
                                 solver_status::time_limit_reached, solver_status::aborted, solver_status::invalid_problem, solver_status::line_search_failed})
        CHECK(oracle::claim_of(s) == oracle::claim::none);
}
