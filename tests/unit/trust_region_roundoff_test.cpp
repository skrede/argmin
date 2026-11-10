#include "argmin/detail/gain_ratio.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

constexpr double eps = std::numeric_limits<double>::epsilon();

// Every stopping test sits below the residual hs076 can attain in double, so only the trust region ends the run.
solver_options<> unattainable_tolerances()
{
    solver_options<> opts{};
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-20);
    opts.set_objective_threshold(1e-20);
    opts.set_step_threshold(1e-20);
    return opts;
}

struct floor_run
{
    solver_status status;
    std::uint32_t iterations;
    std::uint32_t trailing_null_steps;
    double objective_error;
    double violation;
};

template <sqp_mode Mode>
floor_run run_to_floor()
{
    const hs076<> problem;
    step_budget_solver<tr_sqp_policy<double, 4, Mode>, 4, hs076<>> solver(problem, problem.initial_point(), unattainable_tolerances());
    floor_run run{solver_status::running, 0, 0, 0.0, 0.0};
    while(solver.status() == solver_status::running && run.iterations < 200)
    {
        const auto r = solver.step();
        run.trailing_null_steps = r.is_null_step ? run.trailing_null_steps + 1 : 0;
        ++run.iterations;
        run.objective_error = std::abs(r.objective_value - problem.optimal_value());
        run.violation = r.constraint_violation;
    }
    run.status = solver.status();
    return run;
}

void check_floor(const floor_run& run)
{
    CHECK(run.status == solver_status::roundoff_limited);
    CHECK(run.trailing_null_steps == 1);
    CHECK(run.objective_error <= 1e-12);
    CHECK(run.violation <= 1e-12);
}

}

TEST_CASE("a trust-region SQP stops on the first rejection within rounding at the hs076 optimum", "[tr_sqp]")
{
    SECTION("accurate")
    {
        check_floor(run_to_floor<sqp_mode::accurate>());
    }
    SECTION("fast")
    {
        check_floor(run_to_floor<sqp_mode::fast>());
    }
}

TEST_CASE("the trust radius floor is the configured floor or the relative machine precision of x", "[tr_sqp]")
{
    CHECK(detail::trust_radius_floor(1e-12, 2.0) == 1e-12);
    CHECK(detail::trust_radius_floor(1e-12, 1e6) == eps * 1e6);
    CHECK(detail::trust_radius_floor(0.0, 0.5) == eps);
}

TEST_CASE("merit reductions are within rounding only when both lie inside 10 eps of the merit", "[tr_sqp]")
{
    const double merit = -4.0;
    const double rounding = 10.0 * eps * 4.0;
    CHECK(detail::reductions_within_rounding(-rounding, rounding, merit));
    CHECK(detail::reductions_within_rounding(0.0, 0.0, 0.0));
    CHECK_FALSE(detail::reductions_within_rounding(-2.0 * rounding, 0.0, merit));
    CHECK_FALSE(detail::reductions_within_rounding(0.0, -2.0 * rounding, merit));
    CHECK_FALSE(detail::reductions_within_rounding(std::numeric_limits<double>::quiet_NaN(), 0.0, merit));
}
