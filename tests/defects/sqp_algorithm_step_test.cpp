#include "sqp_algorithm_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace argmin;
using defects::nowhere;

namespace
{

// |x - 1|^2 with the inactive row 10 - x0 - x1 >= 0.
defects::row_quadratic bowl_inside_row()
{
    return {2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::Vector2d::Constant(-2.0), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(-1.0, -1.0), Eigen::VectorXd::Constant(1, -10.0), nowhere};
}

// From (1, 1) the step toward x0 = 3 crosses the NaN cutoff x0 > 1 while x1 <= 0.5 is violated by 0.5.
defects::row_quadratic nan_beyond_start_with_violated_row()
{
    return defects::shifted_bowl(Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(0.0, -1.0), Eigen::VectorXd::Constant(1, -0.5), 1.0);
}

step_result<double> feasible_start_ascent_step()
{
    const defects::flipped_gradient problem{bowl_inside_row()};
    step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}};
    return solver.step();
}

// 1e-4 x0 - 1 = 0 with objective (x1 - 1)^2: the row norm squared is 1e-8, the projected one 1e-16.
double violation_after_trust_region_solve(double row_scale)
{
    const defects::row_quadratic problem{
        Eigen::Vector2d(0.0, 2.0).asDiagonal(), Eigen::Vector2d(0.0, -2.0), Eigen::RowVector2d(row_scale, 0.0), Eigen::VectorXd::Ones(1), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), nowhere};
    solver_options<> opts;
    opts.max_iterations = 200;
    step_budget_solver solver{tr_sqp_policy<>{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    const solve_result<double> r = solver.solve(opts);
    return std::abs(row_scale * r.x[0] - 1.0);
}

// 0.5 x^2 - 2x with the inactive row 3 - x >= 0: the exact model step from 0 lands on the minimizer 2.
double trust_region_first_step_endpoint()
{
    const defects::row_quadratic problem{
        Eigen::MatrixXd::Ones(1, 1), Eigen::VectorXd::Constant(1, -2.0), Eigen::MatrixXd(0, 1), Eigen::VectorXd(0), Eigen::MatrixXd::Constant(1, 1, -1.0), Eigen::VectorXd::Constant(1, -3.0), nowhere};
    tr_sqp_policy<> policy;
    policy.options.initial_trust_radius = 10.0;
    step_budget_solver solver{policy, problem, Eigen::VectorXd::Zero(1).eval(), solver_options<>{}};
    solver.step();
    return solver.state().x[0];
}

}

TEST_CASE("filter_nw_sqp starts with a positive minimum filter violation", "[known-defect][!shouldfail]")
{
    const defects::row_quadratic problem = bowl_inside_row();
    step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}};
    CHECK(solver.state().filter.theta_min() > 0.0);
}

TEST_CASE("filter_nw_sqp restores feasibility from a start whose full step enters the NaN region", "[sqp]")
{
    const defects::row_quadratic problem = nan_beyond_start_with_violated_row();
    step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, Eigen::Vector2d(1.0, 1.0).eval(), solver_options<>{}};
    solver.step();
    CHECK(solver.state().x[0] == 1.0);
    CHECK(solver.state().x[1] <= 0.5);
}

TEST_CASE("filter_nw_sqp files the pre-restoration iterate in its filter after a restoration", "[known-defect][!shouldfail]")
{
    const defects::row_quadratic problem = nan_beyond_start_with_violated_row();
    step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, Eigen::Vector2d(1.0, 1.0).eval(), solver_options<>{}};
    solver.step();
    CHECK(solver.state().filter.is_acceptable(-4.5, 0.0));
}

TEST_CASE("filter_nw_sqp takes a null step instead of a zero-length restoration at a feasible iterate", "[known-defect][!shouldfail]")
{
    CHECK(feasible_start_ascent_step().is_null_step);
}

TEST_CASE("tr_sqp reduces a unit-scaled equality violation", "[sqp]")
{
    CHECK(violation_after_trust_region_solve(1.0) < 0.5);
}

TEST_CASE("tr_sqp reduces the violation of an equality row scaled by 1e-4", "[known-defect][!shouldfail]")
{
    CHECK(violation_after_trust_region_solve(1e-4) < 0.5);
}

TEST_CASE("tr_sqp takes the exact model step when its only row is inactive", "[known-defect][!shouldfail]")
{
    CHECK(std::abs(trust_region_first_step_endpoint() - 2.0) <= 1e-6);
}
