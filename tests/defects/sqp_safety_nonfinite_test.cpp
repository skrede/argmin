#include "sqp_safety_problems.h"

#include "oracle/callback_tracer.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace argmin;

namespace
{

// The bowl minimized at (3, 0) with the inactive row x1 + 100 >= 0; f is NaN for x0 > nan_beyond.
defects::row_quadratic bowl(double nan_beyond)
{
    return defects::shifted_bowl(Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(0.0, 1.0), Eigen::VectorXd::Constant(1, -100.0), nan_beyond);
}

oracle::injection nan_jacobian_at_call(std::uint32_t k)
{
    return {oracle::callback::constraint_jacobian, oracle::non_finite::nan, oracle::injection_mode::kth_call, k};
}

struct boxed_bowl : defects::row_quadratic
{
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -10.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }
};

}

TEMPLATE_TEST_CASE("an SQP reset onto a non-finite start latches, and a reset_clear onto a finite one converges", "[sqp]", tr_sqp_policy<>, kraft_slsqp_policy<>, nw_sqp_policy<>)
{
    const defects::row_quadratic problem = bowl(5.0);
    const solver_options<> opts;
    step_budget_solver solver{TestType{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    CHECK(solver.reset(Eigen::Vector2d(6.0, 0.0).eval()) == solver_status::non_finite_evaluation);
    const solve_result<double> latched = solver.solve(opts);
    CHECK(latched.status == solver_status::non_finite_evaluation);
    CHECK(latched.iterations == 0);
    CHECK(solver.reset_clear(Eigen::Vector2d::Zero().eval()) == solver_status::running);
    const solve_result<double> recovered = solver.solve(opts);
    CHECK(recovered.status == solver_status::converged);
    CHECK((recovered.x - Eigen::Vector2d(3.0, 0.0)).norm() <= 1e-6);
}

TEST_CASE("multistart over tr_sqp reports non_finite_evaluation for a NaN Jacobian at its start", "[sqp]")
{
    const defects::row_quadratic bare = bowl(defects::nowhere);
    const oracle::traced<defects::row_quadratic> problem{bare, nan_jacobian_at_call(1)};
    const solver_options<> opts;
    step_budget_solver solver{multistart_policy<tr_sqp_policy<>>{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    CHECK(solver.status() == solver_status::non_finite_evaluation);
    const solve_result<double> r = solver.solve(opts);
    CHECK(r.status == solver_status::non_finite_evaluation);
    CHECK(r.iterations == 0);
}

TEST_CASE("a multistart member over kraft_slsqp that reaches a NaN Jacobian ends there and the next member converges", "[sqp]")
{
    const defects::row_quadratic bare = bowl(defects::nowhere);
    const oracle::traced<defects::row_quadratic> problem{bare, nan_jacobian_at_call(2)};
    const solver_options<> opts;
    step_budget_solver solver{multistart_policy<kraft_slsqp_policy<>>{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    const solve_result<double> r = solver.solve(opts);
    CHECK(r.status == solver_status::converged);
    CHECK((r.x - Eigen::Vector2d(3.0, 0.0)).norm() <= 1e-6);
}

TEST_CASE("a multistart member over kraft_slsqp that starts at a non-finite point is never stepped", "[sqp]")
{
    const boxed_bowl problem{bowl(4.0)};
    step_budget_solver solver{multistart_policy<kraft_slsqp_policy<>>{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}};
    for(std::uint32_t i = 0; i < 100 && solver.state().restart_count < 3; ++i)
        solver.step();
    REQUIRE(solver.state().restart_count == 3);
    REQUIRE_FALSE(std::isfinite(solver.state().inner.objective_value));
    solver.step();
    CHECK(solver.state().restart_count == 4);
}
