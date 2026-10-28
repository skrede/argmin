#include "sqp_safety_problems.h"
#include "bound_ls_al_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>
#include <variant>

using namespace argmin;

namespace
{

using vector1 = Eigen::Vector<double, 1>;
using lm_stepper = stepper<lm_policy<double, 1>, 1, defects::sqrt_residual>;

lm_stepper stepped_into_nan(const defects::sqrt_residual& problem)
{
    lm_stepper st{lm_policy<double, 1>{}, problem, vector1{1.0}, solver_options<>{}, lm_policy<double, 1>::options_type{}};
    st.step();
    return st;
}

// (x - 1)^2 with an exact NaN at the start x = 0; the gradient stays finite everywhere.
struct nan_at_origin
{
    static constexpr std::int32_t problem_dimension = 1;

    std::int32_t dimension() const { return 1; }

    double value(const vector1& x) const { return x[0] == 0.0 ? std::numeric_limits<double>::quiet_NaN() : (x[0] - 1.0) * (x[0] - 1.0); }

    void gradient(const vector1& x, vector1& g) const { g[0] = 2.0 * (x[0] - 1.0); }
};

using nan_start_solver = step_budget_solver<lbfgsb_policy<double, 1>, 1, nan_at_origin>;

nan_start_solver solver_from_nan_start(const nan_at_origin& problem)
{
    return {lbfgsb_policy<double, 1>{}, problem, vector1{0.0}, solver_options<>{}, lbfgsb_policy<double, 1>::options_type{}};
}

}

TEST_CASE("stepper ends diverged after a step into a non-finite residual", "[status]")
{
    const defects::sqrt_residual problem{};
    CHECK(stepped_into_nan(problem).status() == solver_status::diverged);
}

TEST_CASE("stepper converged() is false after a solve that ended diverged", "[known-defect][!shouldfail]")
{
    const defects::sqrt_residual problem{};
    CHECK_FALSE(stepped_into_nan(problem).converged());
}

TEST_CASE("solve does not claim converged with a NaN objective from a NaN start", "[known-defect][!shouldfail]")
{
    const nan_at_origin problem{};
    nan_start_solver solver = solver_from_nan_start(problem);
    const solve_result<double, 1> r = solver.solve();
    CHECK_FALSE((r.status == solver_status::converged && std::isnan(r.objective_value)));
}

TEST_CASE("solve returns the iterate the solver holds when its start objective is NaN", "[known-defect][!shouldfail]")
{
    const nan_at_origin problem{};
    nan_start_solver solver = solver_from_nan_start(problem);
    const solve_result<double, 1> r = solver.solve();
    CHECK(r.x[0] == solver.state().x[0]);
}

TEST_CASE("the stationarity criterion does not accept an infeasible null step", "[known-defect][!shouldfail]")
{
    step_result<double> r;
    r.is_null_step = true;
    r.constraint_violation = 1.0;
    r.kkt_residual = 0.0;
    CHECK(default_convergence<double>{}.check(r, 2) != solver_status::converged);
}

TEST_CASE("a caller's explicit stall window survives the policy's hint", "[known-defect][!shouldfail]")
{
    const defects::repeat_counting_rosenbrock problem{};
    solver_options<> opts;
    opts.set_stall_threshold(1e-9);
    opts.set_stall_window(7);
    step_budget_solver<lbfgsb_policy<double, 2>, 2, defects::repeat_counting_rosenbrock> solver{lbfgsb_policy<double, 2>{}, problem, Eigen::Vector2d(-1.2, 1.0), opts, lbfgsb_policy<double, 2>::options_type{}};
    CHECK(std::get<stall_tolerance_criterion<double>>(solver.convergence().criteria).window == 7);
}

// Two equality residuals of 0.6e-6 each at the start: their L1 sum and their maximum straddle the 1e-6 floor.
TEST_CASE("the solver and its result report the same constraint violation for one iterate", "[known-defect][!shouldfail]")
{
    const defects::row_quadratic problem{
        2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Zero(2), Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), defects::nowhere};
    step_budget_solver solver{nw_sqp_policy<>{}, problem, Eigen::VectorXd::Constant(2, 1.0 + 0.6e-6), solver_options<>{}};
    const solve_result<double> r = solver.step_n(0);
    CHECK(solver.constraint_violation() == r.constraint_violation);
}
