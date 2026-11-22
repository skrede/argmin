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

#include "argmin/test_functions/rosenbrock.h"

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

// Reports diverged from its first step, as a family's own divergence test would, without evaluating anything.
struct diverging_policy
{
    using scalar_type = double;

    struct state_type
    {
        vector1 x;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const vector1& x0, const solver_options<Convergence>&)
    {
        return {x0};
    }

    step_result<double> step(state_type& s) { return {.objective_value = 1.0, .x_norm = s.x.norm(), .policy_status = solver_status::diverged}; }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

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

TEST_CASE("stepper converged() is false after a solve that ended diverged", "[status]")
{
    const nan_at_origin problem{};
    stepper<diverging_policy, 1, nan_at_origin> st{problem, vector1{1.0}};
    st.step();
    REQUIRE(st.status() == solver_status::diverged);
    CHECK_FALSE(st.converged());
}

TEST_CASE("stepper converged() is false after a rejected start", "[status]")
{
    const rosenbrock<double> problem{};
    stepper st{lbfgsb_policy<>{}, problem, Eigen::VectorXd::Zero(3)};
    REQUIRE(st.status() == solver_status::invalid_problem);
    CHECK_FALSE(st.converged());
}

TEST_CASE("solve does not claim converged with a NaN objective from a NaN start", "[status]")
{
    const nan_at_origin problem{};
    nan_start_solver solver = solver_from_nan_start(problem);
    const solve_result<double, 1> r = solver.solve();
    CHECK_FALSE((r.status == solver_status::converged && std::isnan(r.objective_value)));
}

TEST_CASE("solve returns the iterate the solver holds when its start objective is NaN", "[status]")
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

TEST_CASE("a caller's explicit stall window survives the policy's hint", "[status]")
{
    const defects::repeat_counting_rosenbrock problem{};
    solver_options<> opts;
    opts.set_stall_threshold(1e-9);
    opts.set_stall_window(7);
    step_budget_solver<lbfgsb_policy<double, 2>, 2, defects::repeat_counting_rosenbrock> solver{lbfgsb_policy<double, 2>{}, problem, Eigen::Vector2d(-1.2, 1.0), opts, lbfgsb_policy<double, 2>::options_type{}};
    CHECK(std::get<stall_tolerance_criterion<double>>(solver.convergence().criteria).window == 7);
    CHECK(std::get<stall_tolerance_criterion<double>>(solver.convergence().criteria).window_in_force() == 7);
}

TEST_CASE("an explicit solve takes the policy's stall window only where the caller left it unset", "[status]")
{
    const defects::repeat_counting_rosenbrock problem{};
    solver_options<> opts;
    opts.max_iterations = 1;
    step_budget_solver<lbfgsb_policy<double, 2>, 2, defects::repeat_counting_rosenbrock> solver{
        lbfgsb_policy<double, 2>{}, problem, Eigen::Vector2d(-1.2, 1.0), opts, lbfgsb_policy<double, 2>::options_type{.stall_window = 9}};
    const auto& stall = std::get<stall_tolerance_criterion<double>>(opts.convergence.criteria);
    solver.solve(opts);
    CHECK_FALSE(stall.window.has_value());
    CHECK(stall.window_in_force() == 9);
    opts.set_stall_window(7);
    solver.reset_clear(Eigen::Vector2d(-1.2, 1.0));
    solver.solve(opts);
    CHECK(stall.window == 7);
    CHECK(stall.window_in_force() == 7);
}

// Two equality residuals of 0.6e-6 each at the start: their L1 sum and their maximum straddle the 1e-6 floor.
TEST_CASE("the solver and its result report the same constraint violation for one iterate", "[status]")
{
    const defects::row_quadratic problem{
        2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Zero(2), Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), defects::nowhere};
    step_budget_solver solver{nw_sqp_policy<>{}, problem, Eigen::VectorXd::Constant(2, 1.0 + 0.6e-6), solver_options<>{}};
    const solve_result<double> r = solver.step_n(0);
    CHECK(solver.constraint_violation() == r.constraint_violation);
}
