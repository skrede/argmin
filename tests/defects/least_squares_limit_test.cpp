#include "bound_ls_al_problems.h"
#include "non_finite_limit_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

template <typename Policy>
Policy dogleg(Policy policy)
{
    policy.options.use_dogleg = true;
    return policy;
}

template <typename Policy>
Policy damping_ceiling(Policy policy, double lambda_max)
{
    policy.options.lambda_max = lambda_max;
    return policy;
}

// The status after a vanished-step end at the NaN wall from x = 1, a hot reset to x = -3 and a second solve there.
template <typename Policy>
solver_status status_after_reset(Policy policy)
{
    const defects::stepped_wall_residual problem{};
    step_budget_solver<Policy, 1, defects::stepped_wall_residual> solver{policy, problem, Eigen::Vector<double, 1>(1.0), solver_options<>{}};
    if(solver.solve().status != solver_status::non_finite_evaluation)
        return solver_status::running;
    solver.reset(Eigen::Vector<double, 1>(-3.0));
    return solver.solve().status;
}

// The damping (or radius) the terminal step's trial was computed with, against the wall where every trial is NaN.
template <typename Policy, typename Limit>
double limit_of_last_trial(Policy policy, Limit limit)
{
    const defects::walled_line_residual problem{};
    step_budget_solver<Policy, 1, defects::walled_line_residual> solver{policy, problem, Eigen::Vector<double, 1>(1.0), solver_options<>{}};
    double last = std::numeric_limits<double>::quiet_NaN();
    for(std::int32_t k = 0; k < 200 && solver.status() == solver_status::running; ++k)
    {
        last = limit(solver.state());
        solver.step();
    }
    return solver.status() == solver_status::non_finite_evaluation ? last : std::numeric_limits<double>::quiet_NaN();
}

}

TEST_CASE("a hot reset after a vanished-step end clears the least-squares non-finite record", "[least_squares]")
{
    CHECK(status_after_reset(lm_policy<double, 1>{}) == solver_status::converged);
    CHECK(status_after_reset(projected_gn_policy<double, 1>{}) == solver_status::converged);
    CHECK(status_after_reset(dogleg(projected_gn_policy<double, 1>{})) == solver_status::converged);
    CHECK(status_after_reset(dogleg(projected_gradient_gn_policy<double, 1>{})) == solver_status::converged);
}

TEST_CASE("projected_gradient_gn accepts a zero projected direction after a non-finite rejection as an ordinary step", "[least_squares]")
{
    const defects::blocked_corner_residual problem{};
    projected_gradient_gn_policy<double, 2> policy;
    auto state = policy.init(problem, Eigen::Vector2d::Zero().eval(), solver_options<>{});
    std::int32_t null_steps = 0;
    std::int32_t accepted_steps = 0;
    for(std::int32_t k = 0; k < 6; ++k)
    {
        const step_result<double> r = policy.step(state);
        CHECK_FALSE((r.improved && r.is_null_step));
        CHECK_FALSE(r.policy_status.has_value());
        null_steps += r.is_null_step ? 1 : 0;
        accepted_steps += r.improved ? 1 : 0;
    }
    CHECK(null_steps == 3);
    CHECK(accepted_steps == 3);
}

TEST_CASE("the least-squares solvers try a trial at their damping ceiling or radius floor before ending non_finite_evaluation", "[least_squares]")
{
    const auto lambda = [](const auto& s) { return s.lambda; };
    const auto delta = [](const auto& s) { return s.delta; };
    CHECK(limit_of_last_trial(damping_ceiling(lm_policy<double, 1>{}, 1e3), lambda) == 1e3);
    CHECK(limit_of_last_trial(damping_ceiling(projected_gn_policy<double, 1>{}, 1e3), lambda) == 1e3);
    CHECK(limit_of_last_trial(damping_ceiling(projected_gradient_gn_policy<double, 1>{}, 1e3), lambda) == 1e3);
    CHECK(limit_of_last_trial(dogleg(projected_gn_policy<double, 1>{}), delta) <= std::numeric_limits<double>::epsilon());
    CHECK(limit_of_last_trial(dogleg(projected_gradient_gn_policy<double, 1>{}), delta) <= std::numeric_limits<double>::epsilon());
}
