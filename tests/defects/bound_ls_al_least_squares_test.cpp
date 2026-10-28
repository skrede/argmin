#include "bound_ls_al_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace argmin;

namespace
{

solve_result<double, 1> lm_solve_from_one()
{
    using policy = lm_policy<double, 1>;
    const defects::sqrt_residual problem{};
    step_budget_solver<policy, 1, defects::sqrt_residual> solver{policy{}, problem, Eigen::Vector<double, 1>(1.0), solver_options<>{}, policy::options_type{}};
    return solver.solve();
}

template <typename Policy>
bool reports_status_within_five_steps()
{
    const defects::sqrt_box_residual problem{};
    Policy policy;
    auto state = policy.init(problem, Eigen::Vector2d::Zero().eval(), solver_options<>{});
    for(std::int32_t k = 0; k < 5; ++k)
    {
        if(policy.step(state).policy_status)
            return true;
    }
    return false;
}

}

TEST_CASE("lm does not end diverged after one trial into a non-finite residual", "[known-defect][!shouldfail]")
{
    CHECK(lm_solve_from_one().status != solver_status::diverged);
}

TEST_CASE("lm solves the square-root residual after a trial into a non-finite residual", "[known-defect][!shouldfail]")
{
    CHECK(std::abs(lm_solve_from_one().x[0] - 0.01) <= 1e-6);
}

// g = (1, 0.1) at the origin while the clipped damped step is d = (0, ~1.5), so g^T d > 0.
TEST_CASE("projected_gradient_gn rejects a non-descent direction before backtracking", "[known-defect][!shouldfail]")
{
    const defects::counting_linear_residual problem{};
    projected_gradient_gn_policy<double, 2> policy;
    policy.options.initial_lambda = 1e-12;
    auto state = policy.init(problem, Eigen::Vector2d::Zero().eval(), solver_options<>{});
    problem.residual_calls = 0;
    static_cast<void>(policy.step(state));
    CHECK(problem.residual_calls <= 1);
}

TEST_CASE("projected_gn reports a status when its finite-difference Jacobian is non-finite", "[known-defect][!shouldfail]")
{
    CHECK(reports_status_within_five_steps<projected_gn_policy<double, 2>>());
}

TEST_CASE("projected_gradient_gn reports a status when its finite-difference Jacobian is non-finite", "[known-defect][!shouldfail]")
{
    CHECK(reports_status_within_five_steps<projected_gradient_gn_policy<double, 2>>());
}
