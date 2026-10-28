#include "stochastic_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using defects::counting_box;

namespace
{

cmaes_policy<> seeded_cmaes(std::uint64_t seed)
{
    cmaes_policy<> policy;
    policy.options.seed = seed;
    return policy;
}

counting_box box(const Eigen::Vector2d& lower, const Eigen::Vector2d& upper)
{
    return {lower, upper, 0, 0};
}

solve_result<double> solve_from_outside_unit_interval(const defects::linear_unit_interval& problem)
{
    solver_options<> options;
    options.max_iterations = 100;
    step_budget_solver solver{seeded_cmaes(11), problem, Eigen::VectorXd::Constant(1, -5.0).eval(), options};
    return solver.solve();
}

}

TEST_CASE("cmaes rejects a generation whose objective values are all NaN before ranking it", "[known-defect][!shouldfail]")
{
    const defects::nan_everywhere problem;
    cmaes_policy<> policy = seeded_cmaes(5);
    auto s = policy.init(problem, Eigen::Vector2d(0.1, 0.2), solver_options<>{});
    const step_result<double> r = policy.step(s);
    CHECK(r.policy_status == solver_status::diverged);
}

TEST_CASE("cmaes reset restores the anisotropic initial scaling of a fresh run", "[known-defect][!shouldfail]")
{
    const counting_box problem = box(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.01));
    const Eigen::Vector2d x0(0.5, 0.005);
    cmaes_policy<> fresh_policy = seeded_cmaes(1);
    const auto fresh = fresh_policy.init(problem, x0, solver_options<>{});
    cmaes_policy<> policy = seeded_cmaes(1);
    auto s = policy.init(problem, x0, solver_options<>{});
    for(std::int32_t k = 0; k < 5; ++k)
        policy.step(s);
    policy.reset(s, x0);
    CHECK(s.D.isApprox(fresh.D, 1e-12));
    CHECK(s.C.isApprox(fresh.C, 1e-12));
}

TEST_CASE("cmaes reset_clear keeps the start point it is handed as the distribution mean", "[known-defect][!shouldfail]")
{
    const counting_box problem = box(Eigen::Vector2d::Constant(-5.0), Eigen::Vector2d::Constant(5.0));
    cmaes_policy<> policy = seeded_cmaes(3);
    auto s = policy.init(problem, Eigen::Vector2d(1.0, 1.0), solver_options<>{});
    const Eigen::Vector2d anchor(2.5, -1.25);
    policy.reset_clear(s, anchor);
    CHECK(s.mean == anchor);
}

TEST_CASE("cmaes returns a point inside the box when started outside it", "[known-defect][!shouldfail]")
{
    const defects::linear_unit_interval problem{0};
    const solve_result<double> result = solve_from_outside_unit_interval(problem);
    CHECK((result.x[0] >= 0.0 && result.x[0] <= 1.0));
}

TEST_CASE("cmaes evaluates no point outside the box when started outside it", "[known-defect][!shouldfail]")
{
    const defects::linear_unit_interval problem{0};
    solve_from_outside_unit_interval(problem);
    CHECK(problem.calls_outside_box == 0);
}

TEST_CASE("cmaes counts every objective call in its function evaluations", "[known-defect][!shouldfail]")
{
    const counting_box problem = box(Eigen::Vector2d::Constant(-5.0), Eigen::Vector2d::Constant(5.0));
    solver_options<> options;
    options.max_iterations = 20;
    step_budget_solver solver{seeded_cmaes(13), problem, Eigen::VectorXd::Ones(2).eval(), options};
    const solve_result<double> result = solver.solve();
    CHECK(static_cast<std::int64_t>(result.function_evaluations) == problem.calls);
}
