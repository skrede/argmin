#include "conventions_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <utility>

using namespace argmin;

namespace
{

struct counted_solve
{
    std::int64_t reported;
    std::int64_t counted;
};

template <typename Policy, typename Problem>
counted_solve solve_counting(Policy policy, const Problem& problem)
{
    problem.calls = 0;
    solver_options<> options;
    options.max_iterations = 200;
    step_budget_solver solver{std::move(policy), problem, Eigen::VectorXd::Zero(2).eval(), options};
    const solve_result<double> result = solver.solve();
    return {static_cast<std::int64_t>(result.function_evaluations), problem.calls};
}

using al_policy = augmented_lagrangian_policy<lbfgsb_policy<>>;

}

TEST_CASE("nw_sqp reports every objective evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(nw_sqp_policy<>{}, defects::counting_row_bowl{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("filter_nw_sqp reports every objective evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(filter_nw_sqp_policy<>{}, defects::counting_row_bowl{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("cobyla reports every objective evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(cobyla_policy<>{}, defects::counting_row_bowl{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("byrd_lbfgsb reports every objective evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(byrd_lbfgsb_policy<>{}, defects::counting_box_bowl{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("projected_gradient_gn reports every residual evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(projected_gradient_gn_policy<>{}, defects::counting_box_residual{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("the augmented Lagrangian reports every objective evaluation in its function evaluations", "[known-defect][!shouldfail]")
{
    const counted_solve solve = solve_counting(al_policy{}, defects::counting_row_bowl{});
    CHECK(solve.reported == solve.counted);
}

TEST_CASE("every evaluation-count policy evaluates its counting problem")
{
    CHECK(solve_counting(nw_sqp_policy<>{}, defects::counting_row_bowl{}).counted > 0);
    CHECK(solve_counting(filter_nw_sqp_policy<>{}, defects::counting_row_bowl{}).counted > 0);
    CHECK(solve_counting(cobyla_policy<>{}, defects::counting_row_bowl{}).counted > 0);
    CHECK(solve_counting(byrd_lbfgsb_policy<>{}, defects::counting_box_bowl{}).counted > 0);
    CHECK(solve_counting(projected_gradient_gn_policy<>{}, defects::counting_box_residual{}).counted > 0);
    CHECK(solve_counting(al_policy{}, defects::counting_row_bowl{}).counted > 0);
}
