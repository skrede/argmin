#include "bound_ls_al_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using defects::unbounded;

namespace
{

// 0.5 |x - (2, 0.5)|^2 on [-1, 1]^2 started at its bound-constrained minimizer (1, 0.5), where the projected
// gradient is exactly zero.
template <typename Policy>
solver_status status_of_warm_start_at_optimum()
{
    const defects::box_bowl problem{{2.0, 0.5}, Eigen::Vector2d::Constant(-1.0), Eigen::Vector2d::Constant(1.0), {}};
    step_budget_solver<Policy, 2, defects::box_bowl> solver{Policy{}, problem, Eigen::Vector2d(1.0, 0.5), solver_options<>{}, typename Policy::options_type{}};
    return solver.solve().status;
}

template <typename Policy>
std::int32_t repeated_gradient_calls_over_ten_steps()
{
    const defects::repeat_counting_rosenbrock problem{};
    Policy policy;
    auto state = policy.init(problem, Eigen::Vector2d(-1.2, 1.0), solver_options<>{});
    for(std::int32_t k = 0; k < 10; ++k)
        static_cast<void>(policy.step(state));
    return problem.repeated_gradient_calls;
}

template <typename Policy>
bool first_evaluation_inside_box()
{
    const defects::box_bowl problem{{0.5, 0.5}, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {}};
    Policy policy;
    static_cast<void>(policy.init(problem, Eigen::Vector2d(-1.0, 2.0), solver_options<>{}));
    const Eigen::Vector2d x = *problem.first_value_at;
    return (x.array() >= 0.0).all() && (x.array() <= 1.0).all();
}

}

TEST_CASE("lbfgsb reports converged on a warm start at the bound-constrained optimum", "[lbfgsb]")
{
    CHECK(status_of_warm_start_at_optimum<lbfgsb_policy<double, 2>>() == solver_status::converged);
}

TEST_CASE("byrd_lbfgsb reports converged on a warm start at the bound-constrained optimum", "[known-defect][!shouldfail]")
{
    CHECK(status_of_warm_start_at_optimum<byrd_lbfgsb_policy<double, 2>>() == solver_status::converged);
}

TEST_CASE("lbfgsb evaluates the gradient once per accepted point", "[known-defect][!shouldfail]")
{
    CHECK(repeated_gradient_calls_over_ten_steps<lbfgsb_policy<double, 2>>() == 0);
}

TEST_CASE("byrd_lbfgsb evaluates the gradient once per accepted point", "[known-defect][!shouldfail]")
{
    CHECK(repeated_gradient_calls_over_ten_steps<byrd_lbfgsb_policy<double, 2>>() == 0);
}

TEST_CASE("lbfgsb evaluates its first objective inside the box when started outside it", "[known-defect][!shouldfail]")
{
    CHECK(first_evaluation_inside_box<lbfgsb_policy<double, 2>>());
}

TEST_CASE("byrd_lbfgsb evaluates its first objective inside the box when started outside it", "[known-defect][!shouldfail]")
{
    CHECK(first_evaluation_inside_box<byrd_lbfgsb_policy<double, 2>>());
}

// No bound is active at (0.5, 0), so the all-free branch runs although -H g leaves the box; the projected path and
// subspace minimization reach the constrained minimizer (0, -1) in one step.
TEST_CASE("lbfgsb reaches the constrained minimizer in one step when the free direction leaves the box", "[known-defect][!shouldfail]")
{
    const defects::box_bowl problem{{-0.5, -1.0}, Eigen::Vector2d(0.0, -unbounded), Eigen::Vector2d::Constant(unbounded), {}};
    lbfgsb_policy<double, 2> policy;
    auto state = policy.init(problem, Eigen::Vector2d(0.5, 0.0), solver_options<>{});
    static_cast<void>(policy.step(state));
    CHECK((state.x - Eigen::Vector2d(0.0, -1.0)).norm() <= 1e-12);
}
