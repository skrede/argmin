#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

// 10 (x1 - x0^2)^2 + (1 - x0)^2 on [0, 10]^2 with the active row 2.5 - x0 - x1 >= 0. Every evaluation at a point
// outside the box is counted.
struct rosenbrock_in_box
{
    static constexpr std::int32_t problem_dimension = 2;

    mutable std::int32_t outside;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const vector2& x) const
    {
        note(x);
        const double a = x[1] - x[0] * x[0];
        return 10.0 * a * a + (1.0 - x[0]) * (1.0 - x[0]);
    }

    void gradient(const vector2& x, vector2& g) const
    {
        note(x);
        g[0] = -40.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 20.0 * (x[1] - x[0] * x[0]);
    }

    template <typename Values>
    void constraints(const vector2& x, Values& out) const
    {
        note(x);
        out[0] = 2.5 - x[0] - x[1];
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2& x, Jacobian& J) const
    {
        note(x);
        J.setConstant(-1.0);
    }

    vector2 lower_bounds() const { return vector2::Zero(); }

    vector2 upper_bounds() const { return vector2::Constant(10.0); }

    void note(const vector2& x) const
    {
        if((x.array() < 0.0).any() || (x.array() > 10.0).any())
            ++outside;
    }
};

using asymptote_policies = std::tuple<mma_policy<double, 2>, gcmma_policy<double, 2>, alternative::gcmma::raa_augmented_policy<double, 2>, alternative::gcmma::move_limit_shrink_policy<double, 2>>;

// An unordered window would trip std::clamp's lo <= hi precondition inside the step before this check sees it.
template <typename Solver>
bool steps_keep_ordered_windows(Solver& solver, std::int32_t steps)
{
    for(std::int32_t k = 0; k < steps; ++k)
    {
        solver.step();
        if((solver.state().alpha.array() > solver.state().beta.array()).any())
            return false;
    }
    return true;
}

// 60 steps from (3, 3) move the iterate toward (1, 1), far from the upper corner of the box.
template <typename Solver>
void contract_asymptotes(Solver& solver)
{
    for(std::int32_t k = 0; k < 60; ++k)
        solver.step();
}

}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy re-anchors its asymptotes around a hot reset point", "[release-safety][hot-reset]", asymptote_policies)
{
    const rosenbrock_in_box problem{};
    step_budget_solver solver{TestType{}, problem, vector2(3.0, 3.0), solver_options<>{}};
    contract_asymptotes(solver);
    const vector2 target(9.5, 9.5);
    REQUIRE(solver.reset(target) == solver_status::running);
    const step_budget_solver fresh{TestType{}, problem, target, solver_options<>{}};
    CHECK(solver.state().x == target);
    CHECK(solver.state().L == fresh.state().L);
    CHECK(solver.state().U == fresh.state().U);
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy steps through an ordered window after a hot reset", "[release-safety][hot-reset]", asymptote_policies)
{
    const rosenbrock_in_box problem{};
    step_budget_solver solver{TestType{}, problem, vector2(3.0, 3.0), solver_options<>{}};
    contract_asymptotes(solver);
    REQUIRE(solver.reset(vector2(9.5, 9.5)) == solver_status::running);
    CHECK(steps_keep_ordered_windows(solver, 5));
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy projects an out-of-box hot reset start onto the box", "[release-safety][hot-reset]", asymptote_policies)
{
    const rosenbrock_in_box problem{};
    step_budget_solver solver{TestType{}, problem, vector2(3.0, 3.0), solver_options<>{}};
    contract_asymptotes(solver);
    REQUIRE(solver.reset(vector2(12.0, -1.0)) == solver_status::running);
    CHECK(solver.state().x == vector2(10.0, 0.0));
    CHECK(steps_keep_ordered_windows(solver, 5));
    CHECK(problem.outside == 0);
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy projects an out-of-box start onto the box at construction", "[release-safety][hot-reset]", asymptote_policies)
{
    const rosenbrock_in_box problem{};
    step_budget_solver solver{TestType{}, problem, vector2(12.0, -1.0), solver_options<>{}};
    CHECK(solver.state().x == vector2(10.0, 0.0));
    CHECK(steps_keep_ordered_windows(solver, 5));
    CHECK(problem.outside == 0);
}

TEMPLATE_LIST_TEST_CASE("a moving-asymptote policy steps through an ordered window from a far out-of-box start", "[release-safety][hot-reset]", asymptote_policies)
{
    const rosenbrock_in_box problem{};
    step_budget_solver solver{TestType{}, problem, vector2(40.0, -1.0), solver_options<>{}};
    CHECK(steps_keep_ordered_windows(solver, 5));
    CHECK(problem.outside == 0);
}
