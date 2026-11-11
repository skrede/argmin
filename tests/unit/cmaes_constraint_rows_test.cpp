#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

using pwq = alternative::cmaes::pwq_reparameterization_policy<double, 2>;
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<double, 2>;
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<double, 2>;

// |x|^2 on the [-5, 5]^2 box with n_eq equality and n_ineq inequality copies of the row 10 - x0 - x1, counting every
// objective and constraint call.
struct rowed_box
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t n_eq;
    std::int32_t n_ineq;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return n_eq; }

    std::int32_t num_inequality() const { return n_ineq; }

    double value(const Eigen::Vector2d& x) const
    {
        ++evaluations;
        return x.squaredNorm();
    }

    template <typename Values>
    void constraints(const Eigen::Vector2d& x, Values& c) const
    {
        ++evaluations;
        c.setConstant(10.0 - x.sum());
    }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Constant(-5.0); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(5.0); }
};

const Eigen::Vector2d start(0.5, 0.5);

template <typename Policy>
Policy seeded()
{
    Policy policy;
    policy.options.seed = 7u;
    return policy;
}

template <typename Policy>
void check_rejected_unevaluated(const rowed_box& problem)
{
    step_budget_solver solver{seeded<Policy>(), problem, start, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(solver.solve().status == solver_status::invalid_problem);
    CHECK(problem.evaluations == 0);
}

}

TEMPLATE_TEST_CASE("CMA-ES rejects a problem with constraint rows as an invalid problem without evaluating it", "[cmaes][invalid_problem]", pwq, repair_l2, no_repair)
{
    SECTION("an inequality row")
    {
        check_rejected_unevaluated<TestType>(rowed_box{.n_eq = 0, .n_ineq = 1, .evaluations = 0});
    }
    SECTION("an equality row")
    {
        check_rejected_unevaluated<TestType>(rowed_box{.n_eq = 1, .n_ineq = 0, .evaluations = 0});
    }
}

TEMPLATE_TEST_CASE("a CMA-ES reset rejects constraint rows its problem gained and runs once they are gone", "[cmaes][invalid_problem]", pwq, repair_l2, no_repair)
{
    rowed_box problem{.n_eq = 0, .n_ineq = 0, .evaluations = 0};
    step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
    REQUIRE(solver.status() == solver_status::running);
    problem.n_ineq = 1;
    CHECK(solver.reset(start) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(start) == solver_status::invalid_problem);
    const std::int32_t evaluated = problem.evaluations;
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluated);
    problem.n_ineq = 0;
    CHECK(solver.reset(start) == solver_status::running);
    CHECK(solver.step().policy_status != solver_status::invalid_problem);
}

TEMPLATE_TEST_CASE("CMA-ES solves a box-only problem that declares zero constraint rows", "[cmaes]", pwq, repair_l2, no_repair)
{
    const rowed_box problem{.n_eq = 0, .n_ineq = 0, .evaluations = 0};
    step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
    const auto result = solver.solve();
    CHECK(result.status != solver_status::invalid_problem);
    CHECK(result.objective_value < 1e-6);
}
