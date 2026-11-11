#include "rowed_box.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/step_and_time_budget_solver.h"

#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"

#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <tuple>
#include <cstdint>

using namespace argmin;
using namespace argmin::rowed;

namespace
{

using pwq = alternative::cmaes::pwq_reparameterization_policy<double, 2>;
using lbfgsb = lbfgsb_policy<double, 2>;
using witness_families = std::tuple<pwq, lbfgsb, restarting_policy<lbfgsb>>;
using family_group = basic_solver_group<round_robin_schedule, 2, rowed_box, lbfgsb, pwq>;

template <typename Result>
void check_reports_no_value(const Result& result)
{
    CHECK(result.status == solver_status::invalid_problem);
    CHECK(result.iterations == 0);
    CHECK(result.function_evaluations == 0);
    CHECK(std::isnan(result.objective_value));
    CHECK(std::isnan(result.gradient_norm));
    CHECK(std::isnan(result.constraint_violation));
}

template <typename Solver>
void check_rejected_step(Solver& solver)
{
    const auto step = solver.step();
    CHECK(step.policy_status == solver_status::invalid_problem);
    CHECK(step.evaluations == 0);
    CHECK(std::isnan(step.objective_value));
    CHECK(std::isnan(step.gradient_norm));
    CHECK(std::isnan(step.constraint_violation));
    CHECK(std::isnan(step.bound_violation));
    CHECK(std::isnan(solver.gradient_norm()));
    CHECK(std::isnan(solver.constraint_violation()));
    CHECK(std::isnan(solver.bound_violation()));
}

// A step before the solve shows that the step path counted no iteration either.
template <typename Solver>
void check_rejected_solve(Solver& solver, const rowed_box& problem)
{
    const std::int32_t evaluations = problem.evaluations;
    check_rejected_step(solver);
    check_reports_no_value(solver.solve());
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
}

template <typename Solver>
void check_rejected_on_reset(Solver& solver, rowed_box& problem)
{
    static_cast<void>(solver.solve());
    problem.n_ineq += 1;
    REQUIRE(solver.reset(start) == solver_status::invalid_problem);
    check_rejected_solve(solver, problem);
}

template <typename Solver>
void check_rejected_on_wrong_length(Solver& solver, const rowed_box& problem)
{
    static_cast<void>(solver.solve());
    REQUIRE(solver.reset(Eigen::VectorXd::Constant(3, 0.5)) == solver_status::invalid_problem);
    check_rejected_solve(solver, problem);
}


family_group make_group(const rowed_box& problem)
{
    return family_group{problem, Eigen::VectorXd(start), solver_options<>{}, {lbfgsb::options_type{}, seeded<pwq>().options}};
}

void check_group_rejected(family_group& group, const rowed_box& problem)
{
    const std::int32_t evaluations = problem.evaluations;
    check_reports_no_value(group.solve());
    for(const auto& member : group.results())
    {
        CHECK(member.status == solver_status::invalid_problem);
        CHECK(std::isnan(member.objective_value));
        CHECK(std::isnan(member.gradient_norm));
        CHECK(std::isnan(member.constraint_violation));
    }
    CHECK(problem.evaluations == evaluations);
}

}

TEMPLATE_LIST_TEST_CASE("a solve rejected at construction counts no iteration and reports no value", "[invalid_problem]", witness_families)
{
    const rowed_box problem{.n_eq = 0, .n_ineq = 1, .evaluations = 0};
    SECTION("step_budget_solver")
    {
        step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
        check_rejected_solve(solver, problem);
    }
    SECTION("time_budget_solver")
    {
        time_budget_solver solver{seeded<TestType>(), problem, start, time_budget_options<>{}};
        check_rejected_solve(solver, problem);
    }
    SECTION("step_and_time_budget_solver")
    {
        step_and_time_budget_solver solver{seeded<TestType>(), problem, start, time_budget_options<>{}};
        check_rejected_solve(solver, problem);
    }
    SECTION("stepper")
    {
        stepper solver{seeded<TestType>(), problem, start, solver_options<>{}};
        check_rejected_step(solver);
        CHECK(problem.evaluations == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("a solve rejected on reset counts no iteration and reports no value", "[invalid_problem]", witness_families)
{
    rowed_box problem{.n_eq = 0, .n_ineq = 0, .evaluations = 0};
    SECTION("rows the problem gained")
    {
        step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
        check_rejected_on_reset(solver, problem);
    }
    SECTION("a start of the wrong length")
    {
        step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
        check_rejected_on_wrong_length(solver, problem);
    }
    SECTION("stepper")
    {
        stepper solver{seeded<TestType>(), problem, start, solver_options<>{}};
        static_cast<void>(solver.step());
        problem.n_ineq = 1;
        REQUIRE(solver.reset(start) == solver_status::invalid_problem);
        check_rejected_step(solver);
    }
}

TEST_CASE("an augmented Lagrangian solve rejected on reset counts no iteration and reports no value", "[invalid_problem]")
{
    rowed_box problem{.n_eq = 0, .n_ineq = 1, .evaluations = 0};
    step_budget_solver solver{augmented_lagrangian_policy<lbfgsb>{}, problem, start, solver_options<>{}};
    check_rejected_on_reset(solver, problem);
}

TEST_CASE("a solver group whose members are all rejected counts no iteration and reports no value", "[invalid_problem]")
{
    rowed_box problem{.n_eq = 0, .n_ineq = 1, .evaluations = 0};
    SECTION("at construction")
    {
        family_group group = make_group(problem);
        check_group_rejected(group, problem);
    }
    SECTION("on reset")
    {
        problem.n_ineq = 0;
        family_group group = make_group(problem);
        static_cast<void>(group.solve());
        problem.n_ineq = 1;
        REQUIRE(group.reset(Eigen::VectorXd(start)) == solver_status::invalid_problem);
        check_group_rejected(group, problem);
    }
}
