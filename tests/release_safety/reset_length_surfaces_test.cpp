#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"

#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <chrono>
#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

struct counting_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    double value(const vector2& x) const
    {
        ++evaluations;
        return (x - vector2(1.0, 2.0)).squaredNorm();
    }

    void gradient(const vector2& x, vector2& g) const
    {
        ++evaluations;
        g = 2.0 * (x - vector2(1.0, 2.0));
    }
};

// A deadline far enough out that the solve loop reaches the step instead of reporting the time limit first.
time_budget_options<> generous_deadline()
{
    time_budget_options<> opts;
    opts.max_time = std::chrono::minutes(10);
    return opts;
}

struct resizable_bowl
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t n;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        ++evaluations;
        return x.squaredNorm();
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        ++evaluations;
        g = 2.0 * x;
    }
};

template <typename Solver>
solver_status restart(Solver& solver, const Eigen::VectorXd& x0, bool clear)
{
    return clear ? solver.reset_clear(x0) : solver.reset(x0);
}

template <typename Solver, typename Problem>
void check_wrong_length_is_rejected(Solver& solver, const Problem& problem, std::int32_t length, bool clear)
{
    const Eigen::VectorXd before = solver.state().x;
    const std::int32_t evaluations = problem.evaluations;
    CHECK(restart(solver, Eigen::VectorXd::Constant(length, 3.0), clear) == solver_status::invalid_problem);
    CHECK(solver.state().x == before);
    CHECK(solver.status() == solver_status::invalid_problem);
    const auto step = solver.step();
    REQUIRE(step.policy_status.has_value());
    CHECK(*step.policy_status == solver_status::invalid_problem);
    if constexpr(requires { solver.solve(); })
        CHECK(solver.solve().status == solver_status::invalid_problem);
    CHECK(solver.state().x == before);
    CHECK(problem.evaluations == evaluations);
}

template <typename Solver>
void check_valid_reset_recovers(Solver& solver, std::int32_t length, bool clear)
{
    CHECK(restart(solver, Eigen::VectorXd::Constant(length, 3.0), clear) == solver_status::running);
    CHECK(solver.status() == solver_status::running);
    const auto step = solver.step();
    CHECK_FALSE(step.policy_status == solver_status::invalid_problem);
    CHECK(solver.state().x != Eigen::VectorXd::Constant(length, 3.0));
}

template <typename Solver>
void check_surface(Solver& solver, const counting_bowl& problem, bool clear)
{
    check_wrong_length_is_rejected(solver, problem, 1, clear);
    check_wrong_length_is_rejected(solver, problem, 3, clear);
    check_valid_reset_recovers(solver, 2, clear);
}

}

TEST_CASE("a step_budget_solver over lbfgsb rejects a reset of the wrong length and recovers on a valid one", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    const counting_bowl problem{.evaluations = 0};
    step_budget_solver solver{lbfgsb_policy<double, 2>{}, problem, vector2(0.0, 0.0), solver_options<>{}};
    check_surface(solver, problem, clear);
}

TEST_CASE("a stepper over lbfgsb rejects a reset of the wrong length and recovers on a valid one", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    const counting_bowl problem{.evaluations = 0};
    stepper solver{lbfgsb_policy<double, 2>{}, problem, vector2(0.0, 0.0), solver_options<>{}};
    check_surface(solver, problem, clear);
}

TEST_CASE("a time_budget_solver over lbfgsb rejects a reset of the wrong length and recovers on a valid one", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    const counting_bowl problem{.evaluations = 0};
    time_budget_solver solver{lbfgsb_policy<double, 2>{}, problem, vector2(0.0, 0.0), generous_deadline()};
    check_surface(solver, problem, clear);
}

TEST_CASE("a step_and_time_budget_solver over lbfgsb rejects a reset of the wrong length and recovers on a valid one", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    const counting_bowl problem{.evaluations = 0};
    step_and_time_budget_solver solver{lbfgsb_policy<double, 2>{}, problem, vector2(0.0, 0.0), generous_deadline()};
    check_surface(solver, problem, clear);
}

TEST_CASE("a solver group reports its first rejecting member on a reset of the wrong length", "[release-safety][reset-length]")
{
    const counting_bowl problem{.evaluations = 0};
    basic_solver_group<round_robin_schedule, 2, counting_bowl, lbfgsb_policy<double, 2>, lbfgsb_policy<double, 2>> group{problem, Eigen::VectorXd::Zero(2)};
    const std::int32_t evaluations = problem.evaluations;
    CHECK(group.reset(Eigen::VectorXd::Zero(3)) == solver_status::invalid_problem);
    const auto step = group.step();
    REQUIRE(step.policy_status.has_value());
    CHECK(*step.policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    CHECK(group.reset(Eigen::VectorXd::Constant(2, 3.0)) == solver_status::running);
    CHECK_FALSE(group.step().policy_status == solver_status::invalid_problem);
}

TEST_CASE("a dynamic-dimension solver rejects a reset whose length differs from the problem dimension", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    const resizable_bowl problem{.n = 2, .evaluations = 0};
    step_budget_solver solver{lbfgsb_policy<>{}, problem, Eigen::VectorXd::Zero(2).eval(), solver_options<>{}};
    check_wrong_length_is_rejected(solver, problem, 1, clear);
    check_wrong_length_is_rejected(solver, problem, 3, clear);
    check_valid_reset_recovers(solver, 2, clear);
}

TEST_CASE("a dynamic-dimension solver rejects every reset once its problem reports another dimension", "[release-safety][reset-length]")
{
    const bool clear = GENERATE(false, true);
    resizable_bowl problem{.n = 2, .evaluations = 0};
    step_budget_solver solver{lbfgsb_policy<>{}, problem, Eigen::VectorXd::Zero(2).eval(), solver_options<>{}};
    problem.n = 3;
    check_wrong_length_is_rejected(solver, problem, 2, clear);
    check_wrong_length_is_rejected(solver, problem, 3, clear);
}
