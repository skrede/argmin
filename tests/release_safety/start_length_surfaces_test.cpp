#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"

#include "argmin/result/status.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <chrono>
#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

template <std::int32_t N>
struct counting_bowl
{
    static constexpr std::int32_t problem_dimension = N;

    std::int32_t n;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return n; }

    double value(const Eigen::Vector<double, N>& x) const
    {
        ++evaluations;
        return (x.array() - 1.0).matrix().squaredNorm();
    }

    void gradient(const Eigen::Vector<double, N>& x, Eigen::Vector<double, N>& g) const
    {
        ++evaluations;
        g = 2.0 * (x.array() - 1.0).matrix();
    }
};

using fixed_bowl = counting_bowl<2>;
using dynamic_bowl = counting_bowl<dynamic_dimension>;

// A deadline far enough out that the solve loop reaches the step instead of reporting the time limit first.
time_budget_options<> generous_deadline()
{
    time_budget_options<> opts;
    opts.max_time = std::chrono::minutes(10);
    return opts;
}

template <typename Solver, typename Problem>
void check_construction_rejected(Solver& solver, const Problem& problem)
{
    CHECK(solver.status() == solver_status::invalid_problem);
    const auto step = solver.step();
    REQUIRE(step.policy_status.has_value());
    CHECK(*step.policy_status == solver_status::invalid_problem);
    if constexpr(requires { solver.solve(); })
    {
        const auto result = solver.solve();
        CHECK(result.status == solver_status::invalid_problem);
        CHECK(result.iterations == 0);
        CHECK(result.function_evaluations == 0);
    }
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(problem.evaluations == 0);
}

template <typename Solver>
void check_valid_reset_initializes(Solver& solver, bool clear)
{
    const Eigen::VectorXd start = Eigen::VectorXd::Constant(2, 3.0);
    CHECK((clear ? solver.reset_clear(start) : solver.reset(start)) == solver_status::running);
    CHECK(solver.status() == solver_status::running);
    CHECK(solver.state().x == start);
    if constexpr(requires { solver.solve(); })
        CHECK((solver.solve().x.array() - 1.0).abs().maxCoeff() < 1e-4);
    else
        CHECK_FALSE(solver.step().policy_status.has_value());
}

template <typename Solver>
void check_wrong_length_surface(const fixed_bowl& problem, bool clear, auto&&... opts)
{
    for(const std::int32_t length : {1, 3})
    {
        Solver solver{lbfgsb_policy<double, 2>{}, problem, Eigen::VectorXd::Constant(length, 3.0).eval(), opts...};
        check_construction_rejected(solver, problem);
    }
    Solver solver{lbfgsb_policy<double, 2>{}, problem, Eigen::VectorXd::Constant(1, 3.0).eval(), opts...};
    check_valid_reset_initializes(solver, clear);
}

}

TEST_CASE("a step_budget_solver over lbfgsb rejects a start of the wrong length at construction", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .evaluations = 0};
    check_wrong_length_surface<step_budget_solver<lbfgsb_policy<double, 2>, 2, fixed_bowl>>(problem, GENERATE(false, true), solver_options<>{});
}

TEST_CASE("a stepper over lbfgsb rejects a start of the wrong length at construction", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .evaluations = 0};
    check_wrong_length_surface<stepper<lbfgsb_policy<double, 2>, 2, fixed_bowl>>(problem, GENERATE(false, true), solver_options<>{});
}

TEST_CASE("a time_budget_solver over lbfgsb rejects a start of the wrong length at construction", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .evaluations = 0};
    check_wrong_length_surface<time_budget_solver<lbfgsb_policy<double, 2>, 2, fixed_bowl>>(problem, GENERATE(false, true), generous_deadline());
}

TEST_CASE("a step_and_time_budget_solver over lbfgsb rejects a start of the wrong length at construction", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .evaluations = 0};
    check_wrong_length_surface<step_and_time_budget_solver<lbfgsb_policy<double, 2>, 2, fixed_bowl>>(problem, GENERATE(false, true), generous_deadline());
}

TEST_CASE("a fixed-dimension solver built from a start of the right length behaves as before", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .evaluations = 0};
    step_budget_solver dynamic_start{lbfgsb_policy<double, 2>{}, problem, Eigen::VectorXd::Constant(2, 3.0).eval(), solver_options<>{}};
    step_budget_solver fixed_start{lbfgsb_policy<double, 2>{}, problem, vector2(3.0, 3.0), solver_options<>{}};
    CHECK(dynamic_start.status() == solver_status::running);
    CHECK(fixed_start.status() == solver_status::running);
    const auto dynamic_result = dynamic_start.solve();
    const auto fixed_result = fixed_start.solve();
    CHECK((dynamic_result.x.array() - 1.0).abs().maxCoeff() < 1e-4);
    CHECK(dynamic_result.x == fixed_result.x);
    CHECK(dynamic_result.iterations == fixed_result.iterations);
}

TEST_CASE("a dynamic-dimension solver rejects a start whose length differs from the problem dimension at construction", "[release-safety][start-length]")
{
    const dynamic_bowl problem{.n = 2, .evaluations = 0};
    for(const std::int32_t length : {1, 3})
    {
        step_budget_solver solver{lbfgsb_policy<>{}, problem, Eigen::VectorXd::Constant(length, 3.0).eval(), solver_options<>{}};
        check_construction_rejected(solver, problem);
        CHECK(solver.solve().x.size() == 0);
    }
    step_budget_solver solver{lbfgsb_policy<>{}, problem, Eigen::VectorXd::Constant(3, 3.0).eval(), solver_options<>{}};
    check_valid_reset_initializes(solver, GENERATE(false, true));
}

TEST_CASE("a valid reset after a rejected construction initializes with the caller's policy options", "[release-safety][start-length]")
{
    const bool clear = GENERATE(false, true);
    const fixed_bowl bowl{.n = 2, .evaluations = 0};
    lbfgsb_policy<double, 2>::options_type lbfgsb_opts;
    lbfgsb_opts.line_search_type = lbfgsb_line_search::armijo;
    step_budget_solver lbfgsb{lbfgsb_policy<double, 2>{}, bowl, Eigen::VectorXd::Zero(3).eval(), solver_options<>{}, lbfgsb_opts};
    check_valid_reset_initializes(lbfgsb, clear);
    CHECK(lbfgsb.policy().options.line_search_type == lbfgsb_line_search::armijo);

    const hs076<> problem;
    mma_policy<double, 4>::options_type mma_opts;
    mma_opts.raai = 1e-3;
    step_budget_solver mma{mma_policy<double, 4>{}, problem, Eigen::VectorXd::Zero(3).eval(), solver_options<>{}, mma_opts};
    CHECK(mma.status() == solver_status::invalid_problem);
    const Eigen::VectorXd start = problem.initial_point();
    CHECK((clear ? mma.reset_clear(start) : mma.reset(start)) == solver_status::running);
    CHECK(mma.state().opts.raai == 1e-3);
    CHECK_FALSE(mma.step().policy_status.has_value());
}
