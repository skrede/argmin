#include "oracle/state_machine.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace argmin;

namespace
{

struct gradient_counting_rosenbrock : rosenbrock<double, 2>
{
    gradient_counting_rosenbrock()
        : gradients{0}
    {}

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        ++gradients;
        rosenbrock<double, 2>::gradient(x, g);
    }

    mutable std::uint32_t gradients;
};

using counting_solver = step_budget_solver<lbfgsb_policy<double, 2>, 2, gradient_counting_rosenbrock>;

counting_solver solver_at_start(const gradient_counting_rosenbrock& problem)
{
    return {lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), solver_options<>{}, lbfgsb_policy<double, 2>::options_type{}};
}

template <typename Solve>
solve_result<double, 2> solve_counting_gradients(const gradient_counting_rosenbrock& problem, Solve&& solve)
{
    const std::uint32_t before = problem.gradients;
    const solve_result<double, 2> result = solve();
    CHECK(problem.gradients == before);
    return result;
}

}

TEST_CASE("a solve aborted before its first step reports no gradient norm and evaluates none", "[oracle][state-machine]")
{
    const gradient_counting_rosenbrock problem;
    counting_solver solver = solver_at_start(problem);
    solver.abort();
    const solve_result<double, 2> result = solve_counting_gradients(problem, [&] { return solver.solve(); });
    CHECK(result.status == solver_status::aborted);
    CHECK(result.iterations == 0);
    CHECK(std::isnan(result.gradient_norm));
}

TEST_CASE("a zero-budget solve reports no gradient norm and evaluates none", "[oracle][state-machine]")
{
    const gradient_counting_rosenbrock problem;
    counting_solver solver = solver_at_start(problem);
    const solve_result<double, 2> result = solve_counting_gradients(problem, [&] { return solver.step_n(0); });
    CHECK(result.status == solver_status::budget_exhausted);
    CHECK(result.iterations == 0);
    CHECK(std::isnan(result.gradient_norm));
}

TEST_CASE("a solve from a latched status returns the terminal solve's result and evaluates none", "[oracle][state-machine]")
{
    const gradient_counting_rosenbrock problem;
    counting_solver solver = solver_at_start(problem);
    const solve_result<double, 2> first = solver.solve();
    REQUIRE(oracle::is_claim(first.status));
    const solve_result<double, 2> result = solve_counting_gradients(problem, [&] { return solver.solve(); });
    CHECK(result.status == first.status);
    CHECK(result.iterations == first.iterations);
    CHECK(result.function_evaluations == first.function_evaluations);
    CHECK(oracle::same_bits(Eigen::VectorXd(result.x), Eigen::VectorXd(first.x)));
    CHECK(oracle::same_bits(result.objective_value, first.objective_value));
    CHECK(oracle::same_bits(result.constraint_violation, first.constraint_violation));
    CHECK(oracle::same_bits(result.gradient_norm, first.gradient_norm));
}
