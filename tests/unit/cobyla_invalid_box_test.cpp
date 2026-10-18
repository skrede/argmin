#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

template <typename Scalar>
struct counted_box_problem
{
    using vector_type = Eigen::Vector<Scalar, 2>;

    static constexpr std::int32_t problem_dimension = 2;

    vector_type lower;
    vector_type upper;
    std::int32_t& calls;

    std::int32_t dimension() const { return 2; }

    Scalar value(const vector_type& x) const
    {
        ++calls;
        return x.squaredNorm();
    }

    void constraints(const vector_type& x, Eigen::VectorX<Scalar>& c) const
    {
        ++calls;
        c[0] = Scalar(100) - x[0] - x[1];
    }

    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 1; }
    vector_type lower_bounds() const { return lower; }
    vector_type upper_bounds() const { return upper; }
};

template <typename Scalar>
counted_box_problem<Scalar> make_problem(Eigen::Vector<Scalar, 2> lower, Eigen::Vector<Scalar, 2> upper,
                                         std::int32_t& calls)
{
    return counted_box_problem<Scalar>{.lower = lower, .upper = upper, .calls = calls};
}

template <typename Scalar>
Eigen::VectorX<Scalar> start_point()
{
    return Eigen::VectorX<Scalar>{{Scalar(0.5), Scalar(0.2)}};
}

template <typename Scalar>
bool rejected(const step_result<Scalar>& result)
{
    return result.policy_status && *result.policy_status == solver_status::invalid_problem;
}

template <typename Scalar>
using box_solver = step_budget_solver<cobyla_policy<Scalar>, 2, counted_box_problem<Scalar>,
                                      default_convergence<Scalar>>;

template <typename Scalar>
box_solver<Scalar> make_solver(const counted_box_problem<Scalar>& problem)
{
    return step_budget_solver{cobyla_policy<Scalar>{}, problem, start_point<Scalar>(),
                              solver_options<default_convergence<Scalar>>{}};
}

template <typename Scalar>
void check_rejected_at(Eigen::Vector<Scalar, 2> lower, Eigen::Vector<Scalar, 2> upper, std::int32_t coordinate)
{
    std::int32_t calls = 0;
    const auto problem = make_problem<Scalar>(lower, upper, calls);
    auto solver = make_solver(problem);
    CHECK(calls == 0);
    CHECK(rejected(solver.step()));
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.state().invalid_coordinate == coordinate);
    CHECK(calls == 0);
}

}

TEMPLATE_TEST_CASE("cobyla rejects an inverted box before evaluating anything", "[cobyla][invalid_problem]",
                   float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    check_rejected_at<TestType>(V(1, -1), V(0, 1), 0);
}

TEMPLATE_TEST_CASE("cobyla rejects a fixed coordinate and names it", "[cobyla][invalid_problem]", float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    check_rejected_at<TestType>(V(0.5, -1), V(0.5, 1), 0);
}

TEMPLATE_TEST_CASE("cobyla rejects an inverted second coordinate and names it", "[cobyla][invalid_problem]",
                   float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    check_rejected_at<TestType>(V(-1, 1), V(1, 0), 1);
}

TEMPLATE_TEST_CASE("cobyla rejects a NaN lower bound and names it", "[cobyla][invalid_problem]", float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    check_rejected_at<TestType>(V(-1, std::numeric_limits<TestType>::quiet_NaN()), V(1, 1), 1);
}

TEMPLATE_TEST_CASE("cobyla rejects an inverted box in solve with no evaluation", "[cobyla][invalid_problem]",
                   float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    std::int32_t calls = 0;
    const auto problem = make_problem<TestType>(V(1, -1), V(0, 1), calls);
    auto solver = make_solver(problem);
    const auto result = solver.solve();
    CHECK(result.status == solver_status::invalid_problem);
    CHECK(result.function_evaluations == 0);
    CHECK(result.x == V(0.5, 0.2));
    CHECK_FALSE(std::isfinite(result.objective_value));
    CHECK(calls == 0);
}

TEMPLATE_TEST_CASE("cobyla rejects an inverted box again after a reset", "[cobyla][invalid_problem]", float,
                   double)
{
    using V = Eigen::Vector<TestType, 2>;
    std::int32_t calls = 0;
    const auto problem = make_problem<TestType>(V(1, -1), V(0, 1), calls);
    auto solver = make_solver(problem);
    const Eigen::VectorX<TestType> restart{{TestType(0.3), TestType(0.4)}};
    SECTION("reset")
    {
        solver.reset(restart);
    }
    SECTION("reset_clear")
    {
        solver.reset_clear(restart);
    }
    CHECK(rejected(solver.step()));
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.state().x == restart);
    CHECK(calls == 0);
}

TEMPLATE_TEST_CASE("cobyla accepts an infinite and one-sided box", "[cobyla][invalid_problem]", float, double)
{
    using V = Eigen::Vector<TestType, 2>;
    constexpr TestType inf = std::numeric_limits<TestType>::infinity();
    std::int32_t calls = 0;
    const auto problem = make_problem<TestType>(V(-inf, 0), V(inf, inf), calls);
    auto solver = make_solver(problem);
    CHECK(calls > 0);
    CHECK(solver.state().invalid_coordinate == -1);
    CHECK_FALSE(rejected(solver.step()));
}
