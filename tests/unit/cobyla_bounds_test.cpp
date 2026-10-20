#include "cobyla_box_cases.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

using namespace argmin;
using namespace argmin::test;

namespace
{

struct box_outcome
{
    box_outcome() : clamped(1), status(solver_status::running), objective_value(0), record() {}

    std::uint32_t clamped;
    solver_status status;
    double objective_value;
    box_record<double> record;
};

template <typename P>
box_outcome solve_recorded(const P& problem, double initial_trust_radius = 0.0)
{
    box_outcome outcome;
    const box_recorder<double, P> recorder{problem, outcome.record};
    cobyla_policy<double> policy{};
    policy.options.initial_trust_radius = initial_trust_radius;
    step_budget_solver solver{policy, recorder, problem.initial_point(), publication_options<double>()};
    const auto result = solver.solve();
    outcome.clamped = solver.state().clamped_evaluations;
    outcome.status = result.status;
    outcome.objective_value = result.objective_value;
    return outcome;
}

void check_in_box(const box_outcome& outcome)
{
    CHECK(outcome.record.value_calls > 0);
    CHECK(outcome.record.outside_calls == 0);
    CHECK(outcome.clamped == 0);
}

void check_solved_in_box(const box_outcome& outcome)
{
    check_in_box(outcome);
    CHECK(succeeded(outcome.status));
    CHECK(std::abs(outcome.objective_value) < 1e-6);
}

double auto_radius(Eigen::VectorXd lower, Eigen::VectorXd upper, double supplied = 0.0)
{
    return detail::cobyla_auto_trust_radius(supplied, lower, upper, static_cast<std::int32_t>(lower.size()));
}

}

TEST_CASE("cobyla keeps every evaluation inside the box on the narrow-coordinate witness", "[cobyla][bounds]")
{
    check_solved_in_box(solve_recorded(narrow_witness<double>()));
}

TEST_CASE("cobyla keeps every evaluation inside the box on the undershoot witness", "[cobyla][bounds]")
{
    check_solved_in_box(solve_recorded(undershoot<double>()));
}

TEST_CASE("cobyla keeps the box under a supplied radius wider than the box", "[cobyla][bounds]")
{
    SECTION("narrow-coordinate witness")
    {
        check_in_box(solve_recorded(narrow_witness<double>(), 2.0));
    }
    SECTION("undershoot witness")
    {
        check_in_box(solve_recorded(undershoot<double>(), 2.0));
    }
}

TEST_CASE("cobyla projects an out-of-box start and evaluates the projection first", "[cobyla][bounds]")
{
    const box_outcome outcome = solve_recorded(start_outside<double>());
    check_solved_in_box(outcome);
    CHECK(outcome.record.first_point == Eigen::Vector2d(0.0, 10.0));
}

TEST_CASE("cobyla keeps every evaluation inside a three-scale box", "[cobyla][bounds]")
{
    check_solved_in_box(solve_recorded(three_scale<double>()));
}

TEST_CASE("cobyla keeps the box after a reset_clear and restarts its counter", "[cobyla][bounds]")
{
    box_record<double> record;
    const auto problem = narrow_witness<double>();
    const box_recorder<double, narrow_case<double>> recorder{problem, record};
    step_budget_solver solver{cobyla_policy<double>{}, recorder, problem.initial_point(),
                              publication_options<double>()};
    solver.solve();
    solver.reset_clear(Eigen::VectorXd{{0.55, 1.0}});
    CHECK(solver.state().clamped_evaluations == 0);
    solver.solve();
    CHECK(record.outside_calls == 0);
    CHECK(solver.state().clamped_evaluations == 0);
}

TEST_CASE("cobyla caps its automatic radius at half the smallest finite range", "[cobyla][bounds]")
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    CHECK(auto_radius(Eigen::Vector2d(0.0, -10.0), Eigen::Vector2d(0.6, 10.0)) == 0.3);
    CHECK(auto_radius(Eigen::Vector2d(0.0, -10.0), Eigen::Vector2d(1.0, 10.0)) == 0.5);
    CHECK(auto_radius(Eigen::Vector2d(-inf, -inf), Eigen::Vector2d(inf, inf)) == 1.0);
    CHECK(auto_radius(Eigen::Vector2d(0.0, -inf), Eigen::Vector2d(inf, inf)) == 1.0);
    CHECK(auto_radius(Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 1.0)) == 0.2);
    CHECK(auto_radius(Eigen::Vector2d(0.0, -10.0), Eigen::Vector2d(0.6, 10.0), 5.0) == 5.0);
}

TEMPLATE_TEST_CASE("cobyla evaluates every publication-cohort problem inside its box", "[cobyla][bounds]",
                   hs006<double>, hs007<double>, hs021<double>, hs023<double>, hs024<double>, hs026<double>,
                   hs027<double>, hs028<double>, hs029<double>, hs030<double>, hs031<double>, hs034<double>,
                   hs035<double>, hs036<double>, hs037<double>, hs039<double>, hs040<double>, hs043<double>,
                   hs044<double>, hs048<double>, hs050<double>, hs051<double>, hs052<double>, hs071<double>,
                   hs076<double>)
{
    check_in_box(solve_recorded(TestType{}));
}
