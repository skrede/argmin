#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>

using namespace argmin;

namespace
{

using vec2 = Eigen::Vector<double, 2>;

// hs021 (box x0 >= 2) whose evaluations record whether any point they saw lay outside the box.
struct watched_hs021 : hs021<>
{
    watched_hs021() : left_box{false} {}

    double value(const vec2& x) const
    {
        watch(x);
        return hs021<>::value(x);
    }

    void gradient(const vec2& x, vec2& g) const
    {
        watch(x);
        hs021<>::gradient(x, g);
    }

    void constraints(const vec2& x, auto& c) const
    {
        watch(x);
        hs021<>::constraints(x, c);
    }

    void watch(const vec2& x) const { left_box = left_box || (x.array() < lower_bounds().array()).any() || (x.array() > upper_bounds().array()).any(); }

    mutable bool left_box;
};

// The publication benchmark's constrained stopping tests, under which the unprojected start ended in a false claim.
solver_options<> bench_tolerances()
{
    solver_options<> opts{};
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-6);
    opts.set_step_threshold(1e-6);
    opts.set_stationarity_threshold(1e-8);
    return opts;
}

bool in_box(const watched_hs021& problem, const vec2& x)
{
    return (x.array() >= problem.lower_bounds().array()).all() && (x.array() <= problem.upper_bounds().array()).all();
}

}

TEMPLATE_TEST_CASE("a trust-region SQP projects an out-of-box start onto the box before evaluating it",
                   "[tr_sqp]",
                   (tr_sqp_policy<double, 2, sqp_mode::fast>),
                   (tr_sqp_policy<double, 2, sqp_mode::accurate>),
                   (filter_trsqp_policy<double, 2, sqp_mode::fast>),
                   (filter_trsqp_policy<double, 2, sqp_mode::accurate>))
{
    const watched_hs021 problem;
    const vec2 outside{-1.0, 60.0};
    step_budget_solver<TestType, 2, watched_hs021> solver(problem, outside, bench_tolerances());

    SECTION("at construction")
    {
        CHECK(solver.state().x == vec2{2.0, 50.0});
        CHECK_FALSE(problem.left_box);
    }

    SECTION("on reset")
    {
        REQUIRE(solver.reset(vec2{60.0, -60.0}) == solver_status::running);
        CHECK(solver.state().x == vec2{50.0, -50.0});
        CHECK_FALSE(problem.left_box);
    }

    SECTION("and its solve from the hs021 start stays in the box and reaches the optimum on its bound")
    {
        REQUIRE(solver.reset(problem.initial_point()) == solver_status::running);
        const auto result = solver.solve();
        CHECK(result.status == solver_status::converged);
        CHECK(in_box(problem, result.x));
        CHECK(std::abs(result.objective_value - problem.optimal_value()) <= 1e-8);
        CHECK_FALSE(problem.left_box);
    }
}
