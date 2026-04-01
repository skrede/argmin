#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Bound-constrained Rosenbrock wrapper for testing box constraints.
struct bounded_rosenbrock
{
    nablapp::rosenbrock<> inner{.n = 2};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return inner.dimension(); }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { inner.gradient(x, g); }
    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

// Concept satisfaction checks
static_assert(differentiable<bounded_rosenbrock>);
static_assert(bound_constrained<bounded_rosenbrock>);
static_assert(differentiable<rosenbrock<>>);
static_assert(!bound_constrained<rosenbrock<>>);

TEST_CASE("lbfgsb_policy satisfies policy contract", "[lbfgsb]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;

    SECTION("init and step compile and return finite values")
    {
        lbfgsb_policy policy;
        auto state = policy.init(problem, x0, opts);
        auto sr = policy.step(state);

        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
        CHECK(std::isfinite(sr.step_size));
        CHECK(std::isfinite(sr.objective_change));
    }

    SECTION("reset preserves curvature")
    {
        lbfgsb_policy policy;
        auto state = policy.init(problem, x0, opts);

        for(int i = 0; i < 5; ++i)
            policy.step(state);

        int curvature_size = state.B.size();
        CHECK(curvature_size > 0);

        Eigen::VectorXd new_x0{{0.5, 0.5}};
        policy.reset(state, new_x0);

        CHECK(state.B.size() == curvature_size);
        CHECK(state.x.isApprox(new_x0));
        CHECK(state.iteration == 0);
    }

    SECTION("reset_clear clears curvature")
    {
        lbfgsb_policy policy;
        auto state = policy.init(problem, x0, opts);

        for(int i = 0; i < 5; ++i)
            policy.step(state);

        CHECK(state.B.size() > 0);

        Eigen::VectorXd new_x0{{0.5, 0.5}};
        policy.reset_clear(state, new_x0);

        CHECK(state.B.size() == 0);
        CHECK(state.x.isApprox(new_x0));
    }
}

TEST_CASE("L-BFGS-B converges on unconstrained Rosenbrock", "[lbfgsb]")
{
    SECTION("2D Rosenbrock")
    {
        rosenbrock<> problem{.n = 2};
        Eigen::VectorXd x0{{-1.2, 1.0}};
        solver_options opts;
        opts.gradient_tolerance = 1e-6;
        opts.max_iterations = 200;

        basic_solver<lbfgsb_policy> solver{problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.status == solver_status::converged);
        CHECK(result.x[0] == Approx(1.0).margin(1e-4));
        CHECK(result.x[1] == Approx(1.0).margin(1e-4));
        CHECK(result.objective_value < 1e-8);
    }

    SECTION("10D Rosenbrock")
    {
        rosenbrock<> problem{.n = 10};
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(10, -1.0);
        solver_options opts;
        opts.gradient_tolerance = 1e-6;
        opts.max_iterations = 500;

        basic_solver<lbfgsb_policy> solver{problem, x0, opts};
        auto result = solver.solve();

        CHECK(result.status == solver_status::converged);
        for(int i = 0; i < 10; ++i)
            CHECK(result.x[i] == Approx(1.0).margin(1e-3));
        CHECK(result.objective_value < 1e-6);
    }
}

TEST_CASE("L-BFGS-B respects box constraints", "[lbfgsb]")
{
    constexpr double inf = std::numeric_limits<double>::infinity();

    SECTION("bounds exclude optimum -- solution at boundary")
    {
        // x[0] constrained to [0.5, 0.8]; true min at x[0]=1.0 excluded.
        bounded_rosenbrock problem{
            .inner = {.n = 2},
            .lb = Eigen::VectorXd{{0.5, -inf}},
            .ub = Eigen::VectorXd{{0.8, inf}},
        };

        Eigen::VectorXd x0{{0.6, 0.5}};
        solver_options opts;
        opts.gradient_tolerance = 1e-6;
        opts.max_iterations = 500;
        opts.step_tolerance = 1e-15;
        opts.objective_tolerance = 1e-15;

        basic_solver<lbfgsb_policy> solver{problem, x0, opts};
        auto result = solver.solve();

        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached
               || result.status == solver_status::stalled
               || result.status == solver_status::max_iterations));

        // Solution must respect bounds
        CHECK(result.x[0] >= 0.5 - 1e-10);
        CHECK(result.x[0] <= 0.8 + 1e-10);

        // The constrained optimum satisfies the KKT conditions.
        // With b=5, objective is (1-x0)^2 + 5*(x1-x0^2)^2.
        // The solution should be better than the starting point.
        CHECK(result.objective_value < problem.value(x0));
    }

    SECTION("loose bounds -- converges to unconstrained optimum")
    {
        bounded_rosenbrock problem{
            .inner = {.n = 2},
            .lb = Eigen::VectorXd{{-10.0, -10.0}},
            .ub = Eigen::VectorXd{{10.0, 10.0}},
        };

        Eigen::VectorXd x0{{-1.2, 1.0}};
        solver_options opts;
        opts.gradient_tolerance = 1e-5;
        opts.max_iterations = 500;
        opts.step_tolerance = 1e-15;
        opts.objective_tolerance = 1e-15;

        basic_solver<lbfgsb_policy> solver{problem, x0, opts};
        auto result = solver.solve();

        CHECK((result.status == solver_status::converged
               || result.status == solver_status::stalled));
        CHECK(result.x[0] == Approx(1.0).margin(1e-2));
        CHECK(result.x[1] == Approx(1.0).margin(1e-2));
    }
}

TEST_CASE("L-BFGS-B step/solve/step_n consistency", "[lbfgsb]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;
    opts.max_iterations = 200;

    SECTION("step returns finite values")
    {
        basic_solver<lbfgsb_policy> solver{problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
            CHECK(std::isfinite(sr.step_size));
        }

        auto result = solver.step_n(100);
        CHECK(std::isfinite(result.objective_value));
    }

    SECTION("solve converges from scratch")
    {
        basic_solver<lbfgsb_policy> solver{problem, x0, opts};
        auto result = solver.solve();
        CHECK(result.status == solver_status::converged);
    }
}

TEST_CASE("Two L-BFGS-B policies in solver_group (SC5)", "[lbfgsb][solver_group]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.gradient_tolerance = 1e-6;

    basic_solver_group<round_robin_schedule, lbfgsb_policy, lbfgsb_policy> group{
        problem, x0, opts};

    // Step 10 times, each step should produce finite results
    for(int i = 0; i < 10; ++i)
    {
        auto sr = group.step();
        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
    }

    // Run to convergence or budget
    auto result = group.step_n(200);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::budget_exhausted));
    CHECK(std::isfinite(result.objective_value));
    CHECK(result.x.size() == 2);
}
