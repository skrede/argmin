#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp;

namespace
{

// Bound-constrained Rosenbrock WITHOUT gradient -- BOBYQA test problem.
// Satisfies objective && bound_constrained but NOT differentiable.
struct bobyqa_rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;
    int n{2};
    double a{1};
    double b{5};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < n - 1; ++i)
        {
            double t1 = a - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + b * t2 * t2;
        }
        return f;
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// Booth function: (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2
// Minimum at (1, 3), f* = 0.
struct bobyqa_booth
{
    static constexpr int problem_dimension = 2;
    Eigen::Vector<double, 2> lb;
    Eigen::Vector<double, 2> ub;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = x[0] + 2.0 * x[1] - 7.0;
        double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const { return lb; }
    Eigen::Vector<double, 2> upper_bounds() const { return ub; }
};

// 6D sum of 2D Rosenbrock pairs (matching liepp joint space dimension).
struct bobyqa_rosenbrock_6d
{
    static constexpr int problem_dimension = 6;
    Eigen::Vector<double, 6> lb;
    Eigen::Vector<double, 6> ub;

    int dimension() const { return 6; }

    double value(const Eigen::Vector<double, 6>& x) const
    {
        double f = 0.0;
        for(int i = 0; i < 5; ++i)
        {
            double t1 = 1.0 - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + 5.0 * t2 * t2;
        }
        return f;
    }

    Eigen::Vector<double, 6> lower_bounds() const { return lb; }
    Eigen::Vector<double, 6> upper_bounds() const { return ub; }
};

}

// Concept satisfaction: BOBYQA test problems satisfy objective && bound_constrained
// but NOT differentiable (no gradient method).
static_assert(objective<bobyqa_rosenbrock>);
static_assert(bound_constrained<bobyqa_rosenbrock>);
static_assert(!differentiable<bobyqa_rosenbrock>);

static_assert(objective<bobyqa_booth>);
static_assert(bound_constrained<bobyqa_booth>);
static_assert(!differentiable<bobyqa_booth>);

static_assert(objective<bobyqa_rosenbrock_6d>);
static_assert(bound_constrained<bobyqa_rosenbrock_6d>);
static_assert(!differentiable<bobyqa_rosenbrock_6d>);

TEST_CASE("bobyqa concept satisfaction", "[bobyqa]")
{
    SECTION("test problems satisfy correct concepts")
    {
        // Compile-time checks above; runtime section for test discovery.
        SUCCEED("bobyqa_rosenbrock satisfies objective && bound_constrained && !differentiable");
    }
}

TEST_CASE("bobyqa bound-constrained Rosenbrock 2D", "[bobyqa]")
{
    bobyqa_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);  // effectively disabled for derivative-free
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.x[0] == Approx(1.0).margin(5e-3));
    CHECK(result.x[1] == Approx(1.0).margin(5e-3));
    CHECK(result.objective_value < 1e-3);

    // Bounds must be respected
    CHECK(result.x[0] >= -5.0 - 1e-10);
    CHECK(result.x[0] <= 5.0 + 1e-10);
    CHECK(result.x[1] >= -5.0 - 1e-10);
    CHECK(result.x[1] <= 5.0 + 1e-10);
}

TEST_CASE("bobyqa tight bounds", "[bobyqa]")
{
    bobyqa_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{0.0, 0.0}},
        .ub = Eigen::VectorXd{{0.8, 0.8}},
    };

    Eigen::VectorXd x0{{0.4, 0.4}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // Solution must be within bounds
    CHECK(result.x[0] >= -1e-10);
    CHECK(result.x[0] <= 0.8 + 1e-10);
    CHECK(result.x[1] >= -1e-10);
    CHECK(result.x[1] <= 0.8 + 1e-10);

    // Should improve over starting point
    CHECK(result.objective_value < problem.value(x0));
}

TEST_CASE("bobyqa Booth function", "[bobyqa]")
{
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // Booth minimum at (1, 3)
    CHECK(result.x[0] == Approx(1.0).margin(1e-3));
    CHECK(result.x[1] == Approx(3.0).margin(1e-3));
    CHECK(result.objective_value < 1e-4);
}

TEST_CASE("bobyqa higher dimensional (n=6)", "[bobyqa]")
{
    bobyqa_rosenbrock_6d problem{
        .lb = Eigen::Vector<double, 6>::Constant(-2.0),
        .ub = Eigen::Vector<double, 6>::Constant(2.0),
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, -0.5);
    solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // Should converge reasonably close to (1,1,...,1)
    for(int i = 0; i < 6; ++i)
    {
        CHECK(result.x[i] == Approx(1.0).margin(0.1));
        CHECK(result.x[i] >= -2.0 - 1e-10);
        CHECK(result.x[i] <= 2.0 + 1e-10);
    }

    CHECK(result.objective_value < 1.0);
}

TEST_CASE("bobyqa step solve step_n", "[bobyqa]")
{
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    SECTION("step returns finite values")
    {
        basic_solver solver{bobyqa_policy{}, problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.step_size));
        }
    }

    SECTION("step_n with budget")
    {
        basic_solver solver{bobyqa_policy{}, problem, x0, opts};
        auto result = solver.step_n(50);
        CHECK(std::isfinite(result.objective_value));
        CHECK(result.iterations <= 50);
    }

    SECTION("solve converges")
    {
        basic_solver solver{bobyqa_policy{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value < 1e-2);
    }
}

TEST_CASE("bobyqa fixed-dimension crash regression", "[bobyqa]")
{
    // Regression guard: prior to JacobiSVD fix, BOBYQA would SIGABRT when
    // BDCSVD was applied to the small Vandermonde-like matrix Phi in
    // quadratic_model::build_model. This test verifies no crash occurs.
    bobyqa_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_objective_threshold(1e-6);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.status != solver_status::diverged);
    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("bobyqa custom interpolation points", "[bobyqa]")
{
    bobyqa_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    SECTION("default 2n+1 interpolation points")
    {
        basic_solver solver{bobyqa_policy{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value < 1e-3);
    }
}
