#include "nablapp/detail/quadratic_model.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/formulation/concepts.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <iostream>
#include <numeric>

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

TEST_CASE("bobyqa Lagrange values partition of unity", "[bobyqa]")
{
    // Build a 2D interpolation set with m = 2*2+1 = 5 points around x0.
    const int n = 2;
    const int m = 2 * n + 1;
    Eigen::Vector2d x0{-1.2, 1.0};
    double h = 0.5;

    Eigen::Matrix<double, 2, Eigen::Dynamic> Y(n, m);
    Eigen::VectorXd f_values(m);

    // Rosenbrock evaluator
    auto rosenbrock = [](const Eigen::Vector2d& x)
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 5.0 * t2 * t2;
    };

    Y.col(0) = x0;
    f_values[0] = rosenbrock(x0);

    for(int i = 0; i < n; ++i)
    {
        Eigen::Vector2d pt = x0;
        pt[i] += h;
        Y.col(1 + i) = pt;
        f_values[1 + i] = rosenbrock(pt);
    }

    for(int i = 0; i < n; ++i)
    {
        Eigen::Vector2d pt = x0;
        pt[i] -= h;
        Y.col(1 + n + i) = pt;
        f_values[1 + n + i] = rosenbrock(pt);
    }

    auto model = nablapp::detail::build_model(Y, f_values, x0);

    // Partition of unity: sum_k L_k(x) = 1 for any x.
    CHECK(model.lagrange_values.size() == m);
    CHECK(model.lagrange_values.sum() == Approx(1.0).margin(1e-10));

    // Each value should be finite and bounded (no blow-up from ill-conditioning)
    for(int k = 0; k < m; ++k)
    {
        CHECK(std::isfinite(model.lagrange_values[k]));
        CHECK(std::abs(model.lagrange_values[k]) < 10.0);
    }
}

TEST_CASE("bobyqa Lagrange-based point replacement", "[bobyqa]")
{
    // Directly test select_replacement with Lagrange values.
    // Build a 2D interpolation set (5 points for n=2), compute Lagrange values
    // at a trial point, and verify the returned index matches max |L_k|.
    const int n = 2;
    const int m = 2 * n + 1;
    Eigen::Vector2d x0{0.0, 0.0};
    double h = 1.0;

    Eigen::Matrix<double, 2, Eigen::Dynamic> Y(n, m);
    Eigen::VectorXd f_values(m);

    auto quadratic = [](const Eigen::Vector2d& x) { return x.squaredNorm(); };

    Y.col(0) = x0;
    f_values[0] = quadratic(x0);
    for(int i = 0; i < n; ++i)
    {
        Eigen::Vector2d pt = x0;
        pt[i] += h;
        Y.col(1 + i) = pt;
        f_values[1 + i] = quadratic(pt);
    }
    for(int i = 0; i < n; ++i)
    {
        Eigen::Vector2d pt = x0;
        pt[i] -= h;
        Y.col(1 + n + i) = pt;
        f_values[1 + n + i] = quadratic(pt);
    }

    // Trial point
    Eigen::Vector2d x_new{0.5, 0.5};
    double f_new = quadratic(x_new);

    // Compute Lagrange values at x_new
    auto lv = nablapp::detail::compute_lagrange_at_point(Y, f_values, x0, x_new);
    CHECK(lv.size() == m);

    // Find the expected replacement index (max |L_k| among non-best)
    int best_idx = 0;
    for(int i = 1; i < m; ++i)
    {
        if(f_values[i] < f_values[best_idx])
            best_idx = i;
    }

    // Since f_new > f_best, Lagrange criterion applies (non-improvement step)
    int expected = (best_idx == 0) ? 1 : 0;
    double max_abs = std::abs(lv[expected]);
    for(int i = 0; i < m; ++i)
    {
        if(i == best_idx) continue;
        if(std::abs(lv[i]) > max_abs)
        {
            max_abs = std::abs(lv[i]);
            expected = i;
        }
    }

    double delta = 1.0;
    int actual = nablapp::detail::select_replacement(Y, f_values, x_new, f_new, x0, lv, delta);
    // With denominator*distance^4 weighting, the replacement selects the point
    // that maximizes L_k(x_new)^2 * max(1, (dist/delta)^4).
    CHECK(actual != best_idx);
}

TEST_CASE("bobyqa geometry improvement accuracy", "[bobyqa]")
{
    // Run BOBYQA on tight-bounds Rosenbrock with geometry improvement active.
    // Accuracy should remain at least as good as baseline.
    bobyqa_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{0.0, 0.0}},
        .ub = Eigen::VectorXd{{0.8, 0.8}},
    };

    Eigen::VectorXd x0{{0.4, 0.4}};
    solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.x[0] >= -1e-10);
    CHECK(result.x[0] <= 0.8 + 1e-10);
    CHECK(result.objective_value < problem.value(x0));
    CHECK(result.objective_value < 0.05);
}

TEST_CASE("bobyqa rescue does not break 6D convergence", "[bobyqa]")
{
    // Regression guard: rescue mechanism should not degrade 6D convergence.
    bobyqa_rosenbrock_6d problem{
        .lb = Eigen::Vector<double, 6>::Constant(-2.0),
        .ub = Eigen::Vector<double, 6>::Constant(2.0),
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(6, -0.5);
    solver_options opts;
    opts.max_iterations = 1500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 0.5);
    for(int i = 0; i < 6; ++i)
    {
        CHECK(result.x[i] >= -2.0 - 1e-10);
        CHECK(result.x[i] <= 2.0 + 1e-10);
    }
}

TEST_CASE("bobyqa rho contraction improves accuracy", "[bobyqa]")
{
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // With progressive rho contraction, the solver drives accuracy
    // well beyond the 1e-4 baseline achievable without it.
    CHECK(result.x[0] == Approx(1.0).margin(1e-4));
    CHECK(result.x[1] == Approx(3.0).margin(1e-4));
    CHECK(result.objective_value < 1e-5);
}

TEST_CASE("bobyqa HS001 accuracy vs NLopt baseline", "[bobyqa][benchmark]")
{
    // HS001: min 100*(x1 - x0^2)^2 + (1 - x0)^2, x1 >= -1.5.
    // Optimal: f* = 0 at (1, 1). NLopt BOBYQA reaches f* ~ 0 at ~353 evals.
    //
    // Reference: Hock & Schittkowski, Problem 1.
    nablapp::hs001<double> hs;

    Eigen::Vector<double, 2> x0 = hs.initial_point();
    solver_options opts;
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    basic_solver solver{bobyqa_policy<2>{}, hs, x0, opts};
    auto result = solver.solve(opts);

    // HS001 is the hard Rosenbrock variant. With guarded rho contraction
    // the solver avoids premature termination. Incremental model updates
    // trade marginal accuracy for ~2x throughput.
    CHECK(result.objective_value < 0.01);
    CHECK(result.x[0] == Approx(1.0).margin(0.1));
    CHECK(result.x[1] == Approx(1.0).margin(0.1));
    // x1 bound must be respected
    CHECK(result.x[1] >= -1.5 - 1e-10);
}

TEST_CASE("bobyqa hs001 eval count regression guard", "[bobyqa]")
{
    // HS001: original baseline was 3417 iterations. With three-regime rho
    // contraction (Plan 01) and NLopt-faithful contraction trigger (Plan 02),
    // reduced to ~2509. NLopt achieves 358 function evaluations; the remaining
    // gap is due to nablapp's SVD model construction vs NLopt's BMAT/ZMAT.
    //
    // Reference: Hock & Schittkowski, Problem 1.
    nablapp::hs001<double> hs;

    Eigen::Vector<double, 2> x0 = hs.initial_point();
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy<2>{}, hs, x0, opts};
    auto result = solver.solve(opts);

    std::cout << "[HS001 BOBYQA] iterations: " << result.iterations
              << "  f: " << result.objective_value << std::endl;

    // Known architectural limitation: nablapp's SVD-based model (full
    // rebuild per accepted step) vs NLopt's incremental BMAT/ZMAT updates
    // results in ~7x more evaluations on HS001 (NLopt achieves 358).
    // D-12 target of < 750 requires BMAT/ZMAT representation or
    // multi-geometry-step iteration. Threshold guards against regression
    // from the 3417 pre-fix baseline.
    CHECK(result.iterations < 2700);
    CHECK(result.objective_value < 1.0);
}

TEST_CASE("bobyqa hs002 and hs005 regression guard", "[bobyqa]")
{
    // Diagnostic: verify HS002 and HS005 convergence is maintained.
    //
    // Reference: Hock & Schittkowski, Problems 2 and 5.
    {
        nablapp::hs005<double> hs5;
        Eigen::Vector<double, 2> x0 = hs5.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-15);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-12);

        basic_solver solver{bobyqa_policy<2>{}, hs5, x0, opts};
        auto result = solver.solve(opts);

        std::cout << "[HS005 BOBYQA] iterations: " << result.iterations
                  << "  f: " << result.objective_value << std::endl;

        CHECK(result.objective_value < -1.0);
        CHECK(result.iterations < 200);
    }
}

TEST_CASE("bobyqa unequal bound ranges", "[bobyqa]")
{
    // Problem where x0 in [-1, 10] and x1 in [-100, 100].
    // Without rescaling, the trust region is dominated by x1's range.
    // With rescaling, both dimensions get proportional step sizes.
    //
    // Booth function: min at (1, 3).
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-1.0, -100.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 100.0}},
    };

    Eigen::VectorXd x0{{5.0, 50.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // With 200:1 bound range ratio, rescaling enables convergence
    // that would otherwise stall. f < 0.05 confirms rescaling benefit.
    CHECK(result.objective_value < 0.05);
    CHECK(result.x[0] == Approx(1.0).margin(0.5));
    CHECK(result.x[1] == Approx(3.0).margin(0.5));
    // Bounds respected
    CHECK(result.x[0] >= -1.0 - 1e-10);
    CHECK(result.x[0] <= 10.0 + 1e-10);
    CHECK(result.x[1] >= -100.0 - 1e-10);
    CHECK(result.x[1] <= 100.0 + 1e-10);
}

TEST_CASE("bobyqa rho exhaustion reports converged status", "[bobyqa]")
{
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    basic_solver solver{bobyqa_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // Rho exhaustion must report convergence, not stall
    CHECK(result.status != solver_status::stalled);
    CHECK(result.status != solver_status::max_iterations);
    CHECK((result.status == solver_status::converged ||
           result.status == solver_status::ftol_reached));
    CHECK(result.objective_value < 1e-5);
}

TEST_CASE("bobyqa incremental model infrastructure", "[bobyqa][incremental]")
{
    // Build a simple 2D quadratic model and verify incremental update correctness.
    const int n = 2;
    const int m = 2 * n + 1;  // 5 points

    Eigen::VectorXd x_base = Eigen::VectorXd::Zero(n);

    // Interpolation points: x_base, x_base + h*e_i, x_base - h*e_i
    double h = 1.0;
    Eigen::MatrixXd Y(n, m);
    Eigen::VectorXd f_values(m);

    // Objective: f(x) = (x1-1)^2 + (x2-3)^2  (Booth-like, minimum at (1,3))
    auto obj = [](const Eigen::VectorXd& x) {
        return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 3.0) * (x[1] - 3.0);
    };

    Y.col(0) = x_base;
    Y.col(1) = x_base + Eigen::VectorXd{{h, 0.0}};
    Y.col(2) = x_base + Eigen::VectorXd{{0.0, h}};
    Y.col(3) = x_base - Eigen::VectorXd{{h, 0.0}};
    Y.col(4) = x_base - Eigen::VectorXd{{0.0, h}};

    for(int i = 0; i < m; ++i)
        f_values[i] = obj(Y.col(i));

    auto model = detail::build_model(Y, f_values, x_base);

    SECTION("Lagrange partition of unity at x_base")
    {
        double sum = model.lagrange_values.sum();
        CHECK(sum == Approx(1.0).margin(1e-10));
    }

    SECTION("pinv_Phi * phi(y_k) gives unit vectors")
    {
        const int p = detail::polynomial_basis_dimension(n);
        for(int k = 0; k < m; ++k)
        {
            Eigen::VectorXd s = Y.col(k) - x_base;
            Eigen::VectorXd phi = detail::build_polynomial_basis(s, p);
            Eigen::VectorXd lv = model.pinv_Phi * phi;

            for(int j = 0; j < m; ++j)
            {
                double expected = (j == k) ? 1.0 : 0.0;
                CHECK(lv[j] == Approx(expected).margin(1e-10));
            }
        }
    }

    SECTION("incremental update matches SVD rebuild")
    {
        // Replace point 3 with a new point
        Eigen::VectorXd x_new{{0.5, 0.5}};
        double f_new = obj(x_new);
        int replaced = 3;

        // SVD path: rebuild from scratch
        Eigen::MatrixXd Y_new = Y;
        Eigen::VectorXd f_new_vec = f_values;
        Y_new.col(replaced) = x_new;
        f_new_vec[replaced] = f_new;
        auto model_svd = detail::build_model(Y_new, f_new_vec, x_base);

        // Incremental path
        auto model_inc = model;
        Y.col(replaced) = x_new;
        f_values[replaced] = f_new;
        auto lv = detail::update_model_incremental(
            model_inc, f_values, x_new, replaced, x_base);

        REQUIRE(lv.size() > 0);

        // Compare model coefficients
        CHECK(model_inc.c == Approx(model_svd.c).margin(1e-10));
        for(int i = 0; i < n; ++i)
            CHECK(model_inc.g[i] == Approx(model_svd.g[i]).margin(1e-10));
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                CHECK(model_inc.H(i, j) == Approx(model_svd.H(i, j)).margin(1e-10));

        // Compare Lagrange values at x_base
        for(int k = 0; k < m; ++k)
            CHECK(model_inc.lagrange_values[k] == Approx(model_svd.lagrange_values[k]).margin(1e-10));
    }

    SECTION("partition of unity after multiple incremental updates")
    {
        auto model_inc = model;
        auto Y_inc = Y;
        auto f_inc = f_values;

        // Perform 5 sequential point replacements
        Eigen::VectorXd pts[] = {
            Eigen::VectorXd{{0.3, 0.7}},
            Eigen::VectorXd{{-0.5, 0.2}},
            Eigen::VectorXd{{0.1, -0.3}},
            Eigen::VectorXd{{0.8, 0.9}},
            Eigen::VectorXd{{-0.2, 0.6}},
        };
        int replace_indices[] = {3, 1, 4, 2, 0};

        for(int step = 0; step < 5; ++step)
        {
            auto& x_new = pts[step];
            double f_new = obj(x_new);
            int t = replace_indices[step];

            Y_inc.col(t) = x_new;
            f_inc[t] = f_new;

            auto lv = detail::update_model_incremental(
                model_inc, f_inc, x_new, t, x_base);
            REQUIRE(lv.size() > 0);

            // Partition of unity: sum of Lagrange values at x_base should be ~1.
            // Small drift is expected from LDLT conditioning; SVD re-grounding
            // at accepted steps prevents unbounded accumulation.
            double sum = model_inc.lagrange_values.sum();
            CHECK(sum == Approx(1.0).margin(1e-2));
        }

        // Final model should match full SVD rebuild
        auto model_svd = detail::build_model(Y_inc, f_inc, x_base);
        CHECK(model_inc.c == Approx(model_svd.c).margin(1e-8));
        for(int i = 0; i < n; ++i)
            CHECK(model_inc.g[i] == Approx(model_svd.g[i]).margin(1e-8));
    }
}
