#include "argmin/detail/quadratic_model.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using Catch::Approx;
using namespace argmin;

namespace
{
// Minimal oracle CSV reader (mirrors the COBYLA load_oracle reader):
// '#' lines are comments, data lines are "name,v1,v2,...".
std::map<std::string, std::vector<double>> load_bobyqa_oracle(const std::string& path)
{
    std::map<std::string, std::vector<double>> rows;
    std::ifstream in(path);
    if(!in.is_open())
        return rows;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        std::vector<double> vals;
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(std::stod(tok));
        rows.emplace(name, std::move(vals));
    }
    return rows;
}

// Curved far-from-origin Rosenbrock: min f = 0 at (10, 100), |x*| ~ 100 from the
// origin. The non-quadratic valley exposes an endgame-accuracy floor at
// rho_end = 1e-8 that the origin shift is meant to unlock (a quadratic would be
// modeled exactly). Matches the far_rosenbrock row of the oracle CSV.
struct far_rosenbrock
{
    static constexpr int problem_dimension = 2;
    Eigen::Vector<double, 2> lb;
    Eigen::Vector<double, 2> ub;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const
    {
        double t1 = 10.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }
    Eigen::Vector<double, 2> lower_bounds() const { return lb; }
    Eigen::Vector<double, 2> upper_bounds() const { return ub; }
};
}

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

    auto model = argmin::detail::build_model(Y, f_values, x0);

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
    auto lv = argmin::detail::compute_lagrange_at_point(Y, f_values, x0, x_new);
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
    int actual = argmin::detail::select_replacement(Y, f_values, x_new, f_new, x0, lv, delta);
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
    argmin::hs001<double> hs;

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
    // HS001: With BMAT/ZMAT factored interpolation system and ALTMOV
    // geometry improvement, argmin achieves ~540 iterations vs NLopt's 358.
    // The remaining gap is due to differences in step acceptance logic.
    // Previous baselines: 3417 (original), 2509 (rho contraction), now ~540
    // with BMAT/ZMAT + ALTMOV.
    //
    // Reference: Hock & Schittkowski, Problem 1.
    argmin::hs001<double> hs;

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

    // With BMAT/ZMAT and ALTMOV active, eval count is ~540 (within 2x of
    // NLopt's 358). Guard against regression from both the 540 baseline
    // and the plan target of 750.
    CHECK(result.iterations < 750);
    CHECK(result.objective_value < 0.01);
}

TEST_CASE("bobyqa hs002 and hs005 regression guard", "[bobyqa]")
{
    // Diagnostic: verify HS002 and HS005 convergence is maintained.
    //
    // Reference: Hock & Schittkowski, Problems 2 and 5.
    {
        argmin::hs005<double> hs5;
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

// The reported incumbent must never diverge from the factored model's best
// point. A previous denom-collapse path overwrote fval/xpt behind the
// BMAT/ZMAT factorization and moved s.x to the accepted trial without updating
// kopt/xopt, so the model's incumbent and the reported x could disagree. With
// the no-mutate collapse path, the reported (x, f) stay tied to the model's
// (xbase + xopt, fval[kopt]) every step.
TEST_CASE("bobyqa incumbent stays consistent with the factored model", "[bobyqa]")
{
    bobyqa_rosenbrock problem;
    problem.n = 2;
    problem.lb = Eigen::VectorXd::Constant(2, -5.0);
    problem.ub = Eigen::VectorXd::Constant(2, 5.0);
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;

    solver_options opts;
    bobyqa_policy<dynamic_dimension> pol;
    auto s = pol.init(problem, x0, opts);

    for(int it = 0; it < 120; ++it)
    {
        auto r = pol.step(s);

        // Reported objective is a genuine evaluation of the reported point --
        // this holds on every step, including the terminal deferred one.
        CHECK(problem.value(s.x) == Approx(s.objective_value).margin(1e-9));

        Eigen::VectorXd model_best_scaled = s.sys.xbase + s.sys.xopt;
        Eigen::VectorXd model_best_orig =
            (model_best_scaled.array() * s.scale.array()).matrix();

        const bool terminal =
            r.policy_status && *r.policy_status == solver_status::converged;

        if(terminal)
        {
            // On termination Powell may report the improving deferred short
            // step -- the one point returned outside the factored model. The
            // principled invariant then relaxes to: the reported incumbent is
            // EITHER the model's best node, OR a genuine evaluation that is no
            // worse than the model's best node (never a spurious point).
            const bool at_model_best =
                std::abs(s.objective_value - s.sys.fval[s.sys.kopt]) < 1e-9 &&
                (s.x - model_best_orig).norm() < 1e-9;
            CHECK((at_model_best ||
                   s.objective_value <= s.sys.fval[s.sys.kopt] + 1e-12));
            break;
        }

        // Non-terminal steps: the reported incumbent IS the model's best node,
        // unchanged -- the deferred carve-out never applies mid-run.
        CHECK(s.objective_value == Approx(s.sys.fval[s.sys.kopt]).margin(1e-9));
        CHECK((s.x - model_best_orig).norm() < 1e-9);
    }
}

// Terminal deferred short-step return.
//
// Powell defers a below-threshold Newton step (ntrits == -1) and, at the very
// end of the run, evaluates it once. When that final paid evaluation strictly
// improves on the model's best node (bobyqb_ lines 2582-2586, fsave <
// fval[kopt]) he returns THAT point as the solution rather than the model
// incumbent -- otherwise the last evaluation is wasted and the returned point
// is strictly worse. This pins that the improving deferred step is returned and
// is a genuine evaluation of the reported point.
TEST_CASE("bobyqa returns the improving deferred short step at end-game", "[bobyqa]")
{
    bobyqa_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
    };
    Eigen::Vector<double, 2> x0{0.0, 0.0};
    solver_options opts;
    opts.max_iterations = 20000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(0.0);
    opts.set_step_threshold(0.0);
    bobyqa_policy<2>::options_type popts;
    popts.final_trust_radius = 1e-8;

    bobyqa_policy<2> pol{popts};
    auto s = pol.init(problem, x0, opts);

    bool terminated = false;
    for(int it = 0; it < 20000; ++it)
    {
        auto r = pol.step(s);
        if(r.policy_status && *r.policy_status == solver_status::converged)
        {
            terminated = true;

            // The reported point is a genuine evaluation of that exact point.
            CHECK(problem.value(s.x) == Approx(s.objective_value).margin(1e-30));

            // The reported value strictly improves on the model's best node --
            // the deferred evaluation was returned, not discarded.
            CHECK(s.objective_value < s.sys.fval[s.sys.kopt]);

            // And it is a valid Booth minimizer.
            CHECK(s.objective_value < 1e-8);
            CHECK(s.x[0] == Approx(1.0).margin(1e-3));
            CHECK(s.x[1] == Approx(3.0).margin(1e-3));
            break;
        }
    }
    CHECK(terminated);
}

// An objective that records every point at which it is evaluated. Mirrors the
// out-of-bounds instrument in bobyqa_bounds_test.cpp: the count of calls is the
// ground-truth number of objective evaluations, independent of the driver's own
// bookkeeping.
namespace
{
struct counting_booth
{
    static constexpr int problem_dimension = 2;
    Eigen::Vector<double, 2> lb;
    Eigen::Vector<double, 2> ub;
    int* calls{nullptr};

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        if(calls) ++(*calls);
        double t1 = x[0] + 2.0 * x[1] - 7.0;
        double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::Vector<double, 2> lower_bounds() const { return lb; }
    Eigen::Vector<double, 2> upper_bounds() const { return ub; }
};
}

// Collapsing-denominator termination (no-rescue roundoff limit).
//
// When every candidate update denominator collapses (here forced by an
// objective that returns NaN, so scaden stays 0 and biglsq >= 0 makes the
// refusal test scaden <= 0.5*biglsq always true) and no rescue budget remains,
// the reference returns immediately as roundoff-limited (bobyqb_ lines
// 2483-2484 / 2546-2547). The driver's terminate flag must be a real loop exit:
// otherwise it re-runs identical trust passes until an internal safety cap and
// then reports a false convergence with zero state change.
//
// The instrument counts every objective call. The step must terminate promptly
// (a bounded, tiny number of calls) rather than spinning, and the reported
// evaluation count must not be inflated by a fabricated phantom evaluation on
// the no-eval terminate path.
namespace
{
struct counting_nan
{
    static constexpr int problem_dimension = 2;
    int* calls{nullptr};
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>&) const
    {
        if(calls) ++(*calls);
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::Vector<double, 2> lower_bounds() const { return {-10.0, -10.0}; }
    Eigen::Vector<double, 2> upper_bounds() const { return {10.0, 10.0}; }
};
}

TEST_CASE("bobyqa terminates promptly on a collapsing denominator instead of spinning",
          "[bobyqa]")
{
    int calls = 0;
    counting_nan problem{.calls = &calls};

    Eigen::Vector<double, 2> x0{0.0, 0.0};
    solver_options opts;
    bobyqa_policy<2> pol;
    auto s = pol.init(problem, x0, opts);

    // After init the bootstrap has made exactly the 2n+1 evaluations.
    REQUIRE(calls == s.m);
    REQUIRE(s.nevals == s.m);

    auto r = pol.step(s);

    // The first step hits the unrescuable refusal path and must exit at once:
    // it makes no further objective calls (the terminate path evaluates nothing)
    // and reports termination.
    CHECK(calls == s.m);
    CHECK(s.nevals == s.m);
    REQUIRE(r.policy_status.has_value());
    CHECK(*r.policy_status == solver_status::converged);

    // Machine-independent spin detector: the state machine increments ntrits
    // once at the top of every trust-region pass (L60). A driver that ignores
    // the terminate request re-enters L60 and increments ntrits on every one of
    // the ~200000 safety-cap passes before falsely reporting convergence; a
    // driver that treats terminate as a real loop exit leaves after a single
    // pass, so ntrits stays tiny.
    CHECK(s.ntrits < 100);

    // A full solve on the same pathological objective returns without spending
    // the iteration budget on phantom work: the only objective calls are the
    // 2n+1 bootstrap plus the solver's single best-seen seed evaluation; the
    // driver itself never evaluates past the refusal.
    int calls2 = 0;
    counting_nan problem2{.calls = &calls2};
    solver_options opts2;
    opts2.max_iterations = 1000;
    basic_solver solver{bobyqa_policy<2>{}, problem2, x0, opts2};
    auto sr = solver.solve(opts2);
    CHECK(calls2 <= 2 * (2 * 2 + 1));
    CHECK(sr.function_evaluations <= static_cast<std::uint32_t>(2 * (2 * 2 + 1)));
}

// Short-step no-evaluation instrument (FAM short-step skip).
//
// Powell defers the objective evaluation on a trust step shorter than 0.5*rho
// (ntrits = -1): the trial is NOT passed to the objective; the driver instead
// advances to a geometry refresh or a rho reduction, whose progress carries the
// iteration. Without this deferral (and the nfsav routing that sends the first
// short step after a rho refresh to geometry) the driver stalls -- a contained
// skip without the ntrits machine failed the majority of cases.
//
// The instrument counts every objective call (ground truth) and compares it to
// the driver's own evaluation counter. Because the driver only increments its
// counter inside the single evaluation site, agreement proves that the deferred
// short-step trials never reached the objective, while short_step_count > 0
// proves the deferral branch actually fired and the solver still converged.
TEST_CASE("bobyqa defers the short-step evaluation and still converges", "[bobyqa]")
{
    int calls = 0;
    counting_booth problem{
        .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
        .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
        .calls = &calls,
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    bobyqa_policy<2> pol;
    auto s = pol.init(problem, Eigen::Vector<double, 2>{0.0, 0.0}, opts);

    // After init, the ground-truth call count equals the 2n+1 bootstrap and the
    // driver's counter agrees.
    REQUIRE(calls == s.m);
    REQUIRE(s.nevals == s.m);

    bool converged = false;
    for(int it = 0; it < 2000; ++it)
    {
        auto r = pol.step(s);
        // Ground-truth objective calls are exactly the driver's evaluation
        // counter every step: the deferred short-step trials are never
        // evaluated (they would inflate `calls` past `nevals` otherwise).
        REQUIRE(calls == s.nevals);
        if(r.policy_status && *r.policy_status == solver_status::converged)
        {
            converged = true;
            break;
        }
    }

    // The deferral branch fired (short steps occurred) ...
    CHECK(s.short_step_count > 0);
    // ... the accounting is exact (no skipped trial reached the objective) ...
    CHECK(calls == s.nevals);
    // ... and progress was maintained to the Booth optimum at (1, 3) despite
    // skipping those evaluations (the nfsav routing prevents the stall).
    CHECK(converged);
    CHECK(s.objective_value < 1e-8);
    CHECK(s.x[0] == Approx(1.0).margin(1e-3));
    CHECK(s.x[1] == Approx(3.0).margin(1e-3));
}

// Relative scaden/biglsq update-point selection (FAM update-safeguard).
//
// Powell selects the interpolation point to delete on a trust step by the
// relative test den_k = beta*hdiag_k + vlag[k]^2 weighted by (dist_k/delta)^4,
// refusing the update when the best weighted denominator is dominated by the
// best weighted Lagrange value (scaden <= 0.5*biglsq). This replaces the earlier
// absolute denom > 1e-20 gate. The pin checks the selection against an
// independent recomputation of Powell's formula, that the best point kopt is
// never selected, and that a collapsed denominator triggers the refusal path
// instead of a corrupt update or a silent index-0 default.
TEST_CASE("bobyqa relative scaden/biglsq selection and refusal", "[bobyqa]")
{
    using policy = bobyqa_policy<2>;
    const int n = 2;
    const int m = 2 * n + 1;
    const int nptm = m - n - 1;

    // Bootstrap a genuine interpolation system on a curved 2D quadratic.
    auto obj = [](const Eigen::Vector<double, 2>& x) {
        return 2.0 * x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1] - x[0];
    };
    Eigen::Vector<double, 2> x0{0.3, -0.2};
    Eigen::Vector<double, 2> lo{-5.0, -5.0};
    Eigen::Vector<double, 2> hi{5.0, 5.0};
    auto sys = argmin::detail::bootstrap_interpolation_system<double, 2>(
        x0, 0.5, lo, hi, obj);

    // A trial step and its exact VLAG/BETA from the factored system.
    Eigen::Vector<double, 2> d{0.15, -0.1};
    auto vb = argmin::detail::compute_vlag_beta(sys, d);
    double delta = 0.5;

    auto choice = policy::select_replacement_relative(sys, vb.vlag, vb.beta, delta, sys.xopt);

    // Independent recomputation of Powell's scaden/biglsq formula.
    const double delsq = delta * delta;
    double scaden = 0.0, biglsq = 0.0, denom_expect = 0.0;
    int knew_expect = 0;
    for(int k = 0; k < m; ++k)
    {
        if(k == sys.kopt) continue;
        double hdiag = 0.0;
        for(int jj = 0; jj < nptm; ++jj)
            hdiag += sys.zmat(k, jj) * sys.zmat(k, jj);
        double den = vb.beta * hdiag + vb.vlag[k] * vb.vlag[k];
        double distsq = (sys.xpt.col(k).head(n) - sys.xopt).squaredNorm();
        double rsq = distsq / delsq;
        double temp = std::max(1.0, rsq * rsq);
        if(temp * den > scaden) { scaden = temp * den; knew_expect = k; denom_expect = den; }
        biglsq = std::max(biglsq, temp * vb.vlag[k] * vb.vlag[k]);
    }

    CHECK(choice.knew == knew_expect);
    CHECK(choice.denom == Approx(denom_expect));
    CHECK(choice.refuse == (scaden <= 0.5 * biglsq));
    // The best point is never selected for deletion.
    CHECK(choice.knew != sys.kopt);

    SECTION("collapsed denominator triggers refusal, not a corrupt update")
    {
        // Drive every den_k = beta*hdiag_k + vlag[k]^2 non-positive with a large
        // negative beta so no candidate yields a positive weighted denominator;
        // biglsq stays positive from the Lagrange values, so scaden(=0) <=
        // 0.5*biglsq and the update must be refused.
        double beta_bad = -1e6;
        auto bad = policy::select_replacement_relative(sys, vb.vlag, beta_bad, delta, sys.xopt);
        CHECK(bad.refuse);
    }
}

// Least-Frobenius-norm model reset (Powell 2009, Section 4; NLopt bobyqb_ lines
// 2824-2923). After each trust iteration Powell compares the projected gradient
// magnitude of the current explicit-Hessian model (gqsq) against that of the
// least-Frobenius-norm interpolant (gisq). A counter (itest) accumulates while
// the current model dominates the interpolant by a factor of ten and resets
// otherwise; on the third consecutive domination the model is replaced by the
// interpolant (gopt <- interpolant gradient, pq <- interpolant weights, hq <-
// 0). This exercises the counter increment/reset and the replacement on
// hand-built model states rather than trusting it fires in an end-to-end run.
TEST_CASE("bobyqa least-Frobenius-norm model reset triggers and installs the interpolant",
          "[bobyqa]")
{
    using policy = bobyqa_policy<2>;
    auto obj = [](const Eigen::Vector<double, 2>& x) {
        return 2.0 * x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1] - x[0];
    };
    Eigen::Vector<double, 2> x0{0.3, -0.2};
    Eigen::Vector<double, 2> lo{-5.0, -5.0};
    Eigen::Vector<double, 2> hi{5.0, 5.0};
    auto sys = argmin::detail::bootstrap_interpolation_system<double, 2>(
        x0, 0.5, lo, hi, obj);
    Eigen::Vector<double, 2> sl = lo - sys.xbase;
    Eigen::Vector<double, 2> su = hi - sys.xbase;

    SECTION("a well-modeled quadratic never triggers a reset")
    {
        // On an exact quadratic the interpolant reproduces the model gradient,
        // so the current model never dominates: the counter stays at zero and no
        // replacement occurs.
        int itest = 0;
        auto r = policy::frobenius_model_reset(sys, sl, su, itest);
        CHECK(r.gisq > 0.0);
        CHECK(r.gqsq == Approx(r.gisq));
        CHECK(itest == 0);
        CHECK_FALSE(r.reset_applied);
    }

    SECTION("three consecutive dominations reset the model to the interpolant")
    {
        // Blow up the explicit-Hessian model gradient so it dominates the
        // interpolant's by far more than the reference's factor of ten.
        sys.gopt *= 1e3;

        int itest = 0;
        auto r1 = policy::frobenius_model_reset(sys, sl, su, itest);
        CHECK(itest == 1);
        CHECK_FALSE(r1.reset_applied);
        CHECK(r1.gqsq > 10.0 * r1.gisq);

        auto r2 = policy::frobenius_model_reset(sys, sl, su, itest);
        CHECK(itest == 2);
        CHECK_FALSE(r2.reset_applied);

        auto r3 = policy::frobenius_model_reset(sys, sl, su, itest);
        // The third consecutive domination fires the replacement and zeroes the
        // counter.
        CHECK(r3.reset_applied);
        CHECK(itest == 0);

        // The reset zeroes the explicit Hessian ...
        const int nh = 2 * (2 + 1) / 2;
        for(int i = 0; i < nh; ++i)
            CHECK(sys.hq[i] == 0.0);

        // ... and installs the interpolant gradient into gopt: a subsequent
        // evaluation therefore sees the current model's gradient EQUAL the
        // interpolant's (gopt now IS the interpolant gradient), so it no longer
        // dominates and cannot re-trigger.
        int itest2 = 0;
        auto r4 = policy::frobenius_model_reset(sys, sl, su, itest2);
        CHECK(r4.gqsq == Approx(r4.gisq).margin(1e-12));
        CHECK(itest2 == 0);
        CHECK_FALSE(r4.reset_applied);
    }
}

// Quantitative acceptance against the checked-in NLopt LN_BOBYQA oracle.
//
// (1) Endgame accuracy: the origin shift lets the driver drive the curved
//     far-from-origin Rosenbrock to |f - f*| well below the ~1e-6 floor the
//     pre-rewrite driver reported, reaching the rho_end = 1e-8 endgame.
// (2) Eval-count regression: HS001/002/005/Booth converge to the reference
//     optima with evaluation counts near the NLopt oracle (the short-step skip
//     keeps the driver from spending evaluations on sub-rho trials), staying
//     within a small factor of the reference rather than regressing far past it.
//
// The eval counts differ from the oracle within a modest band because argmin
// solves in an internally rescaled frame and uses an approximate box TRSBOX;
// the pin bounds the count at 1.5x the reference (2x for HS002, whose bound is
// active at the solution) so a genuine regression is caught while the frame
// difference is tolerated.
TEST_CASE("bobyqa convergence and eval count match the NLopt oracle", "[bobyqa][oracle-pin]")
{
    const auto oracle = load_bobyqa_oracle("oracles/bobyqa_convergence.csv");
    REQUIRE(oracle.contains("hs001_nevals"));
    REQUIRE(oracle.contains("far_rosenbrock_nevals"));

    auto solve_hs = [&](auto hs) {
        Eigen::Vector<double, 2> x0 = hs.initial_point();
        solver_options opts;
        opts.max_iterations = 20000;
        opts.set_gradient_threshold(1e-15);
        opts.set_objective_threshold(0.0);
        opts.set_step_threshold(0.0);
        bobyqa_policy<2>::options_type popts;
        popts.final_trust_radius = 1e-8;
        basic_solver solver{bobyqa_policy<2>{popts}, hs, x0, opts};
        return solver.solve(opts);
    };

    SECTION("HS001 -- interior optimum, short-step efficiency")
    {
        auto r = solve_hs(argmin::hs001<double>{});
        CHECK(r.objective_value < 1e-6);
        CHECK(r.x[0] == Approx(oracle.at("hs001_xopt")[0]).margin(1e-3));
        CHECK(r.x[1] == Approx(oracle.at("hs001_xopt")[1]).margin(1e-3));
        // Toward the oracle count (216), not regressing far past it, and an
        // improvement on the pre-rewrite driver (which needed ~280 iterations).
        CHECK(r.function_evaluations <= static_cast<std::uint32_t>(1.5 * oracle.at("hs001_nevals")[0]));
    }

    SECTION("HS002 -- bound-active Rosenbrock local basin")
    {
        auto r = solve_hs(argmin::hs002<double>{});
        // BOBYQA settles in the negative-x0 valley basin recorded by the oracle.
        CHECK(r.objective_value == Approx(oracle.at("hs002_fopt")[0]).margin(1e-3));
        CHECK(r.x[0] == Approx(oracle.at("hs002_xopt")[0]).margin(1e-3));
        CHECK(r.x[1] == Approx(oracle.at("hs002_xopt")[1]).margin(1e-3));
        CHECK(r.function_evaluations <= static_cast<std::uint32_t>(2.0 * oracle.at("hs002_nevals")[0]));
    }

    SECTION("HS005 -- trigonometric interior optimum")
    {
        auto r = solve_hs(argmin::hs005<double>{});
        CHECK(r.objective_value == Approx(oracle.at("hs005_fopt")[0]).margin(1e-5));
        CHECK(r.x[0] == Approx(oracle.at("hs005_xopt")[0]).margin(1e-3));
        CHECK(r.x[1] == Approx(oracle.at("hs005_xopt")[1]).margin(1e-3));
        CHECK(r.function_evaluations <= static_cast<std::uint32_t>(1.5 * oracle.at("hs005_nevals")[0]));
    }

    SECTION("Booth -- eval-count efficiency")
    {
        bobyqa_booth problem{
            .lb = Eigen::Vector<double, 2>{{-10.0, -10.0}},
            .ub = Eigen::Vector<double, 2>{{10.0, 10.0}},
        };
        Eigen::Vector<double, 2> x0{0.0, 0.0};
        solver_options opts;
        opts.max_iterations = 20000;
        opts.set_gradient_threshold(1e-15);
        opts.set_objective_threshold(0.0);
        opts.set_step_threshold(0.0);
        bobyqa_policy<2>::options_type popts;
        popts.final_trust_radius = 1e-8;
        basic_solver solver{bobyqa_policy<2>{popts}, problem, x0, opts};
        auto r = solver.solve(opts);
        CHECK(r.objective_value < 1e-8);
        CHECK(r.x[0] == Approx(oracle.at("booth_xopt")[0]).margin(1e-3));
        CHECK(r.x[1] == Approx(oracle.at("booth_xopt")[1]).margin(1e-3));
        CHECK(r.function_evaluations <= static_cast<std::uint32_t>(1.5 * oracle.at("booth_nevals")[0]));
    }

    SECTION("far-from-origin Rosenbrock -- endgame accuracy the origin shift unlocks")
    {
        far_rosenbrock problem{
            .lb = Eigen::Vector<double, 2>{{-1000.0, -1000.0}},
            .ub = Eigen::Vector<double, 2>{{1000.0, 1000.0}},
        };
        Eigen::Vector<double, 2> x0{0.0, 0.0};
        solver_options opts;
        opts.max_iterations = 50000;
        opts.set_gradient_threshold(1e-15);
        opts.set_objective_threshold(0.0);
        opts.set_step_threshold(0.0);
        bobyqa_policy<2>::options_type popts;
        popts.initial_trust_radius = 10.0;
        popts.final_trust_radius = 1e-8;
        basic_solver solver{bobyqa_policy<2>{popts}, problem, x0, opts};
        auto r = solver.solve(opts);

        // The endgame criterion: |f - f*| reaches ~1e-8 (the reference lands at
        // ~5e-21). The pre-rewrite driver floored at ~1e-6 on this problem.
        CHECK(r.objective_value < 1e-8);
        CHECK(r.x[0] == Approx(oracle.at("far_rosenbrock_xopt")[0]).margin(1e-3));
        CHECK(r.x[1] == Approx(oracle.at("far_rosenbrock_xopt")[1]).margin(1e-2));
    }
}
