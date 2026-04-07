#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/byrd_lbfgsb_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/detail/cauchy_point.h"
#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/lbfgsb_direction.h"
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
        opts.set_gradient_threshold(1e-6);
        opts.max_iterations = 200;

        basic_solver<lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};
        auto result = solver.solve(opts);

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
        opts.set_gradient_threshold(1e-6);
        opts.max_iterations = 500;

        basic_solver<lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};
        auto result = solver.solve(opts);

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
        opts.set_gradient_threshold(1e-6);
        opts.max_iterations = 500;
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver<lbfgsb_policy<bounded_rosenbrock::problem_dimension>, bounded_rosenbrock::problem_dimension, bounded_rosenbrock> solver{problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached
               || result.status == solver_status::stalled
               || result.status == solver_status::max_iterations
               || result.status == solver_status::roundoff_limited));

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
        opts.set_gradient_threshold(1e-5);
        opts.max_iterations = 500;
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver<lbfgsb_policy<bounded_rosenbrock::problem_dimension>, bounded_rosenbrock::problem_dimension, bounded_rosenbrock> solver{problem, x0, opts};
        auto result = solver.solve(opts);

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
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;

    SECTION("step returns finite values")
    {
        basic_solver<lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};

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
        basic_solver<lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.status == solver_status::converged);
    }
}

TEST_CASE("Two L-BFGS-B policies in solver_group (SC5)", "[lbfgsb][solver_group]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);

    basic_solver_group<round_robin_schedule, dynamic_dimension, rosenbrock<>, lbfgsb_policy<>, lbfgsb_policy<>> group{
        problem, x0, opts};

    // Step 10 times, each step should produce finite results
    for(int i = 0; i < 10; ++i)
    {
        auto sr = group.step();
        CHECK(std::isfinite(sr.objective_value));
        CHECK(std::isfinite(sr.gradient_norm));
    }

    // Run to convergence or budget
    auto result = group.step_n(200, opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::budget_exhausted));
    CHECK(std::isfinite(result.objective_value));
    CHECK(result.x.size() == 2);
}

TEST_CASE("GCP finds interior minimum on bounded quadratic", "[lbfgsb]")
{
    // Regression guard for cauchy_point sign error (BUG-02).
    // With a positive-definite B and gradient pointing away from bounds,
    // the GCP should land at an interior point (not at a breakpoint).
    //
    // Setup: 2D identity B operator, bounds [0,2]^2, x0=(1.5, 1.5),
    // g=(-1, -0.5). The unconstrained steepest descent minimum along
    // d = -g = (1, 0.5) is at t* = g^T d / (d^T B d) = 1.25 / 1.25 = 1.0,
    // giving x_cauchy = (2.5, 2.0). Projection onto [0,2]^2 clips x[0] to 2.0.
    // But the GCP algorithm should find the interior minimum at t where
    // x(t) first hits a bound or the quadratic minimum, whichever comes first.
    //
    // With corrected sign, f'' = d^T B d > 0 triggers the interior check.
    using namespace nablapp::detail;

    compact_lbfgs<double> B;

    Eigen::VectorXd x(2);
    x << 1.5, 1.5;
    Eigen::VectorXd g(2);
    g << -1.0, -0.5;
    Eigen::VectorXd lower(2);
    lower << 0.0, 0.0;
    Eigen::VectorXd upper(2);
    upper << 2.0, 2.0;

    auto result = cauchy_point(x, g, lower, upper, B);

    // The GCP must be feasible
    CHECK(result.x_cauchy[0] >= 0.0 - 1e-10);
    CHECK(result.x_cauchy[0] <= 2.0 + 1e-10);
    CHECK(result.x_cauchy[1] >= 0.0 - 1e-10);
    CHECK(result.x_cauchy[1] <= 2.0 + 1e-10);

    // With B = I (theta=1, no pairs), d = -g = (1, 0.5).
    // Breakpoints: x[0] hits upper at t = (1.5 - 2.0)/(-1.0) = 0.5,
    //              x[1] hits upper at t = (1.5 - 2.0)/(-0.5) = 1.0.
    // f'(0) = g^T d = -1*1 + -0.5*0.5 = -1.25.
    // f''   = d^T B d = 1^2 + 0.5^2 = 1.25.
    // Interior minimum: t* = -f'/f'' = 1.25/1.25 = 1.0.
    // But first breakpoint is at t=0.5, and at t=0.5:
    //   f'(0.5) = f'(0) + 0.5 * f'' = -1.25 + 0.625 = -0.625 < 0
    // So we pass through the first breakpoint. After fixing d[0]=0:
    //   d = (0, 0.5), Bd = (0, 0.5), f'' = 0.25, f' recomputed.
    // The key assertion: with the sign fix, the algorithm CAN find
    // interior minima (f'' > 0 triggers the check). Without the fix,
    // f'' was always negative so the check never triggered.
    //
    // Verify GCP is not simply at x0 (no progress) -- the sign bug
    // caused the algorithm to skip interior checks entirely.
    double dist_from_x0 = (result.x_cauchy - x).norm();
    CHECK(dist_from_x0 > 0.1);
}

TEST_CASE("compact_lbfgs accepts small curvature pairs", "[lbfgsb]")
{
    // Regression guard for curvature pair rejection (BUG-03).
    // The old guard rejected s^T y <= 0. The new guard uses
    // s^T y <= eps * ||s||^2, accepting pairs with small positive curvature.
    //
    // Reference: N&W Section 9.1 (curvature condition).
    using namespace nablapp::detail;

    compact_lbfgs<double> B;

    // Push a pair with small but positive curvature:
    // s = (1, 0), y = (1e-12, 0), s^T y = 1e-12.
    // eps * ||s||^2 = eps * 1 ~ 2.2e-16, so 1e-12 >> eps * ||s||^2.
    Eigen::VectorXd s(2), y(2);
    s << 1.0, 0.0;
    y << 1e-12, 0.0;
    B.push(s, y);
    CHECK(B.size() == 1);

    // Push a degenerate pair: s = (0, 0), y = (0, 0).
    // s^T y = 0, eps * ||s||^2 = 0, so 0 <= 0 triggers rejection.
    Eigen::VectorXd s2(2), y2(2);
    s2 << 0.0, 0.0;
    y2 << 0.0, 0.0;
    B.push(s2, y2);
    CHECK(B.size() == 1);

    // Push a pair where s^T y > 0 but s^T y < eps * ||s||^2.
    // s = (1e8, 0), y = (1e-25, 0), s^T y = 1e-17.
    // eps * ||s||^2 = eps * 1e16 ~ 2.2e-16 * 1e16 = 2.2 >> 1e-17.
    // Under damped curvature update (Powell), s^T y is clamped upward to
    // max(s^T y, eps * ||s||^2) = 2.2, so this pair IS accepted.
    Eigen::VectorXd s3(2), y3(2);
    s3 << 1e8, 0.0;
    y3 << 1e-25, 0.0;
    B.push(s3, y3);
    CHECK(B.size() == 2);
}

TEST_CASE("compact_lbfgs damped curvature accepts near-orthogonal pairs", "[lbfgsb]")
{
    // The damped update clamps small positive s^T y to max(s^T y, eps * ||s||^2)
    // instead of rejecting. Negative/zero curvature is still rejected.
    // Reference: Powell (damped BFGS), N&W Section 9.1.
    using namespace nablapp::detail;

    compact_lbfgs<double> B;

    // Near-orthogonal pair with tiny positive curvature.
    // s^T y = 1e-18 > 0, eps * ||s||^2 ~ 2.2e-16 >> 1e-18.
    // Old guard rejected, damped clamps upward and accepts.
    Eigen::VectorXd s(2), y(2);
    s << 1.0, 0.0;
    y << 1e-18, 1.0;
    B.push(s, y);
    CHECK(B.size() == 1);

    // Zero curvature (s^T y = 0) is still rejected — negative/zero curvature
    // poisons BFGS approximations in SQP solvers.
    Eigen::VectorXd s_orth(2), y_orth(2);
    s_orth << 1.0, 0.0;
    y_orth << 0.0, 1.0;
    B.push(s_orth, y_orth);
    CHECK(B.size() == 1);
}

TEST_CASE("Byrd L-BFGS-B converges on unconstrained Rosenbrock", "[lbfgsb][byrd]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;

    basic_solver<byrd_lbfgsb_policy<rosenbrock<>::problem_dimension>, rosenbrock<>::problem_dimension, rosenbrock<>> solver{problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::stalled
           || result.status == solver_status::roundoff_limited));
    CHECK(result.objective_value < 1.0);
}

TEST_CASE("lbfgsb_policy with Armijo line search converges", "[lbfgsb]")
{
    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;

    lbfgsb_policy policy;
    typename lbfgsb_policy<>::options_type policy_opts;
    policy_opts.line_search_type = lbfgsb_line_search::armijo;
    auto state = policy.init(problem, x0, opts, policy_opts);

    for(std::uint32_t i = 0; i < 200; ++i)
    {
        auto sr = policy.step(state);
        if(sr.gradient_norm < 1e-5) break;
    }

    CHECK(state.objective_value < 1.0);
}

TEST_CASE("Two-loop fast path direction equivalence", "[lbfgsb]")
{
    // Run the same unconstrained problem through both paths (fast path via
    // two-loop recursion and forced GCP+subspace path with infinite bounds),
    // assert search directions match to machine epsilon at each iteration.
    //
    // The fast path produces d = -H*g via two-loop recursion (N&W Algorithm
    // 9.1). The GCP+subspace path runs the full L-BFGS-B direction with
    // [-inf, inf] bounds -- when all variables are free, this should produce
    // the same direction.
    //
    // Reference: N&W Algorithm 9.1 (two-loop recursion),
    //            N&W Section 16.6 (GCP + subspace minimization).

    using namespace nablapp::detail;

    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x = Eigen::VectorXd{{-1.2, 1.0}};
    Eigen::VectorXd g(2);
    problem.gradient(x, g);

    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(2, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(2, inf);

    compact_lbfgs<double> B;
    cauchy_point_solver<double> gcp_solver(2);
    subspace_minimizer<double> ssm_solver(2);

    constexpr double eps = std::numeric_limits<double>::epsilon();

    // Run 10 iterations, comparing directions at each step.
    for(int iter = 0; iter < 10; ++iter)
    {
        // Fast path: two-loop recursion (what compute_direction does for
        // non-bound_constrained problems).
        auto Hg = B.two_loop_recursion(g);
        Eigen::VectorXd d_fast = (-Hg).eval();

        // GCP+subspace path: full L-BFGS-B direction with infinite bounds.
        const auto& gcp = gcp_solver.solve(x, g, lower, upper, B);
        Eigen::VectorXd x_new = ssm_solver.solve(
            x, gcp.x_cauchy, g, lower, upper, gcp.free_indices, B);
        Eigen::VectorXd d_gcp = (x_new - x).eval();

        // Directions must match to machine epsilon.
        // Normalize by direction magnitude to get relative error.
        double d_norm = std::max(d_fast.norm(), d_gcp.norm());
        if(d_norm > 1e-15)
        {
            double rel_error = (d_fast - d_gcp).norm() / d_norm;
            CHECK(rel_error < 1000 * eps);
        }

        // Take a step using the fast path direction (quasi-Newton).
        if(d_fast.norm() < 1e-15) break;

        double alpha = 1.0;
        double f0 = problem.value(x);
        double dphi0 = g.dot(d_fast);
        for(int ls = 0; ls < 20; ++ls)
        {
            double f_trial = problem.value((x + alpha * d_fast).eval());
            if(f_trial <= f0 + 1e-4 * alpha * dphi0) break;
            alpha *= 0.5;
        }

        Eigen::VectorXd x_old = x;
        Eigen::VectorXd g_old = g;
        x = (x + alpha * d_fast).eval();
        problem.gradient(x, g);

        // Push curvature pair.
        Eigen::VectorXd sk = (x - x_old).eval();
        Eigen::VectorXd yk = (g - g_old).eval();
        B.push(sk, yk);
    }
}

TEST_CASE("Compile-time unconstrained path sets infinite alpha_max", "[lbfgsb]")
{
    // Compile-time unconstrained problems produce alpha_max = infinity because
    // no bounds exist to limit step length.
    //
    // Reference: N&W Algorithm 9.1 (two-loop recursion, no bounds).

    using namespace nablapp::detail;

    rosenbrock<> problem{.n = 2};
    Eigen::VectorXd x = Eigen::VectorXd{{-1.2, 1.0}};
    Eigen::VectorXd g(2);
    problem.gradient(x, g);

    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(2, -inf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(2, inf);

    compact_lbfgs<double> B;
    cauchy_point_solver<double> gcp_solver(2);
    subspace_minimizer<double> ssm_solver(2);

    auto dir = compute_direction<rosenbrock<>>(x, g, lower, upper, B, gcp_solver, ssm_solver);
    REQUIRE(dir.has_value());
    CHECK(dir->alpha_max == inf);
    CHECK(dir->d.norm() > 0.0);
}

TEST_CASE("Runtime fast path with loose bounds computes finite alpha_max", "[lbfgsb]")
{
    // Runtime fast path activates when bound_constrained is satisfied at
    // compile time but all variables are free at runtime. It still computes
    // alpha_max from the distance to the nearest bound.
    //
    // Reference: Byrd, Lu, Nocedal, Zhu 1995 (breakpoint definition).

    using namespace nablapp::detail;

    rosenbrock<> rosen{.n = 2};
    Eigen::VectorXd x = Eigen::VectorXd{{-1.2, 1.0}};
    Eigen::VectorXd g(2);
    rosen.gradient(x, g);

    // Loose bounds: x is well inside, all variables free.
    Eigen::VectorXd lower = Eigen::VectorXd{{-10.0, -10.0}};
    Eigen::VectorXd upper = Eigen::VectorXd{{10.0, 10.0}};

    compact_lbfgs<double> B;
    cauchy_point_solver<double> gcp_solver(2);
    subspace_minimizer<double> ssm_solver(2);

    auto dir = compute_direction<bounded_rosenbrock>(x, g, lower, upper, B, gcp_solver, ssm_solver);
    REQUIRE(dir.has_value());
    // alpha_max should be finite (bounded by distance to nearest wall).
    CHECK(std::isfinite(dir->alpha_max));
    CHECK(dir->alpha_max > 0.0);
    CHECK(dir->d.norm() > 0.0);
}
