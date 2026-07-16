#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numbers>
#include <algorithm>

using Catch::Approx;
using namespace argmin;

namespace
{

// Bounded Rosenbrock in least-squares form (b=5 per project convention).
// r_0 = 1 - x_0, r_1 = sqrt(5)*(x_1 - x_0^2)
// Bounds: x_0 in [-2, 0.5], x_1 in [-2, 2].
// Unconstrained optimum (1,1) is infeasible (x_0 > 0.5).
// Constrained optimum: x_0 = 0.5, x_1 = 0.25. f* = 0.125.
struct bounded_rosenbrock_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

// Bounded exponential fitting: y = a * exp(b * t).
// Bounds enforce physical constraints: a > 0, b <= 0.
struct bounded_exponential_fitting
{
    static constexpr double t[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    static constexpr double y[] = {2.0, 1.2, 0.75, 0.45, 0.28};

    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 5; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < 5; ++i)
        {
            double ri = x(0) * std::exp(x(1) * t[i]) - y[i];
            f += ri * ri;
        }
        return 0.5 * f;
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        for(int i = 0; i < 5; ++i)
            r(i) = x(0) * std::exp(x(1) * t[i]) - y[i];
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        for(int i = 0; i < 5; ++i)
        {
            double e = std::exp(x(1) * t[i]);
            J(i, 0) = e;
            J(i, 1) = x(0) * t[i] * e;
        }
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{0.1, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{5.0, 0.0}}; }
};

// Bounded Powell singular in least-squares form (n=4, m=4).
// r_0 = x_0 + 10*x_1
// r_1 = sqrt(5)*(x_2 - x_3)
// r_2 = (x_1 - 2*x_2)^2
// r_3 = sqrt(10)*(x_0 - x_3)^2
// Bounds: x in [-5,-5,-5,-5] to [2,2,5,5].
// Optimum at origin (interior of bounds).
struct bounded_powell_singular_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 4; }
    int num_residuals() const { return 4; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = x(0) + 10.0 * x(1);
        double r1 = std::sqrt(5.0) * (x(2) - x(3));
        double r2 = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
        double r3 = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
        return 0.5 * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = x(0) + 10.0 * x(1);
        r(1) = std::sqrt(5.0) * (x(2) - x(3));
        r(2) = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
        r(3) = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.setZero();
        J(0, 0) = 1.0;
        J(0, 1) = 10.0;
        J(1, 2) = std::sqrt(5.0);
        J(1, 3) = -std::sqrt(5.0);
        J(2, 1) = 2.0 * (x(1) - 2.0 * x(2));
        J(2, 2) = -4.0 * (x(1) - 2.0 * x(2));
        J(3, 0) = 2.0 * std::sqrt(10.0) * (x(0) - x(3));
        J(3, 3) = -2.0 * std::sqrt(10.0) * (x(0) - x(3));
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-5.0, -5.0, -5.0, -5.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{2.0, 2.0, 5.0, 5.0}};
    }
};

// SE(3)-like 6D proxy problem for IK testing.
// Diagonal weighted residuals: r(i) = w(i) * (x(i) - x*(i)).
// Joint-limit-like box bounds: rotation in [-pi, pi], translation in [-1, 1].
// Optimum at x* (interior of bounds), f* = 0.
struct se3_proxy_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    static constexpr double x_star[] = {0.5, -0.3, 0.8, 0.1, -0.2, 0.15};
    static constexpr double w[] = {1.0, 1.0, 1.0, 0.1, 0.1, 0.1};

    int dimension() const { return 6; }
    int num_residuals() const { return 6; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < 6; ++i)
        {
            double ri = w[i] * (x(i) - x_star[i]);
            f += ri * ri;
        }
        return 0.5 * f;
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        for(int i = 0; i < 6; ++i)
            r(i) = w[i] * (x(i) - x_star[i]);
    }

    void jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.setZero();
        for(int i = 0; i < 6; ++i)
            J(i, i) = w[i];
    }

    Eigen::VectorXd lower_bounds() const
    {
        constexpr double pi = std::numbers::pi;
        return Eigen::VectorXd{{-pi, -pi, -pi, -1.0, -1.0, -1.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        constexpr double pi = std::numbers::pi;
        return Eigen::VectorXd{{pi, pi, pi, 1.0, 1.0, 1.0}};
    }
};

// Bounded Rosenbrock LS without Jacobian -- triggers FD fallback.
struct bounded_rosenbrock_ls_no_jac
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

}

// Concept satisfaction checks for test problem structs.
static_assert(objective<bounded_rosenbrock_ls>);
static_assert(least_squares<bounded_rosenbrock_ls>);
static_assert(bound_constrained<bounded_rosenbrock_ls>);

static_assert(objective<bounded_exponential_fitting>);
static_assert(least_squares<bounded_exponential_fitting>);
static_assert(bound_constrained<bounded_exponential_fitting>);

static_assert(objective<bounded_powell_singular_ls>);
static_assert(least_squares<bounded_powell_singular_ls>);
static_assert(bound_constrained<bounded_powell_singular_ls>);

static_assert(objective<se3_proxy_ls>);
static_assert(least_squares<se3_proxy_ls>);
static_assert(bound_constrained<se3_proxy_ls>);

static_assert(objective<bounded_rosenbrock_ls_no_jac>);
static_assert(!least_squares<bounded_rosenbrock_ls_no_jac>);
static_assert(bound_constrained<bounded_rosenbrock_ls_no_jac>);

// --------------------------------------------------------------------------
// projected_gn_policy tests
// --------------------------------------------------------------------------

TEST_CASE("projected_gn_policy: bounded Rosenbrock LS", "[projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.13);
    CHECK(result.x(0) == Approx(0.5).margin(1e-4));
    CHECK(result.x(1) == Approx(0.25).margin(1e-3));
}

TEST_CASE("projected_gn_policy: bounded exponential fitting", "[projected_gn]")
{
    bounded_exponential_fitting problem;
    Eigen::VectorXd x0{{1.0, -0.5}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.01);
}

TEST_CASE("projected_gn_policy: bounded Powell singular", "[projected_gn]")
{
    bounded_powell_singular_ls problem;
    Eigen::VectorXd x0{{1.0, 1.0, 1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-8);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 1e-6);
}

TEST_CASE("projected_gn_policy: SE(3)-like proxy", "[projected_gn]")
{
    se3_proxy_ls problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 1e-10);
    CHECK(result.x(0) == Approx(0.5).margin(1e-5));
    CHECK(result.x(1) == Approx(-0.3).margin(1e-5));
    CHECK(result.x(2) == Approx(0.8).margin(1e-5));
    CHECK(result.x(3) == Approx(0.1).margin(1e-5));
    CHECK(result.x(4) == Approx(-0.2).margin(1e-5));
    CHECK(result.x(5) == Approx(0.15).margin(1e-5));
}

TEST_CASE("projected_gn_policy: dogleg mode", "[projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    projected_gn_policy<>::options_type policy_opts{};
    policy_opts.use_dogleg = true;

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts, policy_opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.13);
}

// --------------------------------------------------------------------------
// projected_gradient_gn_policy tests
// --------------------------------------------------------------------------

TEST_CASE("projected_gradient_gn_policy: bounded Rosenbrock LS", "[projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.13);
    CHECK(result.x(0) == Approx(0.5).margin(1e-4));
    CHECK(result.x(1) == Approx(0.25).margin(1e-2));
}

TEST_CASE("projected_gradient_gn_policy: SE(3)-like proxy", "[projected_gn]")
{
    se3_proxy_ls problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);

    step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 1e-10);
    CHECK(result.x(0) == Approx(0.5).margin(1e-5));
    CHECK(result.x(1) == Approx(-0.3).margin(1e-5));
    CHECK(result.x(2) == Approx(0.8).margin(1e-5));
    CHECK(result.x(3) == Approx(0.1).margin(1e-5));
    CHECK(result.x(4) == Approx(-0.2).margin(1e-5));
    CHECK(result.x(5) == Approx(0.15).margin(1e-5));
}

TEST_CASE("projected_gradient_gn_policy: dogleg mode", "[projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    projected_gradient_gn_policy<>::options_type policy_opts{};
    policy_opts.use_dogleg = true;

    step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts, policy_opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.13);
}

TEST_CASE("projected_gn_policy: FD Jacobian fallback", "[projected_gn]")
{
    bounded_rosenbrock_ls_no_jac problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.2);
    CHECK(result.x(0) == Approx(0.5).margin(1e-2));
}

// projected_gn_policy populates step_result::kkt_residual via
// detail::kkt_residual_bound on every step. For a bound-constrained
// least-squares problem the value is the projected-gradient infinity-
// norm, which equals zero iff the bound-constrained first-order
// conditions hold.
//
// Reference: N&W 2e Section 16.7 (projected gradient optimality).
TEST_CASE("projected_gn_policy populates kkt_residual", "[projected_gn][kkt]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 10; ++i)
    {
        auto sr = solver.step();
        if(sr.kkt_residual.has_value())
        {
            populated = true;
            CHECK(sr.kkt_residual.value() >= 0.0);
        }
        if(sr.policy_status)
            break;
    }
    CHECK(populated);
}

// projected_gradient_gn_policy populates step_result::kkt_residual via
// detail::kkt_residual_bound on every step through both the backtracking
// and dogleg return paths.
//
// Reference: N&W 2e Section 16.7 (projected gradient optimality).
TEST_CASE("projected_gradient_gn_policy populates kkt_residual",
          "[projected_gn][kkt]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);

    SECTION("backtracking mode")
    {
        step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts};

        bool populated = false;
        for(int i = 0; i < 10; ++i)
        {
            auto sr = solver.step();
            if(sr.kkt_residual.has_value())
            {
                populated = true;
                CHECK(sr.kkt_residual.value() >= 0.0);
            }
            if(sr.policy_status)
                break;
        }
        CHECK(populated);
    }

    SECTION("dogleg mode")
    {
        projected_gradient_gn_policy<>::options_type policy_opts{};
        policy_opts.use_dogleg = true;

        step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts,
                            policy_opts};

        bool populated = false;
        for(int i = 0; i < 10; ++i)
        {
            auto sr = solver.step();
            if(sr.kkt_residual.has_value())
            {
                populated = true;
                CHECK(sr.kkt_residual.value() >= 0.0);
            }
            if(sr.policy_status)
                break;
        }
        CHECK(populated);
    }
}

TEST_CASE("projected_gn_policy: active set exercised", "[projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // x(0) should be at or very near the upper bound 0.5
    CHECK(result.x(0) >= 0.5 - 1e-6);
}

// --------------------------------------------------------------------------
// Gain-ratio witness: direct model reduction at an active bound.
// --------------------------------------------------------------------------

namespace
{

// Bounded linear least squares: r(x) = x - b, so f = 0.5||x - b||^2 is exactly
// quadratic and the Gauss-Newton model is exact. b = (2, 0.25) lies outside the
// box on the first coordinate, so the constrained optimum pins x0 at its upper
// bound 0.5 (active set) with x1 = 0.25 interior; f* = 0.5*(1.5^2) = 1.125.
struct bounded_linear_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    static Eigen::Vector2d bvec() { return Eigen::Vector2d{2.0, 0.25}; }

    double value(const Eigen::VectorXd& x) const
    {
        return 0.5 * (x - bvec()).squaredNorm();
    }
    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r = x - bvec();
    }
    void jacobian(const Eigen::VectorXd& /*x*/, Eigen::MatrixXd& J) const
    {
        J = Eigen::Matrix2d::Identity();
    }
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

}

// The direct model reduction -g^T d - 0.5||J d||^2 is exact for this quadratic,
// even for the projected step d that lands on the active bound, so the gain
// ratio is 1 and each accepted step's reported objective_change equals the true
// objective decrease (no lambda proxy). The old LM predicted-reduction identity
// on the projected step produced a fictitious denominator here.
TEST_CASE("projected_gn_policy: exact gain ratio at an active bound",
          "[projected_gn][gain_ratio]")
{
    bounded_linear_ls problem;
    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);

    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};

    double prev_f = solver.state().objective_value;
    for(int i = 0; i < 30; ++i)
    {
        auto sr = solver.step();
        if(sr.improved)
        {
            double true_change = sr.objective_value - prev_f;
            CHECK(sr.objective_change == Approx(true_change).margin(1e-14));
        }
        prev_f = sr.objective_value;
        if(sr.policy_status)
            break;
        if(sr.kkt_residual.value_or(sr.gradient_norm) < 1e-12)
            break;
    }

    // Constrained optimum: x0 pinned at the upper bound 0.5, x1 = 0.25.
    CHECK(solver.state().x(0) == Approx(0.5).margin(1e-8));
    CHECK(solver.state().x(1) == Approx(0.25).margin(1e-8));
    CHECK(solver.state().objective_value == Approx(1.125).margin(1e-8));
}

// Sibling parity: projected_gradient_gn_policy carries the identical
// defect and fix. Both its backtracking and dogleg paths now use the direct
// model reduction at the actual accepted displacement, so the gain ratio is
// exact on this quadratic and objective_change is honest.
TEST_CASE("projected_gradient_gn_policy: exact gain ratio at an active bound",
          "[projected_gn][gain_ratio]")
{
    bounded_linear_ls problem;
    Eigen::VectorXd x0{{-1.0, -1.0}};

    auto run = [&](bool dogleg)
    {
        solver_options opts;
        opts.max_iterations = 100;
        opts.set_gradient_threshold(1e-12);
        projected_gradient_gn_policy<>::options_type policy_opts{};
        policy_opts.use_dogleg = dogleg;

        step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, x0, opts, policy_opts};

        double prev_f = solver.state().objective_value;
        for(int i = 0; i < 40; ++i)
        {
            auto sr = solver.step();
            if(sr.improved)
            {
                double true_change = sr.objective_value - prev_f;
                CHECK(sr.objective_change == Approx(true_change).margin(1e-14));
            }
            prev_f = sr.objective_value;
            if(sr.policy_status)
                break;
            if(sr.kkt_residual.value_or(sr.gradient_norm) < 1e-12)
                break;
        }

        // Constrained optimum: x0 pinned at the upper bound 0.5, x1 = 0.25.
        CHECK(solver.state().x(0) == Approx(0.5).margin(1e-6));
        CHECK(solver.state().x(1) == Approx(0.25).margin(1e-6));
    };

    SECTION("backtracking mode") { run(false); }
    SECTION("dogleg mode") { run(true); }
}

// detail::gain_ratio identity pin for the two projected Gauss-Newton siblings.
// The fingerprints (final objective, minimizer, and a trajectory hash over
// every step's objective_change and step_size) were captured from the
// pre-extraction build; the extraction must not move any of them.
//
// The final-state and minimizer pins hold bit-exact. The trajectory hash `fp`
// accumulates ~100 steps; the gain_ratio extraction is algebraically identical
// to the former inline arithmetic and reproduces `fp` bit-exactly in Debug, but
// under Release + LTO the cross-function inlining lets the optimizer reassociate
// the accumulation, shifting the final sum by ~1 ULP (relative ~1e-16). The hash
// is therefore pinned to a tight relative bound that tolerates only that
// LTO-driven FP reassociation: a genuine trajectory change moves `fp` by a
// relative amount orders of magnitude larger and is still caught.
constexpr double trajectory_fp_rel_tol = 1e-14;

TEST_CASE("projected_gn_policy: gain_ratio extraction is numerically identical",
          "[projected_gn][gain_ratio]")
{
    bounded_linear_ls problem;
    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);
    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};

    double fp = 0.0;
    for(int i = 0; i < 100; ++i)
    {
        auto r = solver.step();
        fp += r.objective_change * 1e6 + r.step_size;
        if(r.policy_status || r.kkt_residual.value_or(r.gradient_norm) < 1e-12)
            break;
    }
    CHECK(solver.state().objective_value == 1.125);
    CHECK(solver.state().x(0) == 0.5);
    CHECK(solver.state().x(1) == 0.24999999995377051);
    CHECK(fp == Approx(-4156246.7519980022).epsilon(trajectory_fp_rel_tol));
}

TEST_CASE("projected_gradient_gn_policy: gain_ratio extraction is numerically identical",
          "[projected_gn][gain_ratio]")
{
    auto fingerprint = [](bool dogleg, double& f, double& x0, double& x1)
    {
        bounded_linear_ls problem;
        Eigen::VectorXd start{{-1.0, -1.0}};
        solver_options opts;
        opts.max_iterations = 100;
        opts.set_gradient_threshold(1e-12);
        projected_gradient_gn_policy<>::options_type po{};
        po.use_dogleg = dogleg;
        step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, start, opts, po};
        double fp = 0.0;
        for(int i = 0; i < 100; ++i)
        {
            auto r = solver.step();
            fp += r.objective_change * 1e6 + r.step_size;
            if(r.policy_status || r.kkt_residual.value_or(r.gradient_norm) < 1e-12)
                break;
        }
        f = solver.state().objective_value;
        x0 = solver.state().x(0);
        x1 = solver.state().x(1);
        return fp;
    };

    SECTION("backtracking mode")
    {
        double f, x0, x1;
        double fp = fingerprint(false, f, x0, x1);
        CHECK(f == 1.125);
        CHECK(x0 == 0.5);
        CHECK(x1 == 0.25);
        CHECK(fp == Approx(-4156240.7540793722).epsilon(trajectory_fp_rel_tol));
    }
    SECTION("dogleg mode")
    {
        double f, x0, x1;
        double fp = fingerprint(true, f, x0, x1);
        CHECK(f == 1.125);
        CHECK(x0 == 0.5);
        CHECK(x1 == 0.24999999999999786);
        CHECK(fp == Approx(-4156246.903846154).epsilon(trajectory_fp_rel_tol));
    }
}

// --------------------------------------------------------------------------
// Fixed-N identity pin: the compile-time-dimension path must reproduce the
// runtime-dimension trajectory bit-for-bit.
//
// The problem's problem_dimension drives the CTAD deduction guide, which
// rebinds the policy to projected_gn_policy<N>. bounded_powell_singular_ls
// declares dynamic_dimension (rebinds to the dynamic path); the fixed-4
// twin below declares problem_dimension = 4 (rebinds to
// projected_gn_policy<4>, decision variable + box on Vector<double, 4>).
// Both run the identical Powell-singular least-squares math, so the whole
// objective / step-size / iterate sequence must match exactly -- the
// de-type-erasure and axis typing introduce no numerical drift.
// --------------------------------------------------------------------------

namespace
{

// Fixed-dimension twin of bounded_powell_singular_ls: identical residuals and
// Jacobian, but the decision variable is a compile-time Vector<double, 4>.
struct bounded_powell_singular_ls_fixed4
{
    static constexpr int problem_dimension = 4;

    int dimension() const { return 4; }
    int num_residuals() const { return 4; }

    double value(const Eigen::Vector<double, 4>& x) const
    {
        double r0 = x(0) + 10.0 * x(1);
        double r1 = std::sqrt(5.0) * (x(2) - x(3));
        double r2 = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
        double r3 = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
        return 0.5 * (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
    }

    void residuals(const Eigen::Vector<double, 4>& x, Eigen::VectorXd& r) const
    {
        r(0) = x(0) + 10.0 * x(1);
        r(1) = std::sqrt(5.0) * (x(2) - x(3));
        r(2) = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
        r(3) = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
    }

    void jacobian(const Eigen::Vector<double, 4>& x, Eigen::MatrixXd& J) const
    {
        J.setZero();
        J(0, 0) = 1.0;
        J(0, 1) = 10.0;
        J(1, 2) = std::sqrt(5.0);
        J(1, 3) = -std::sqrt(5.0);
        J(2, 1) = 2.0 * (x(1) - 2.0 * x(2));
        J(2, 2) = -4.0 * (x(1) - 2.0 * x(2));
        J(3, 0) = 2.0 * std::sqrt(10.0) * (x(0) - x(3));
        J(3, 3) = -2.0 * std::sqrt(10.0) * (x(0) - x(3));
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-5.0, -5.0, -5.0, -5.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{2.0, 2.0, 5.0, 5.0}};
    }
};

// Drain a solver's per-step trajectory (objective / step_size / gradient_norm
// / first three iterate coordinates) into a coefficient-by-coefficient record.
template <typename Solver>
std::vector<std::array<double, 6>> drain_trajectory(Solver& solver)
{
    std::vector<std::array<double, 6>> trace;
    for(int i = 0; i < 60; ++i)
    {
        auto r = solver.step();
        const auto& s = solver.state();
        trace.push_back({r.objective_value, r.step_size,
                         r.gradient_norm, s.x(0), s.x(1), s.x(2)});
        if(r.policy_status)
            break;
    }
    return trace;
}

// Record the full per-step trajectory from a freshly constructed solver so the
// two dimension axes can be compared coefficient-by-coefficient.
template <typename Problem>
std::vector<std::array<double, 6>> record_trajectory(const Problem& problem,
                                                     const Eigen::VectorXd& x0)
{
    solver_options opts;
    opts.max_iterations = 60;
    opts.set_gradient_threshold(1e-12);
    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};
    return drain_trajectory(solver);
}

// Record the trajectory after exhausting one full solve and then reset()-ing
// the same solver back to x0. Any state that survives reset() (which a freshly
// constructed solver would not carry) would surface as a divergence from the
// fresh trajectory -- this is the state-leakage probe for the run-to-run
// determinism pin.
template <typename Problem>
std::vector<std::array<double, 6>> record_trajectory_after_reset(
    const Problem& problem, const Eigen::VectorXd& x0)
{
    solver_options opts;
    opts.max_iterations = 60;
    opts.set_gradient_threshold(1e-12);
    step_budget_solver solver{projected_gn_policy<>{}, problem, x0, opts};

    // Dirty the internal state with a complete solve, then restart from x0.
    (void)drain_trajectory(solver);
    solver.reset(x0);
    return drain_trajectory(solver);
}

}

TEST_CASE("projected_gn_policy: fixed-N path reproduces the dynamic-N trajectory",
          "[projected_gn][rebind]")
{
    Eigen::VectorXd x0{{3.0, -1.0, 0.0, 1.0}};

    bounded_powell_singular_ls dyn;      // problem_dimension = dynamic_dimension
    bounded_powell_singular_ls_fixed4 fix4; // problem_dimension = 4 -> rebind<4>

    auto dyn_trace = record_trajectory(dyn, x0);
    auto fix_trace = record_trajectory(fix4, x0);

    // Step-count equality is the discrete algorithmic-identity invariant: both
    // instantiations must take identical accept/reject/termination decisions.
    // It stays an exact REQUIRE -- a decision-level disagreement trips here
    // regardless of any per-element float tolerance below.
    REQUIRE(dyn_trace.size() == fix_trace.size());
    REQUIRE(dyn_trace.size() > 5);

    // Cross-path tolerance: the fixed-N and dynamic-N axes run the identical
    // Powell-singular least-squares math, but compile to different machine code
    // (fixed-size unrolled vs dynamic-loop Eigen kernels). Instruction-level
    // rounding -- FMA contraction on arm64 -- legitimately perturbs the two
    // trajectories at the 1-ULP level from the first step, and the singular
    // nonlinearity amplifies it geometrically. The cross-path spread is captured
    // on every run (see the report at the end of this case) and has been
    // measured on continuous integration: max |dyn - fix| = 2.05887e-10 on
    // arm64 (both Xcode legs, identical), and exactly 0 on both x86_64 legs
    // (gcc-14 and clang-18) and on x86_64 Windows. The spread is arm64-specific
    // and attributable to FMA contraction, which is precisely why cross-
    // architecture bit-identity is a deliberate anti-feature rather than a gap.
    // The per-element hybrid tolerance below (absolute leg for the near-zero
    // terminal regime, relative leg for the O(1) early quantities) is ~49x the
    // measured worst case (1e-8 / 2.05887e-10) and >= 6 orders below any
    // algorithmic-drift signal (a diverging branch decision moves quantities at
    // 1e-3+). A separate aggregate bound, below, guards the trajectory-wide
    // maximum against creep toward that per-element wall.
    static constexpr std::array<const char*, 6> quantity_names{
        "objective_value", "step_size", "gradient_norm", "x0", "x1", "x2"};

    double max_abs_dev = 0.0;
    double max_rel_dev = 0.0;
    for(std::size_t i = 0; i < dyn_trace.size(); ++i)
    {
        for(std::size_t k = 0; k < 6; ++k)
        {
            const double a = dyn_trace[i][k];
            const double b = fix_trace[i][k];
            const double abs_dev = std::abs(a - b);
            const double scale = std::max(std::abs(a), std::abs(b));
            const double tol = 1e-8 + 1e-8 * scale;

            max_abs_dev = std::max(max_abs_dev, abs_dev);
            if(scale > 0.0)
                max_rel_dev = std::max(max_rel_dev, abs_dev / scale);

            INFO("step " << i << " quantity " << quantity_names[k]
                         << " dyn=" << a << " fix=" << b
                         << " |dev|=" << abs_dev << " tol=" << tol);
            CHECK_THAT(a, Catch::Matchers::WithinAbs(b, tol));
        }
    }

    // Bound the aggregate spread, not only each element. The per-element check
    // above accepts anything under 1e-8; the trajectory-wide maximum is the
    // quantity that reveals creep toward that wall before it arrives, so it is
    // pinned to a single portable constant.
    //
    // The constant is swept, not guessed. Worst spread measured across every
    // platform this test runs on: 2.05887e-10 (arm64 continuous integration,
    // both Xcode legs); exactly 0 on both x86_64 legs (gcc-14, clang-18) and on
    // x86_64 Windows. One portable bound is used rather than a per-architecture
    // #ifdef, because the variation source under the Release flag set
    // (-march=native -fno-math-errno -fno-trapping-math) is the runner's CPU
    // model within an architecture, which no architecture #ifdef can see.
    //
    // The bound sits at the geometric mean of the worst measurement and the
    // 1e-8 per-element wall: sqrt(2.05887e-10 * 1e-8) ~= 1.43e-9, rounded up to
    // 1.5e-9. That places it ~7.3x above the worst measured spread -- headroom
    // absorbing CPU-model variation within an architecture, so it flakes on
    // neither the x86_64/Windows legs (which read 0) nor arm64 -- and ~6.7x
    // below the 1e-8 wall, so drift toward the wall trips this bound loudly
    // first, under --output-on-failure, before any per-element check fails.
    //
    // Only the absolute aggregate is bounded. The relative aggregate is
    // deliberately not: max_rel_dev = abs_dev / max(|a|,|b|) is structurally
    // confined to [0, 2] no matter how wrong the code becomes, so any bound on
    // it is either vacuous (>= 2) or brittle (its measured value is exactly 1,
    // the near-zero terminal element where fixed-N yields 0). It is kept in the
    // report below as a near-zero diagnostic only.
    constexpr double aggregate_abs_dev_bound = 1.5e-9;
    INFO("aggregate cross-path spread max |dyn - fix| = "
         << max_abs_dev << " must stay under " << aggregate_abs_dev_bound);
    CHECK(max_abs_dev < aggregate_abs_dev_bound);

    // Emit the observed spread unconditionally -- WARN prints regardless of
    // pass or fail -- so every run (including green ones) records the actual
    // cross-path deviation and the standing arm64 capture step reads the
    // magnitude from this line.
    WARN("cross-path trajectory spread over "
         << dyn_trace.size() << " steps x 6 quantities: max |dyn - fix| = "
         << max_abs_dev << " (abs), " << max_rel_dev << " (rel)");
}

// --------------------------------------------------------------------------
// Run-to-run determinism pin: on a fixed target the same binary produces an
// identical instruction stream, so repeating the solve must reproduce the
// trajectory bit-for-bit. This is the real RT/embedded determinism guarantee
// (same input, same output on a given target) -- exact equality is the
// correct strong invariant here and holds on every platform, unlike the
// cross-path comparison above, which spans two distinct instantiations. Both
// dimension axes are checked independently, and reset() is exercised to catch
// state leakage across solver reuse.
// --------------------------------------------------------------------------

namespace
{

void require_bit_identical(const std::vector<std::array<double, 6>>& lhs,
                           const std::vector<std::array<double, 6>>& rhs)
{
    REQUIRE(lhs.size() == rhs.size());
    REQUIRE(lhs.size() > 5);
    for(std::size_t i = 0; i < lhs.size(); ++i)
        for(std::size_t k = 0; k < 6; ++k)
            CHECK(lhs[i][k] == rhs[i][k]); // exact, bit-for-bit run to run
}

}

TEST_CASE("projected_gn_policy: trajectory is run-to-run deterministic on a fixed target",
          "[projected_gn][rebind][determinism]")
{
    Eigen::VectorXd x0{{3.0, -1.0, 0.0, 1.0}};

    SECTION("dynamic-N axis")
    {
        bounded_powell_singular_ls dyn;

        const auto run1 = record_trajectory(dyn, x0);
        const auto run2 = record_trajectory(dyn, x0);
        const auto run_reset = record_trajectory_after_reset(dyn, x0);

        require_bit_identical(run1, run2);
        require_bit_identical(run1, run_reset);
    }

    SECTION("fixed-4 axis")
    {
        bounded_powell_singular_ls_fixed4 fix4;

        const auto run1 = record_trajectory(fix4, x0);
        const auto run2 = record_trajectory(fix4, x0);
        const auto run_reset = record_trajectory_after_reset(fix4, x0);

        require_bit_identical(run1, run2);
        require_bit_identical(run1, run_reset);
    }
}
