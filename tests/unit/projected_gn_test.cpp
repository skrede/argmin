#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>

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

// SE(3)-like 6D proxy problem for IK testing (TEST-11).
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 1e-6);
}

TEST_CASE("projected_gn_policy: SE(3)-like proxy (TEST-11)", "[projected_gn]")
{
    se3_proxy_ls problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
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

    projected_gn_policy::options_type policy_opts{};
    policy_opts.use_dogleg = true;

    basic_solver solver{projected_gn_policy{}, problem, x0, opts, policy_opts};
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

    basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(solver.state().objective_value < 0.13);
    CHECK(result.x(0) == Approx(0.5).margin(1e-4));
    CHECK(result.x(1) == Approx(0.25).margin(1e-2));
}

TEST_CASE("projected_gradient_gn_policy: SE(3)-like proxy (TEST-11)", "[projected_gn]")
{
    se3_proxy_ls problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-12);

    basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts};
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

    projected_gradient_gn_policy::options_type policy_opts{};
    policy_opts.use_dogleg = true;

    basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts, policy_opts};
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};

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
        basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts};

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
        projected_gradient_gn_policy::options_type policy_opts{};
        policy_opts.use_dogleg = true;

        basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts,
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // x(0) should be at or very near the upper bound 0.5
    CHECK(result.x(0) >= 0.5 - 1e-6);
}

// --------------------------------------------------------------------------
// FAM-14 gain-ratio witness: direct model reduction at an active bound.
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

    basic_solver solver{projected_gn_policy{}, problem, x0, opts};

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

// Sibling parity: projected_gradient_gn_policy carries the identical FAM-14
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
        projected_gradient_gn_policy::options_type policy_opts{};
        policy_opts.use_dogleg = dogleg;

        basic_solver solver{projected_gradient_gn_policy{}, problem, x0, opts, policy_opts};

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
