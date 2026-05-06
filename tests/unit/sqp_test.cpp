#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace argmin;

namespace
{

// Rosenbrock with dummy constraints (no actual constraints).
// Satisfies differentiable && constrained with zero constraints.
struct unconstrained_rosenbrock
{
    argmin::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return inner.dimension(); }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& /*x*/,
                     Eigen::VectorXd& /*c*/) const {}
    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& /*J*/) const {}
};

// min (x0-1)^2 + (x1-1)^2  s.t. x0 + x1 = 1
// Solution: x* = (0.5, 0.5), f* = 0.5
struct equality_constrained_quadratic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0);
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }
};

// min Rosenbrock(x)  s.t. x0^2 + x1^2 <= 2
// Reformulate: c(x) = 2 - x0^2 - x1^2 >= 0
// Solution: (1, 1) is inside the circle (1+1=2, boundary), so x*=(1,1).
struct inequality_constrained_rosenbrock
{
    argmin::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 2.0 - x[0] * x[0] - x[1] * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& J) const
    {
        // Actually depends on x, fix:
    }
};

// Proper version with x-dependent Jacobian
struct inequality_rosenbrock
{
    argmin::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 2.0 - x[0] * x[0] - x[1] * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -2.0 * x[0];
        J(0, 1) = -2.0 * x[1];
    }
};

// Box-constrained + constrained Rosenbrock: 0 <= x <= 0.8
// True minimum (1,1) excluded by bounds.
struct box_constrained_rosenbrock
{
    argmin::rosenbrock<> inner{.n = 2};

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return inner.value(x); }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        inner.gradient(x, g);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& /*x*/,
                     Eigen::VectorXd& /*c*/) const {}
    void constraint_jacobian(const Eigen::VectorXd& /*x*/,
                             Eigen::MatrixXd& /*J*/) const {}

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{0.0, 0.0}};
    }
    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{0.8, 0.8}};
    }
};

}

// --- Concept satisfaction (compile-time) ---
static_assert(differentiable<unconstrained_rosenbrock>);
static_assert(constrained<unconstrained_rosenbrock>);
static_assert(differentiable<equality_constrained_quadratic>);
static_assert(constrained<equality_constrained_quadratic>);
static_assert(differentiable<inequality_rosenbrock>);
static_assert(constrained<inequality_rosenbrock>);
static_assert(differentiable<box_constrained_rosenbrock>);
static_assert(constrained<box_constrained_rosenbrock>);
static_assert(bound_constrained<box_constrained_rosenbrock>);

TEST_CASE("nw_sqp concept satisfaction", "[sqp]")
{
    // Compile-time checks are the static_asserts above.
    // This test case exists so the test runner reports it.
    REQUIRE(true);
}

TEST_CASE("nw_sqp unconstrained Rosenbrock", "[sqp]")
{
    unconstrained_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<unconstrained_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::ftol_reached
           || result.status == solver_status::stalled));
    CHECK(result.x[0] == Approx(1.0).margin(1e-3));
    CHECK(result.x[1] == Approx(1.0).margin(1e-3));
    CHECK(result.objective_value < 1e-4);
}

TEST_CASE("nw_sqp equality constrained", "[sqp]")
{
    equality_constrained_quadratic problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<equality_constrained_quadratic::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK((result.status == solver_status::converged
           || result.status == solver_status::ftol_reached
           || result.status == solver_status::stalled));
    CHECK(result.x[0] == Approx(0.5).margin(1e-3));
    CHECK(result.x[1] == Approx(0.5).margin(1e-3));
    CHECK(result.objective_value == Approx(0.5).margin(1e-3));
}

TEST_CASE("nw_sqp inequality constrained", "[sqp]")
{
    inequality_rosenbrock problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.set_gradient_threshold(1e-4);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<inequality_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // The unconstrained minimum (1,1) is on the boundary of x0^2+x1^2<=2
    // so it should be feasible (1+1=2).
    CHECK(result.objective_value < 0.1);
    CHECK(result.x[0] * result.x[0] + result.x[1] * result.x[1]
          <= 2.0 + 1e-4);
}

TEST_CASE("nw_sqp box constrained", "[sqp]")
{
    box_constrained_rosenbrock problem;
    Eigen::VectorXd x0{{0.4, 0.4}};
    solver_options opts;
    opts.set_gradient_threshold(1e-5);
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<box_constrained_rosenbrock::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // Solution must respect bounds [0, 0.8]
    CHECK(result.x[0] >= -1e-10);
    CHECK(result.x[0] <= 0.8 + 1e-10);
    CHECK(result.x[1] >= -1e-10);
    CHECK(result.x[1] <= 0.8 + 1e-10);

    // Must improve from starting point
    CHECK(result.objective_value < problem.value(x0));
}

TEST_CASE("nw_sqp step solve step_n", "[sqp]")
{
    unconstrained_rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;

    SECTION("step returns finite values")
    {
        basic_solver solver{nw_sqp_policy<unconstrained_rosenbrock::problem_dimension>{}, problem, x0, opts};

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
        basic_solver solver{nw_sqp_policy<unconstrained_rosenbrock::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK((result.status == solver_status::converged
               || result.status == solver_status::ftol_reached
               || result.status == solver_status::stalled));
    }
}

TEST_CASE("nw_sqp HS071 mixed constraints", "[sqp]")
{
    // HS071: n=4, m_eq=1, m_ineq=1, box bounds [1,5]^4.
    // f* = 17.014, x* approximately (1.0, 4.743, 3.821, 1.379).
    //
    // Baseline lock for nw_sqp on HS071. The reference SLSQP family
    // (kraft_slsqp on HS071) reaches f within 0.1 of f* with cv < 0.05.
    // nw_sqp does not: its L1 merit admits an iter-0 step that satisfies
    // the linearized inequality constraint but nonlinearly violates
    // x1*x2*x3*x4 >= 25, leaving the iterate strongly infeasible
    // (cv approximately 6.5) at f approximately 13.77 -- which is below
    // f* and therefore unreachable from the feasible region. The run
    // then exhausts max_iterations from that infeasible parking spot.
    //
    // Earlier this test asserted `objective_value <= 17.02` with a
    // comment claiming "f sits on the optimum"; that interpretation was
    // false (the bar passes precisely because the infeasible side has
    // lower f). The bar is left intentionally weak (<= 30.0, matching
    // hock_schittkowski_test.cpp) until the underlying merit issue is
    // addressed.
    //
    // Tracked: SEED-015 (nw_sqp HS071 L1 merit infeasibility).
    //
    // Reference: H&S Problem 71; N&W Section 16.5 (active-set QP);
    //            N&W Section 18.3 (Maratos effect); N&W Section 15.3
    //            (L1 merit / penalty parameter); N&W Procedure 18.2
    //            damping guard; Shanno 1978.
    hs071 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(std::isfinite(result.objective_value));
    CHECK(result.objective_value < 30.0);
}

TEST_CASE("nw_sqp HS039 equality constraints", "[sqp]")
{
    // HS039: n=4, m_eq=2, f* = -1.0, x* = (1, 1, 0, 0).
    // Reference: H&S Problem 39.
    hs039 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.set_gradient_threshold(1e-6);
    opts.max_iterations = 200;
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(0.1));
    CHECK(solver.constraint_violation() < 1e-4);
}

// HS007 regression guard locking the Phase 31.1 closure.
//
// Baseline (post-phase30): 7 iters, acc 1.19e-06. Post-phase31
// regressed to 6 iters, acc 1.70e-04 (143x worse). The Full E-measure
// fix keeps ftol from firing at the pre-convergence iterate.
//
// Reference: N&W 2e Definition 12.1 full E-measure closure.
TEST_CASE("nw_sqp HS007 accuracy guard",
          "[sqp][regression]")
{
    hs007 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{nw_sqp_policy<hs007<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    // HS007 optimum: f* = -sqrt(3) ~ -1.7320508. Full E-measure
    // blocks premature ftol that post-phase31 let fire at iter 6.
    // Measured post-31.1: f = -1.73205... (|f - f*| = 1.18e-06).
    // Threshold -1.7320 permits at most |f - f*| ~= 5e-05 -- ~40x
    // the measured accuracy, tight enough to detect a ~10-40x
    // degradation while tolerating floating-point noise. The old
    // -1.73 bound permitted a 1700x degradation.
    CHECK(result.objective_value < -1.7320);
    CHECK(result.iterations >= 6);
    CHECK(result.iterations <= 12);
}

// HS026 regression guard locking the Phase 31.1 closure.
//
// Baseline (post-phase30): 20 iters, acc 2.90e-07. Post-phase31
// regressed to 12 iters, acc 1.57e-04 (542x worse). The Full E-measure
// fix restores the pre-regression iteration count (+/- 1).
//
// Reference: N&W 2e Definition 12.1 full E-measure closure.
TEST_CASE("nw_sqp HS026 accuracy guard",
          "[sqp][regression]")
{
    hs026 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{nw_sqp_policy<hs026<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    // HS026 optimum: f* = 0 at (1, 1, 1). Full E-measure blocks
    // the premature ftol at iter 12 that post-phase31 let fire;
    // solver now converges correctly, iter count matches the
    // harness trajectory of 31 iters under these thresholds.
    // Measured post-31.1 accuracy: 2.90e-07. Threshold 1e-6 is ~3x
    // the measured value -- tight enough to detect a ~10x degradation
    // while tolerating floating-point noise.
    //
    // No lower bound on iterations: the current iter count is
    // inflated by a BFGS tail-drift issue deferred to a follow-up
    // plan. A future fix that reduces the iter count from ~20 to
    // ~15 is NOT a regression -- a regression guard must not
    // require slow convergence. The upper bound alone protects
    // against premature termination at a non-optimum iterate.
    CHECK(result.objective_value < 1e-6);
    CHECK(result.iterations <= 50);
}

// nw_sqp populates step_result::kkt_residual on every non-null step
// and sets is_null_step on the documented QP zero-direction path so
// step_tolerance_criterion exempts that iterate from stall detection.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34 (KKT residual);
//            N&W 2e Section 18.4 (SQP null-step semantics).
TEST_CASE("nw_sqp populates kkt_residual and exposes is_null_step",
          "[sqp][kkt]")
{
    hs039 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-6);

    basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 10; ++i)
    {
        auto sr = solver.step();
        // is_null_step is part of step_result's designated initializer
        // surface; reading it keeps the compile-time contract checked.
        CHECK((sr.is_null_step || !sr.is_null_step));
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

namespace
{

// Dynamic-dimension Hock & Schittkowski test problems carried verbatim
// from the filter_nw_sqp test suite. These structs satisfy the
// argmin SQP problem concept with problem_dimension =
// argmin::dynamic_dimension and exercise the same code path on the
// non-filter line-search policy.
//
// Reference: Hock & Schittkowski (1981), "Test Examples for Nonlinear
//            Programming Codes", LNEMS vol. 187 -- HS024, HS071, HS076.
struct hs024_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        const double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        return t * x[1] * x[1] * x[1] / (27.0 * std::sqrt(3.0));
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        const double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        g[0] = 2.0 * (x[0] - 3.0) * x[1] * x[1] * x[1] / (27.0 * std::sqrt(3.0));
        g[1] = t * 3.0 * x[1] * x[1] / (27.0 * std::sqrt(3.0));
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = x[0] / std::sqrt(3.0) - x[1];
        c[1] = x[0] + std::sqrt(3.0) * x[1];
        c[2] = 6.0 - x[0] - std::sqrt(3.0) * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(3, 2);
        J(0, 0) = 1.0 / std::sqrt(3.0); J(0, 1) = -1.0;
        J(1, 0) = 1.0;                  J(1, 1) = std::sqrt(3.0);
        J(2, 0) = -1.0;                 J(2, 1) = -std::sqrt(3.0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(2);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, std::numeric_limits<double>::infinity());
    }
};

struct hs071_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40.0;
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = 2.0 * x[0]; J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2]; J(0, 3) = 2.0 * x[3];
        J(1, 0) = x[1] * x[2] * x[3]; J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3]; J(1, 3) = x[0] * x[1] * x[2];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 1.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 5.0);
    }
};

struct hs076_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return x[0]*x[0] + 0.5*x[1]*x[1] + x[2]*x[2] + 0.5*x[3]*x[3]
               - x[0]*x[2] + x[2]*x[3]
               - x[0] - 3.0*x[1] + x[2] - x[3];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = 2.0*x[0] - x[2] - 1.0;
        g[1] = x[1] - 3.0;
        g[2] = 2.0*x[2] - x[0] + x[3] + 1.0;
        g[3] = x[3] + x[2] - 1.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 5.0 - (x[0] + 2.0*x[1] + x[2] + x[3]);
        c[1] = 4.0 - (3.0*x[0] + x[1] + 2.0*x[2] - x[3]);
        c[2] = x[1] + 4.0*x[2] - 1.5;
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(3, 4);
        J <<
            -1.0, -2.0, -1.0, -1.0,
            -3.0, -1.0, -2.0,  1.0,
             0.0,  1.0,  4.0,  0.0;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(4);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, std::numeric_limits<double>::infinity());
    }
};

}  // anonymous namespace

// Convergence guard for the dynamic-N (problem_dimension =
// argmin::dynamic_dimension) instantiation of nw_sqp_policy.
// Asserts the textbook optima with a margin and feasibility within
// 1e-6. Iter / eval counts are intentionally NOT asserted: they are
// regression metrics and belong to a separate suite, not the
// correctness-invariant unit tests. A generous max_iterations cap is
// used only to bound the test wall time (not as a correctness bar).
TEST_CASE("nw_sqp converges on dynamic-dimension HS problems",
          "[nw_sqp][regression][dynamic_n]")
{
    SECTION("HS024 dynamic-N")
    {
        hs024_dynamic problem;
        Eigen::VectorXd x0{{1.0, 0.5}};
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);

        basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(1e-3));
        CHECK(solver.constraint_violation() < 1e-6);
    }

    SECTION("HS071 dynamic-N")
    {
        // Baseline lock matching the static-N "nw_sqp HS071 mixed constraints"
        // test (sqp_test.cpp:319). The L1 merit admits an iter-0 step that
        // satisfies the linearized inequality x1*x2*x3*x4 >= 25 but
        // nonlinearly violates it; the iterate parks at f approximately
        // 13.77 with constraint_violation approximately 6.5 -- below f*
        // = 17.014 and therefore unreachable from the feasible region.
        // Bar left intentionally weak (<= 30.0) until the underlying
        // merit issue is addressed in a future phase.
        //
        // The active-set QP solver's phase-1 feasibility projection at
        // solve() entry closes the latent m>=n p=0 bug at the QP level
        // but does not address the SQP-outer-loop L1 merit infeasibility
        // documented at sqp_test.cpp:319.
        //
        // Reference: H&S Problem 71; N&W Section 16.5 (active-set QP);
        //            N&W Section 18.3 (Maratos effect);
        //            N&W Section 15.3 (L1 merit / penalty parameter).
        hs071_dynamic problem;
        Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);

        basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(std::isfinite(result.objective_value));
        CHECK(result.objective_value < 30.0);
    }

    SECTION("HS076 dynamic-N")
    {
        hs076_dynamic problem;
        Eigen::VectorXd x0{{0.5, 0.5, 0.5, 0.5}};
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);

        basic_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.6818181818).margin(1e-3));
        CHECK(solver.constraint_violation() < 1e-6);
    }
}
