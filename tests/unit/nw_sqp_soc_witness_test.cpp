// Independent correctness witnesses for the nw_sqp-family (dense-BFGS) SQP
// second-order-correction gate. Two cases, each a hand-derived or
// re-baseline-independent assertion of the correct behavior. The first is a
// single-step Maratos SOC-trigger witness paralleling the SLSQP one; the
// second is the HS043 strictly-feasible-descent over-rejection witness.
//
// Reference: Nocedal & Wright 2e Section 18.3 (Maratos effect, second-order
//            correction); IPOPT Section 2.4 (SOC trigger); Wachter & Biegler
//            2006 Section 2.3 (filter envelope); Hock & Schittkowski 1981
//            Problem 43.

#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <numbers>

using Catch::Approx;
using namespace argmin;

namespace
{

// Maratos-effect equality problem (Powell's circle example); see the SLSQP
// witness file for the full derivation. On the circle f reduces to -x0 with
// optimum x* = (1, 0); the Lagrangian Hessian is the identity, so with the
// initial BFGS Hessian B = I the QP reproduces the exact SQP step.
//
//   min  f(x) = 2*(x0^2 + x1^2 - 1) - x0
//   s.t. c(x) = x0^2 + x1^2 - 1 = 0
struct maratos_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return 2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 4.0 * x[0] - 1.0;
        g[1] = 4.0 * x[1];
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
    }
};

double violation_at(const maratos_problem& p, const Eigen::VectorXd& x)
{
    Eigen::VectorXd c;
    p.constraints(x, c);
    return std::abs(c[0]);
}

// Joint equality + inequality Maratos problem: descend along the unit
// circle until a curved inequality becomes active. The objective uses
// the Powell-circle stabilizing term 2*(x0^2 + x1^2 - 1) so that the
// early B = I QP steps reproduce the tangential geodesic step and the
// trajectory genuinely enters the Maratos regime (measured: the SOC
// fires 3-4 times along the run; with the bare f = -x0 it never does).
//
//   min  f(x) = 2*(x0^2 + x1^2 - 1) - x0
//   s.t. c_eq(x)   = x0^2 + x1^2 - 1 = 0
//        c_ineq(x) = x1 - 1/2 + (x0 - sqrt(3)/2)^2 >= 0
//
// On the circle f reduces to -x0. Hand-derived optimum
// x* = (sqrt(3)/2, 1/2), f* = -sqrt(3)/2, both constraints active.
// KKT: with grad f(x*) = (2*sqrt(3) - 1, 2),
// grad c_eq(x*) = (sqrt(3), 1) and grad c_ineq(x*) = (0, 1), the system
// grad f = lambda * grad c_eq + mu * grad c_ineq gives
// lambda = 2 - 1/sqrt(3) and mu = +1/sqrt(3) > 0: strict
// complementarity, independent constraint gradients (LICQ). On the
// circle, d/dtheta [sin(theta) + (cos(theta) - sqrt(3)/2)^2]
// = sqrt(3)/2 > 0 at theta = pi/6, so the feasible arc is
// theta >= pi/6 and f = -cos(theta) is minimized exactly at
// theta = pi/6 = x*.
//
// This is the joint-SOC edge case: both constraints curve, so a SOC
// correction q computed against the inequality rows can violate the
// equality by O(||p||^2); the SOC QP's phase-1 re-projects onto the
// equality manifold, which in turn perturbs inequality feasibility of
// the warm start. The witness asserts the interaction converges to the
// hand-derived optimum.
struct joint_maratos_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return 2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 4.0 * x[0] - 1.0;
        g[1] = 4.0 * x[1];
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        const double a = std::sqrt(3.0) / 2.0;
        c.resize(2);
        c[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        c[1] = x[1] - 0.5 + (x[0] - a) * (x[0] - a);
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        const double a = std::sqrt(3.0) / 2.0;
        J.resize(2, 2);
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
        J(1, 0) = 2.0 * (x[0] - a);
        J(1, 1) = 1.0;
    }
};

// Box-bounded variant with a MIXED finite/infinite box that never
// activates (the iterates live on the unit circle): routes both the
// main and the SOC QP through the bounded overload and exercises the
// finite-bound-row filtering of the warm-start projection.
struct joint_maratos_bounded_problem : joint_maratos_problem
{
    Eigen::VectorXd lower_bounds() const
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return Eigen::Vector2d{-2.0, -inf};
    }

    Eigen::VectorXd upper_bounds() const
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        return Eigen::Vector2d{inf, 2.0};
    }
};

}  // namespace

// WITNESS (nw_sqp-family SOC trigger vs the Maratos effect).
//
// Parallels the SLSQP single-step SOC witness: at a near-feasible iterate
// (h_k = 0) the full unit step increases the violation (h(x_k + p) > h_k). The
// correct trigger fires a second-order correction on this full-step rejection
// whenever theta(x_k + p) >= theta(x_k); a gate keyed to the violation at the
// current iterate (h_k above a fixed threshold) can never fire at h_k = 0,
// which is exactly the Maratos regime. The single-step signature of "SOC
// fired" is diagnostics.soc_retry_count >= 1. Hand-derived quantities at
// x_k = (cos 0.1, sin 0.1):
//   p          = (sin^2 th, -sin th cos th) = ( 0.00996673, -0.09933467)
//   x_k + p    = ( 1.00497093, 0.00049875)
//   h(x_k)     = 0,   h(x_k + p) = 0.00996884 > 0  -> Maratos regime.
TEST_CASE("nw_sqp family fires a second-order correction at a near-feasible "
          "Maratos step",
          "[nw_sqp][soc][witness]")
{
    const double th = 0.1;
    Eigen::VectorXd xk{{std::cos(th), std::sin(th)}};

    maratos_problem geom;
    const double sin_th = std::sin(th);
    const double cos_th = std::cos(th);
    Eigen::VectorXd p{{sin_th * sin_th, -sin_th * cos_th}};
    const double h_k = violation_at(geom, xk);
    const double h_full = violation_at(geom, (xk + p).eval());
    CHECK(h_k == Approx(0.0).margin(1e-12));
    CHECK(h_full > h_k);

    solver_options<> opts;
    opts.max_iterations = 500;

    SECTION("nw_sqp")
    {
        maratos_problem problem;
        step_budget_solver solver{nw_sqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
    }

    SECTION("filter_nw_sqp")
    {
        maratos_problem problem;
        step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
    }
}

// WITNESS (filter_nw_sqp HS043 strictly-feasible-descent over-rejection).
//
// HS043 (4 variables, 3 nonlinear inequalities, optimum f* = -44 at
// (0, 1, 2, -1)) is the canonical filter-lineage over-rejection case: the
// filter line search rejects strictly-feasible quadratic descent trials near
// the curved active manifold and, without a working second-order correction,
// the run parks at the best strictly-feasible iterate f ~ -40.375 instead of
// descending to -44.
//
// Root cause (third in the SOC defect chain, after the trigger and the
// residual re-anchor verified by the single-step witness above): the SOC QP
// was warm-started at the main-QP solution p. p is a stationary point of the
// IDENTICAL QP objective (only the constraint RHS changes between the two
// solves), and active_set_qp_solver's phase-1 projection restores equality
// feasibility only, so an inequality-infeasible warm start froze the
// working-set loop at p (blocking_step_length clamps alpha to 0 against
// violated rows) and the "correction" degenerated to a re-test of the
// already-rejected full step (measured pre-fix: |p_soc| == |p| and
// h(x_soc) == h(x + p) on every fired SOC). The fix is the min-norm LDP
// projection of p onto the corrected inequality polyhedron
// (detail::soc_seed_projection): seed the re-solve at p + argmin ||q|| s.t.
// J_ineq * q >= b_ineq_soc - J_ineq * p, which satisfies the corrected rows
// by construction.
//
// The margin deliberately excludes the pre-fix parking value -40.375: only
// a genuine descent to the -44 basin passes.
TEST_CASE("filter_nw_sqp reaches the HS043 optimum without over-rejection",
          "[filter_nw_sqp][hs043][witness]")
{
    hs043<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-4);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-15);

    step_budget_solver solver{filter_nw_sqp_policy<hs043<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    // The optimum is f* = -44; the over-rejection failure mode parks at
    // f ~ -40.375, well outside this band.
    CHECK(result.objective_value == Approx(-44.0).margin(0.5));
    CHECK(result.constraint_violation <= opts.feasibility_tolerance);
}

// WITNESS (joint equality + inequality SOC interaction).
//
// The warm-start projection corrects the SOC seed against the
// inequality rows only; equality feasibility is restored by the QP
// solver's own phase-1 projection, which can perturb the inequality
// feasibility the projection just established (both corrections are
// O(||p||^2) near a curved active manifold). This witness pins that
// interaction end-to-end on the hand-derived joint problem above: both
// policies must descend along the circle, negotiate the curved
// inequality activation, and land on x* = (sqrt(3)/2, 1/2) with
// f* = -sqrt(3)/2.
TEST_CASE("nw_sqp family reaches the joint equality+inequality Maratos "
          "optimum",
          "[nw_sqp][filter_nw_sqp][soc][witness][joint]")
{
    const double th0 = 80.0 * std::numbers::pi / 180.0;
    Eigen::VectorXd x0{{std::cos(th0), std::sin(th0)}};
    const double f_star = -std::sqrt(3.0) / 2.0;

    solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-15);

    SECTION("nw_sqp")
    {
        joint_maratos_problem problem;
        step_budget_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value == Approx(f_star).margin(1e-6));
        CHECK(result.constraint_violation <= opts.feasibility_tolerance);
    }

    SECTION("filter_nw_sqp")
    {
        joint_maratos_problem problem;
        step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value == Approx(f_star).margin(1e-6));
        CHECK(result.constraint_violation <= opts.feasibility_tolerance);
    }

    // The witness is only meaningful if the SOC retry path is actually
    // exercised along the trajectory: accumulate the per-step
    // diagnostic over a bounded step loop and require at least one
    // firing.
    SECTION("SOC fires along the nw_sqp trajectory")
    {
        joint_maratos_problem problem;
        step_budget_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        std::size_t soc_total = 0;
        for(int k = 0; k < 60; ++k)
            soc_total += solver.step().diagnostics.soc_retry_count;
        CHECK(soc_total >= std::size_t{1});
    }
}

// WITNESS (bounded-branch SOC warm-start projection).
//
// Identical problem under a mixed finite/infinite box that never
// activates: the SOC QP takes the bounded overload, and the warm-start
// projection must stack exactly the finite bound rows (an infinite
// entry would poison the LDP's dual NNLS data). Same hand-derived
// optimum as the unbounded witness.
TEST_CASE("nw_sqp family reaches the joint Maratos optimum under an "
          "inactive mixed finite/infinite box",
          "[nw_sqp][filter_nw_sqp][soc][witness][joint][bounds]")
{
    const double th0 = 80.0 * std::numbers::pi / 180.0;
    Eigen::VectorXd x0{{std::cos(th0), std::sin(th0)}};
    const double f_star = -std::sqrt(3.0) / 2.0;

    solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-15);

    SECTION("nw_sqp")
    {
        joint_maratos_bounded_problem problem;
        step_budget_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value == Approx(f_star).margin(1e-6));
        CHECK(result.constraint_violation <= opts.feasibility_tolerance);
    }

    SECTION("filter_nw_sqp")
    {
        joint_maratos_bounded_problem problem;
        step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve(opts);
        CHECK(result.objective_value == Approx(f_star).margin(1e-6));
        CHECK(result.constraint_violation <= opts.feasibility_tolerance);
    }

    SECTION("SOC fires along the bounded nw_sqp trajectory")
    {
        joint_maratos_bounded_problem problem;
        step_budget_solver solver{nw_sqp_policy<>{}, problem, x0, opts};
        std::size_t soc_total = 0;
        for(int k = 0; k < 60; ++k)
            soc_total += solver.step().diagnostics.soc_retry_count;
        CHECK(soc_total >= std::size_t{1});
    }
}
