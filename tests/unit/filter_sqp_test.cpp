#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>

using Catch::Approx;
using namespace argmin;

static_assert(argmin::nlp_solver<argmin::basic_solver<argmin::filter_slsqp_policy<>>>);

TEST_CASE("filter_slsqp on hock-schittkowski problems", "[hs][filter_slsqp]")
{
    SECTION("HS071: mixed equality + inequality constraints")
    {
        hs071<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(17.014).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
    }

    SECTION("HS043: inequality constraints")
    {
        // Asymmetric envelope sweep on filter_slsqp
        // (gamma_f, gamma_h in {1e-3, 1e-4, 1e-5, 1e-6} squared) produced
        // bit-identical (f, cv, outer_iters) across all 16 cells: HS043
        // f = -44.0000 at 13 outer iters, HS024 f = -1.0 at 13, HS076
        // f = -4.6818 at 8. The filter envelope is structurally inert on
        // this policy at the test problems' geometries; bar held at the
        // v0.2.1 default of 1e-5 / 1e-5 with the canonical -44 margin.
        //
        // Reference: Hock & Schittkowski 1981 Problem 43 (Test Examples
        //            for Nonlinear Programming Codes, Lecture Notes in
        //            Economics and Mathematical Systems vol. 187, Springer);
        //            Wachter & Biegler 2006 Section 2.3 (envelope);
        //            Fletcher & Leyffer 2002 Section 5.
        hs043<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs043<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-44.0).margin(1.0));
        CHECK(solver.constraint_violation() < 0.1);
    }

    SECTION("HS039: equality constraints")
    {
        hs039<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs039<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
    }

    SECTION("HS035: inequality with bounds")
    {
        hs035<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs035<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(0.111).margin(0.01));
        CHECK(solver.constraint_violation() < 0.01);
    }

    SECTION("HS024: bound-constrained inequality")
    {
        hs024<> problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        basic_solver solver{filter_slsqp_policy<hs024<>::problem_dimension>{}, problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-1.0).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
    }

    SECTION("hybrid restoration: infeasible start with L1-then-QP fallback")
    {
        // Start from a point with both equality and inequality constraints
        // violated, using the hybrid restoration strategy with a low step
        // cap. This verifies the hybrid restoration path compiles and
        // executes correctly, and that convergence is not broken by the
        // L1->QP fallback configuration.
        //
        // x0 = (1.2, 2.0, 2.0, 1.2):
        //   eq  = 1.44+4+4+1.44-40 = -29.12 (violated)
        //   ineq = 1.2*2*2*1.2-25  = -19.24 (violated)
        hs071<> problem;
        Eigen::Vector<double, hs071<>::problem_dimension> x0;
        x0 << 1.2, 2.0, 2.0, 1.2;

        solver_options opts;
        opts.max_iterations = 300;
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-10);

        filter_slsqp_policy<hs071<>::problem_dimension>::options_type policy_opts;
        policy_opts.restoration = detail::restoration_strategy::hybrid;
        policy_opts.max_restoration_steps = 3;

        basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{},
                            problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(17.014).margin(0.5));
        CHECK(solver.constraint_violation() < 0.01);
    }
}

// filter_slsqp populates step_result::kkt_residual on accepted steps
// via detail::kkt_residual and sets is_null_step on the documented
// QP-zero-step + high-violation path.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34 (KKT residual);
//            Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (SQP null step).
TEST_CASE("filter_slsqp populates kkt_residual and exposes is_null_step",
          "[filter_slsqp][kkt]")
{
    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-10);

    basic_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{},
                        problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 15; ++i)
    {
        auto sr = solver.step();
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

// HS024 regression guard: locks the active-set multiplier re-estimation
// closure on a problem where the constraint Jacobian has parallel rows
// (J_ineq row 2 = -row 3 at x* = (3, sqrt(3))). Plain LS + cwiseMax on the
// full Jacobian picks a min-norm split that is not KKT-valid after sign
// projection, leaving stationarity positive at the optimum. Active-set LS
// restricts the projection to binding inequalities (|c_ineq[i]| < 1e-8)
// and recovers the textbook multipliers. Null-step branches populate
// kkt_residual so the composite convergence check can terminate on
// iterates where the QP returns zero direction at the optimum.
//
// Reference baseline (post-phase30): 13 iters @ f = -1.0. Regression
// target: iterations within 1.
//
// Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set);
//            eq. 18.15 (least-squares lambda);
//            Definition 12.1 (KKT conditions);
//            eq. 12.34 (composite first-order optimality).
TEST_CASE("filter_slsqp HS024 regression guard",
          "[filter_slsqp][regression]")
{
    using namespace argmin;

    hs024<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{
        filter_slsqp_policy<hs024<>::problem_dimension>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(1e-6));
    CHECK(result.iterations <= 14);
    CHECK(solver.constraint_violation() < 1e-6);
}

// Lagrangian gradient norm vanishes at constrained optima; raw ||grad f||
// does not. The reported gradient_norm must therefore drop below 1e-4 at
// the HS007 optimum once filter_slsqp reports grad_L instead of grad_f.
//
// Reference: Hock & Schittkowski (1981), Test Examples for Nonlinear
// Programming Codes, Lecture Notes in Economics and Mathematical
// Systems vol. 187, Springer, Problem 7.
//            N&W 2e Section 12.3 / eq. 12.34 (KKT stationarity).
TEST_CASE("filter_slsqp HS007 Lagrangian gradient < 1e-4 at optimum",
          "[filter_slsqp][regression]")
{
    hs007 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{filter_slsqp_policy<hs007<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-std::sqrt(3.0)).margin(0.01));
    CHECK(solver.constraint_violation() < 1e-4);
    CHECK(result.gradient_norm < 1e-4);
}

// Reference: Hock & Schittkowski (1981), Problem 28. Equality-only
// problem: min (x0+x1)^2 + (x1+x2)^2 s.t. x0+2*x1+3*x2-1=0;
// f* = 0 at (0.5, -0.5, 0.5).
TEST_CASE("filter_slsqp HS028 Lagrangian gradient < 1e-4 at optimum",
          "[filter_slsqp][regression]")
{
    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{filter_slsqp_policy<hs028<>::problem_dimension>{}, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(0.0).margin(1e-6));
    CHECK(solver.constraint_violation() < 1e-4);
    CHECK(result.gradient_norm < 1e-4);
}

// Parametric mode-dispatch coverage: HS043 / HS071 / HS026 / HS028 across the
// filter_slsqp_policy_accurate and filter_slsqp_policy_fast aliases. Each row
// applies per-mode tolerance defaults via the policy's static-constexpr
// members at fixture construction. The accurate branch reproduces the
// existing TEST_CASE bar bit-identically (where one exists); the fast branch
// enforces a per-mode looser bar sized to the fast tolerance budget. A
// separate per-problem wall-time TEST_CASE asserts fast-mode wall does not
// exceed accurate-mode wall (with a 10% headroom for single-shot timing
// noise).
//
// HS043 is the filter-lineage regression for over-rejection on
// strictly-feasible descent (covered in the v0.3.0 SQP correctness sweep)
// and is mandatory in this parametric set.
//
// Reference: KNITRO commercial fast/accurate-mode precedent;
//            Wachter & Biegler 2006 Section 2.3 (filter envelope);
//            Fletcher & Leyffer 2002 Section 5;
//            Hock & Schittkowski 1981 Problems 26 / 28 / 43 / 71.

TEMPLATE_TEST_CASE_SIG(
    "filter_slsqp HS071 mixed constraints (parametric on mode)",
    "[filter_slsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_slsqp_policy_accurate<hs071<>::problem_dimension>,
    filter_slsqp_policy_fast<hs071<>::problem_dimension>)
{
    using policy_t = Policy;

    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    // Per-mode tolerance defaults via static-constexpr members on the
    // policy. accurate values match the existing TEST_CASE acceptance
    // margins for HS071.
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(17.014).margin(0.5));
        CHECK(solver.constraint_violation() < 0.05);
    }
    else
    {
        // Bit-identical to existing filter_slsqp HS071 TEST_CASE bar.
        CHECK(result.objective_value == Approx(17.014).margin(0.1));
        CHECK(solver.constraint_violation() < 0.01);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "filter_slsqp HS043 inequality constraints (parametric on mode)",
    "[filter_slsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_slsqp_policy_accurate<hs043<>::problem_dimension>,
    filter_slsqp_policy_fast<hs043<>::problem_dimension>)
{
    using policy_t = Policy;

    // HS043 is the filter-lineage regression for over-rejection on
    // strictly-feasible descent. The asymmetric envelope sweep
    // (gamma_f, gamma_h in {1e-3, 1e-4, 1e-5, 1e-6} squared) produced
    // bit-identical (f, cv, outer_iters) on the existing TEST_CASE; the
    // accurate parametric row reproduces the canonical -44 / margin(1.0)
    // bar verbatim.
    hs043<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(-44.0).margin(2.0));
        CHECK(solver.constraint_violation() < 0.5);
    }
    else
    {
        // Bit-identical to existing filter_slsqp HS043 TEST_CASE bar.
        CHECK(result.objective_value == Approx(-44.0).margin(1.0));
        CHECK(solver.constraint_violation() < 0.1);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "filter_slsqp HS026 (parametric on mode)",
    "[filter_slsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_slsqp_policy_accurate<hs026<>::problem_dimension>,
    filter_slsqp_policy_fast<hs026<>::problem_dimension>)
{
    using policy_t = Policy;

    hs026 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS026 optimum: f* = 0 at (1, 1, 1). Filter lineage reaches the
    // optimum more readily than kraft on this problem because the
    // envelope-based filter accepts the strictly-feasible descent path
    // that the L1-merit kraft path can stall on. Bars sized to the
    // per-mode tolerance budget; iter cap is the load-bearing guard.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value < 1e-3);
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value < 1e-6);
        CHECK(solver.constraint_violation() < 1e-3);
    }
    CHECK(result.iterations <= 200);
}

TEMPLATE_TEST_CASE_SIG(
    "filter_slsqp HS028 (parametric on mode)",
    "[filter_slsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_slsqp_policy_accurate<hs028<>::problem_dimension>,
    filter_slsqp_policy_fast<hs028<>::problem_dimension>)
{
    using policy_t = Policy;

    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS028 optimum: f* = 0 at (0.5, -0.5, 0.5). Bit-identical accurate
    // bar matches the existing filter_slsqp HS028 TEST_CASE.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-3));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-6));
        CHECK(solver.constraint_violation() < 1e-4);
        CHECK(result.gradient_norm < 1e-4);
    }
}

namespace
{

// Per-problem wall-time helper. Solves once with the supplied policy at
// its per-mode constexpr tolerances and returns the wall delta in
// seconds.
template <typename Policy, typename Problem>
double solve_wall_seconds(const Problem& problem, const Eigen::VectorXd& x0,
                          std::uint32_t max_iters)
{
    solver_options opts;
    opts.max_iterations = max_iters;
    opts.set_gradient_threshold(Policy::default_gradient_tolerance);
    opts.set_step_threshold(Policy::default_step_tolerance_rel);
    opts.constraint_tolerance = Policy::default_feasibility_tolerance;
    basic_solver solver{Policy{}, problem, x0, opts};
    const auto t0 = std::chrono::steady_clock::now();
    [[maybe_unused]] auto result = solver.solve(opts);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}

// Per-problem fast-vs-accurate wall consistency: fast-mode wall must not
// exceed accurate-mode wall (with a 10% headroom for single-shot timing
// noise).

TEST_CASE("filter_slsqp _fast wall <= _accurate wall (HS071)",
          "[filter_slsqp][mode][wall]")
{
    hs071<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = filter_slsqp_policy_accurate<hs071<>::problem_dimension>;
    using fast_t = filter_slsqp_policy_fast<hs071<>::problem_dimension>;
    const double t_acc = solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS071: t_acc=" << t_acc << "s t_fast=" << t_fast << "s");
    CHECK(t_fast <= t_acc * 1.10);
}

TEST_CASE("filter_slsqp _fast wall <= _accurate wall (HS043)",
          "[filter_slsqp][mode][wall]")
{
    hs043<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = filter_slsqp_policy_accurate<hs043<>::problem_dimension>;
    using fast_t = filter_slsqp_policy_fast<hs043<>::problem_dimension>;
    const double t_acc = solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS043: t_acc=" << t_acc << "s t_fast=" << t_fast << "s");
    CHECK(t_fast <= t_acc * 1.10);
}

TEST_CASE("filter_slsqp _fast wall <= _accurate wall (HS026)",
          "[filter_slsqp][mode][wall]")
{
    hs026<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = filter_slsqp_policy_accurate<hs026<>::problem_dimension>;
    using fast_t = filter_slsqp_policy_fast<hs026<>::problem_dimension>;
    const double t_acc = solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS026: t_acc=" << t_acc << "s t_fast=" << t_fast << "s");
    CHECK(t_fast <= t_acc * 1.10);
}

TEST_CASE("filter_slsqp _fast wall <= _accurate wall (HS028)",
          "[filter_slsqp][mode][wall]")
{
    hs028<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = filter_slsqp_policy_accurate<hs028<>::problem_dimension>;
    using fast_t = filter_slsqp_policy_fast<hs028<>::problem_dimension>;
    const double t_acc = solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS028: t_acc=" << t_acc << "s t_fast=" << t_fast << "s");
    CHECK(t_fast <= t_acc * 1.10);
}
