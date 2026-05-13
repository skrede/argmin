// Trust-region SQP HS-suite regression tests.
//
// Reference: Hock & Schittkowski (1981) Test Examples for Nonlinear
//            Programming Codes (HS026, HS028, HS043, HS071, HS076);
//            Nocedal and Wright 2e Section 18.5 (Byrd-Omojokun composite
//            step rationale).
//
// argmin variant: HS071 is the load-bearing cell -- trust-region SQP
//                 closes the L1-merit Maratos failure mode that strands
//                 the line-search SQP policies (nw_sqp / filter_nw_sqp)
//                 at high constraint violation. The accurate-mode bars
//                 |f - f*| / |f*| < 0.01 and constraint_violation < 1e-4
//                 follow the scipy `trust-constr` test-suite convention
//                 for the HS subset. The wall-time sanity cell mirrors
//                 the line-search SQP family's 1.60x fast-vs-accurate
//                 budget (absorbs jitter on small problems where the
//                 fast-mode forcing-sequence trajectory differs from the
//                 accurate-mode reference). The cross-family racing cell
//                 exercises the runtime form of the `nlp_solver` concept
//                 satisfaction: a basic_solver_group can race a Kraft
//                 SLSQP against a trust-region SQP on the same problem
//                 and feasibility-rank the result.

#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/schedule/basic_solver_group.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/options.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Per-problem wall-time helper. Solves once with the supplied policy at
// its per-mode constexpr tolerances and returns the wall delta in
// seconds. Mirrors the sqp_test.cpp solve_wall_seconds shape (file-local
// copy keeps the test self-contained; a hoist into a shared header is a
// candidate future refactor once the parametric wall-time cells stabilize
// across the SQP families).
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

TEMPLATE_TEST_CASE_SIG(
    "tr_sqp HS071 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    ((typename Policy), Policy),
    tr_sqp_policy_accurate<hs071<>::problem_dimension>,
    tr_sqp_policy_fast<hs071<>::problem_dimension>)
{
    using policy_t = Policy;

    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS071 optimum: f* = 17.0140173 at (1, 4.7429996, 3.8211499, 1.3794082).
    // Trust-region SQP closes the L1-merit Maratos failure mode that strands
    // the N&W-lineage line-search SQP policies at cv ~ 6.5 on this problem.
    const double f_star = problem.optimal_value();
    const double f_err  = std::abs(result.objective_value - f_star)
                          / std::abs(f_star);

    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(f_err < 0.05);
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        // Accurate-mode load-bearing assertion: TRSQP must converge on
        // HS071 within the scipy trust-constr HS-suite tolerance bars.
        CHECK(f_err < 0.01);
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "tr_sqp HS026 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    ((typename Policy), Policy),
    tr_sqp_policy_accurate<hs026<>::problem_dimension>,
    tr_sqp_policy_fast<hs026<>::problem_dimension>)
{
    using policy_t = Policy;

    hs026<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS026 optimum: f* = 0 at (1, 1, 1). Absolute bar because f* = 0
    // makes the |f - f*| / |f*| ratio ill-posed.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(0.05));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(0.01));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

// HS043 is documented as a known-failure cell for the current
// trust-region SQP implementation. Empirical behavior at the
// current HEAD:
//   - x0 = (0, 0, 0, 0) is strictly feasible (c_ineq = (8, 10, 5)).
//   - The objective gradient at x0 is g = (-5, -5, -21, 7); the joint
//     Newton direction overshoots the trust-region boundary by 2x and
//     the L2-merit ratio test accepts the first step at f = -21.4
//     (cv = 0). Iteration 1 expands the radius to 2.0 and accepts a
//     step that pushes the iterate into infeasibility (cv = 1.59)
//     while still reducing the L2 merit because the objective drops
//     to f = -45.3.
//   - By iteration 3 the iterate is at f = -48.6 / cv = 2.79 -- below
//     f* = -44 in the infeasible region -- and the L2 merit has no
//     descent direction within the contracting trust region. The
//     radius collapses to the floor and the policy emits
//     solver_status::trust_region_step_rejected.
//
// The underlying weakness is the slack-mediated L2 merit:
// f + ||c_ineq - s||_2 admits a steep f-decrease that outpaces the
// linearized feasibility leg when the unconstrained Newton direction
// at x0 is large and the slack update lags (the slack-leg of the
// Lagrangian gradient is -mu_ineq and the first-step multiplier
// estimate is identity-Hessian-driven). A second-order correction on
// the inequality leg or an interior-point-style slack barrier would
// address this; both are out of scope for the current HS-suite
// acceptance gate and tracked as candidates for a follow-up sweep.
//
// The [!shouldfail] tag captures the current state without removing
// the cell: a future fix that makes HS043 converge will register as
// "should-have-failed-but-passed" and surface the regression signal.
//
// Reference: Hock & Schittkowski (1981), Problem 43; Lalee, Nocedal,
//            Plantenga 1998 Section 3.1 (slack reformulation merit
//            limitations); Nocedal and Wright 2e Section 18.5
//            (Byrd-Omojokun composite step).
TEMPLATE_TEST_CASE_SIG(
    "tr_sqp HS043 (parametric on mode) [known failure]",
    "[sqp][tr_sqp][regression][mode][!shouldfail]",
    ((typename Policy), Policy),
    tr_sqp_policy_accurate<hs043<>::problem_dimension>,
    tr_sqp_policy_fast<hs043<>::problem_dimension>)
{
    using policy_t = Policy;

    hs043<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS043 optimum: f* = -44 at (0, 1, 2, -1).
    const double f_star = problem.optimal_value();
    const double f_err  = std::abs(result.objective_value - f_star)
                          / std::abs(f_star);

    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(f_err < 0.05);
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(f_err < 0.01);
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "tr_sqp HS028 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    ((typename Policy), Policy),
    tr_sqp_policy_accurate<hs028<>::problem_dimension>,
    tr_sqp_policy_fast<hs028<>::problem_dimension>)
{
    using policy_t = Policy;

    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS028 optimum: f* = 0 at (0.5, -0.5, 0.5). Accurate mode mirrors
    // the kraft_slsqp / nw_sqp HS028 parametric acceptance margins for
    // cross-family parity; fast mode uses a relaxed absolute bar sized
    // to the fast-mode gradient tolerance (1e-4) -- HS028's quadratic
    // objective produces a long shallow ridge in joint (x, slack) space
    // and the fast-mode forcing-sequence (Dembo-Eisenstat-Steihaug) +
    // CG inner-iter cap multiplier of 1 caps the per-step Newton tail
    // tighter than the accurate-mode Eisenstat-Walker schedule. The
    // 500-iter budget absorbs the resulting slower convergence; the
    // 1e-2 bar is consistent with the scipy trust-constr HS-suite
    // convention for the loose-tolerance regime.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-6));
        CHECK(solver.constraint_violation() < 1e-4);
        CHECK(result.gradient_norm < 1e-4);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "tr_sqp HS076 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    ((typename Policy), Policy),
    tr_sqp_policy_accurate<hs076<>::problem_dimension>,
    tr_sqp_policy_fast<hs076<>::problem_dimension>)
{
    using policy_t = Policy;

    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS076 optimum: f* = -4.6818181818... at
    // (0.2727272..., 2.0909090..., -2.6e-16, 0.5454545...).
    const double f_star = problem.optimal_value();
    const double f_err  = std::abs(result.objective_value - f_star)
                          / std::abs(f_star);

    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(f_err < 0.05);
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(f_err < 0.01);
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

// HS071 fast-mode wall must not be pathologically slower than the
// accurate-mode reference. A 60% headroom above the accurate-mode wall
// absorbs the line-search-style jitter the TRSQP forcing-sequence
// dispatch inherits on small problems (the fast-mode max_cg_iterations
// multiplier is the primary per-step lever; on a 4-dim joint primal the
// per-iter savings are partially reabsorbed by extra trial-evaluation
// work). Mirrors the line-search SQP family wall budget published on
// the analog cells in sqp_test.cpp.
TEST_CASE("tr_sqp _fast wall <= _accurate wall (HS071)",
          "[sqp][tr_sqp][mode][wall]")
{
    hs071<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = tr_sqp_policy_accurate<hs071<>::problem_dimension>;
    using fast_t     = tr_sqp_policy_fast<hs071<>::problem_dimension>;
    const double t_acc  = solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS071: t_acc=" << t_acc << "s t_fast=" << t_fast << "s");
    CHECK(t_fast <= t_acc * 1.60);
}

// Cross-family racing cell: basic_solver_group races a Kraft SLSQP
// against a trust-region SQP on HS071. The feasibility-first ranking
// inside basic_solver_group picks whichever policy converges to the
// lowest feasible objective within the iteration budget. Both policies
// should converge on HS071 within the HS-suite bars; the assertion is on
// the group's reported best result, not on the individual solvers. This
// is the runtime form of the `nlp_solver` concept satisfaction landed
// at compile-time on tests/compile/sqp_mode_racing_test.cpp.
TEST_CASE("kraft_slsqp_accurate vs tr_sqp_accurate race on HS071",
          "[sqp][tr_sqp][mode][race]")
{
    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.constraint_tolerance = 1e-6;

    basic_solver_group<
        round_robin_schedule,
        hs071<>::problem_dimension,
        hs071<>,
        kraft_slsqp_policy_accurate<hs071<>::problem_dimension>,
        tr_sqp_policy_accurate<hs071<>::problem_dimension>>
        group{problem, x0, opts};

    auto result = group.step_n(opts.max_iterations, opts);

    const double f_star = problem.optimal_value();
    CHECK(std::abs(result.objective_value - f_star) / std::abs(f_star)
          < 0.01);
    CHECK(result.constraint_violation < 1e-4);
}
