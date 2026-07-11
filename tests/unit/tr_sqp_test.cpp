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
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/options.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <chrono>
#include <limits>
#include <cstdint>
#include <algorithm>

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
    step_budget_solver solver{Policy{}, problem, x0, opts};
    const auto t0 = std::chrono::steady_clock::now();
    [[maybe_unused]] auto result = solver.solve(opts);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// Jitter-robust per-solve wall estimate for the sub-100us HS cells: the
// minimum, over `trials` batches, of the mean wall of `batch` back-to-back
// solves. Batching lifts each measured duration well above the scheduler /
// cache noise floor; the minimum-over-trials discards contended samples
// (noise is one-sided -- it can only add time). A short untimed warmup primes
// cache and branch-prediction state. On a 4-dim joint primal this stabilizes
// the fast-vs-accurate ratio to within ~0.3% run-to-run, versus the ~2x
// spread a single steady_clock sample shows on shared runners.
template <typename Policy, typename Problem>
double min_solve_wall_seconds(const Problem& problem, const Eigen::VectorXd& x0,
                              std::uint32_t max_iters, int batch = 48,
                              int trials = 15)
{
    for(int w = 0; w < 8; ++w)
        (void)solve_wall_seconds<Policy>(problem, x0, max_iters);
    double best = std::numeric_limits<double>::max();
    for(int t = 0; t < trials; ++t)
    {
        double batch_total = 0.0;
        for(int k = 0; k < batch; ++k)
            batch_total += solve_wall_seconds<Policy>(problem, x0, max_iters);
        best = std::min(best, batch_total / static_cast<double>(batch));
    }
    return best;
}

}

TEMPLATE_TEST_CASE(
    "tr_sqp HS071 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs071<>::problem_dimension>,
    tr_sqp_policy_fast<hs071<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs071<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
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

TEMPLATE_TEST_CASE(
    "tr_sqp HS026 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs026<>::problem_dimension>,
    tr_sqp_policy_fast<hs026<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs026<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
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

// HS043 converges to the strict per-mode bar. The earlier slack-
// mediated L2-merit overshoot (a steep f-decrease outpacing the
// linearized feasibility leg on the large iter-0 Newton direction)
// was a symptom of the pre-correction ratio-test objective model and
// the unconditioned penalty: with the ratio test comparing the grad-L
// predicted decrease against the f-measured actual, and the adaptive
// L2-merit penalty growing at the re-derived default_penalty_factor
// when the model predicts a feasibility decrease, the composite step
// no longer trades feasibility for objective and the trajectory lands
// at f* = -44 with zero violation.
//
// Reference: Hock & Schittkowski (1981), Problem 43; Lalee, Nocedal,
//            Plantenga 1998 Section 3.3 (adaptive penalty growth);
//            Nocedal and Wright 2e Section 18.5 (Byrd-Omojokun
//            composite step).
TEMPLATE_TEST_CASE(
    "tr_sqp HS043 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs043<>::problem_dimension>,
    tr_sqp_policy_fast<hs043<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs043<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
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

// HS024 converges to f* = -1 at the strict per-mode bar. The former
// strictly-feasible stall was a symptom of the L2-merit ratio test's
// predicted-reduction shape before the objective model was corrected:
// the feasibility leg penalty * (||c|| - ||A p + c||) read as
// structurally negative for any step that perturbed the linearization
// off the feasibility manifold, so every non-zero step was rejected
// and the radius collapsed. With the ratio test comparing the grad-L
// predicted decrease against the f-measured actual, and the adaptive
// penalty growing at the re-derived default_penalty_factor, the
// composite step is accepted and the cell converges from the
// strictly-feasible start.
//
// Reference: Hock & Schittkowski (1981), Problem 24; Nocedal and
//            Wright 2e Section 18.5 Algorithm 18.4 (Byrd-Omojokun
//            composite step); Lalee, Nocedal, Plantenga 1998 SIAM J.
//            Optim. 8(3):682-706 Section 3.3 (augmented merit +
//            penalty update).
TEMPLATE_TEST_CASE(
    "tr_sqp HS024 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs024<>::problem_dimension>,
    tr_sqp_policy_fast<hs024<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs024<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS024 optimum: f* = -1 at (3, sqrt(3)).
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

// HS035 / HS039 / HS040 / HS050 converge to the strict per-mode bar.
// The former merit-overshoot family (HS035 / HS039 / HS040, sharing
// the equality-and-inequality slack-augmented L2-merit mechanism) and
// the HS050 composite-step-divergence cell were symptoms of the
// pre-correction ratio-test objective model and the unconditioned
// penalty growth, not of a missing acceptance criterion:
//   - With the ratio test comparing the grad-L predicted decrease
//     against the f-measured actual, the composite step no longer
//     accepts trial steps that drive the L2-merit weighted sum down by
//     trading feasibility for objective; the equality-constraint cells
//     stay on the constraint manifold instead of oscillating between
//     feasibility recovery and objective descent.
//   - The adaptive L2-merit penalty growing at the re-derived
//     default_penalty_factor (when the model predicts a feasibility
//     decrease) closes the whole family; the joint sweep confirms full
//     closure on both modes at soc_max_iterations = 0.
//   - HS050 no longer drifts off the first-order optimality conditions:
//     the corrected model keeps the KKT residual contracting along the
//     monotone-descent trajectory.
//
// Reference: Hock & Schittkowski (1981), Problems 35, 39, 40, 50;
//            Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step); Lalee, Nocedal,
//            Plantenga 1998 SIAM J. Optim. 8(3):682-706 Section 3.1
//            (v-optimal restoration) and Section 3.3 (augmented merit
//            penalty update).

TEMPLATE_TEST_CASE(
    "tr_sqp HS035 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs035<>::problem_dimension>,
    tr_sqp_policy_fast<hs035<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs035<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS035 optimum: f* = 1/9 ~ 0.11111 at (4/3, 7/9, 4/9).
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

TEMPLATE_TEST_CASE(
    "tr_sqp HS039 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs039<>::problem_dimension>,
    tr_sqp_policy_fast<hs039<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs039<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS039 optimum: f* = -1 at (1, 1, 0, 0).
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

TEMPLATE_TEST_CASE(
    "tr_sqp HS040 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs040<>::problem_dimension>,
    tr_sqp_policy_fast<hs040<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs040<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS040 optimum: f* = -0.25 at (1, 2^(1/3), 2^(1/2), -2^(-3/4)).
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

TEMPLATE_TEST_CASE(
    "tr_sqp HS050 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs050<>::problem_dimension>,
    tr_sqp_policy_fast<hs050<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs050<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS050 optimum: f* = 0 at (1, 1, 1, 1, 1). Absolute bar because
    // f* = 0 makes the |f - f*| / |f*| ratio ill-posed.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-6));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

TEMPLATE_TEST_CASE(
    "tr_sqp HS028 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs028<>::problem_dimension>,
    tr_sqp_policy_fast<hs028<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
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

TEMPLATE_TEST_CASE(
    "tr_sqp HS076 (parametric on mode)",
    "[sqp][tr_sqp][regression][mode]",
    tr_sqp_policy_accurate<hs076<>::problem_dimension>,
    tr_sqp_policy_fast<hs076<>::problem_dimension>)
{
    using Policy = TestType;
    using policy_t = Policy;

    hs076<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    step_budget_solver solver{policy_t{}, problem, x0, opts};
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
// accurate-mode reference. Both walls are measured with the jitter-robust
// minimum-over-batches estimator (see min_solve_wall_seconds) because a
// single sub-100us steady_clock sample on a shared runner is noise-dominated
// and spuriously breaches the budget. The 60% headroom above the accurate-mode
// wall then absorbs only genuine per-solve variation (the fast-mode
// max_cg_iterations multiplier is the primary per-step lever; on a 4-dim joint
// primal the per-iter savings are partially reabsorbed by extra
// trial-evaluation work). Mirrors the line-search SQP family wall budget
// published on the analog cells in sqp_test.cpp.
TEST_CASE("tr_sqp _fast wall <= _accurate wall (HS071)",
          "[sqp][tr_sqp][mode][wall]")
{
    hs071<> problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    using accurate_t = tr_sqp_policy_accurate<hs071<>::problem_dimension>;
    using fast_t     = tr_sqp_policy_fast<hs071<>::problem_dimension>;
    // Robust min-over-batches estimate: a single sub-100us wall sample is
    // dominated by scheduler/cache noise on shared CI (a lone contended
    // t_fast can spuriously breach the budget while t_acc runs clean). The
    // minimum over repeated batches recovers the true compute cost -- noise
    // only ever adds time -- so the ratio reflects the solver, not the host.
    const double t_acc  = min_solve_wall_seconds<accurate_t>(problem, x0, 200);
    const double t_fast = min_solve_wall_seconds<fast_t>(problem, x0, 200);
    INFO("HS071: t_acc=" << t_acc << "s t_fast=" << t_fast
         << "s ratio=" << (t_fast / t_acc));
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
// ─── Second-order-correction trigger pins ──────────────────────────
//
// Reference: Nocedal and Wright 2e Section 18.3 (Maratos effect; the
//            second-order correction belongs where the constraint
//            violation FAILS to fall on a rejected step);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            Section 2.4 / Algorithm A-5.7..A-5.9 (SOC trigger
//            theta(x + p) >= theta(x_k)).
//
// The SOC retry must fire only in the Maratos regime: a rejected step
// whose trial violation did NOT decrease. A rejected step whose
// violation DID decrease (ordinary merit rejection driven by the
// objective leg) must not enter the SOC loop -- re-solving against
// the corrected residual cannot help there, and the retry burns
// constraint evaluations and can perturb the iterate off the active
// manifold.

namespace
{

// Maratos-shape instance (the classical curved-constraint example):
//   min  2 (x1^2 + x2^2 - 1) - x1   s.t.  x1^2 + x2^2 - 1 = 0.
// From a feasible point on the circle the tangential SQP step leaves
// the trial violation at O(||p||^2) > 0 = theta_k while the merit
// worsens, so the primary step is rejected with the violation NOT
// decreased -- the SOC trigger regime.
struct maratos_circle
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector2d& x) const
    {
        return 2.0 * (x.squaredNorm() - 1.0) - x[0];
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        g[0] = 4.0 * x[0] - 1.0;
        g[1] = 4.0 * x[1];
    }

    void constraints(const Eigen::Vector2d& x, auto& c) const
    {
        c[0] = x.squaredNorm() - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector2d& x, auto& J) const
    {
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
    }
};

// Ordinary-rejection instance: steep quadratic objective against a
// LINEAR equality. From x0 = (0, 0) the composite step's normal leg
// contracts the violation (|0.8 - 1| = 0.2 < 1) but the trial
// objective explodes (f = 100 * 0.8^2 = 64), so the merit ratio test
// rejects while the violation DECREASED -- the non-Maratos rejection
// regime where the SOC must stay out.
struct steep_objective_linear_eq
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector2d& x) const
    {
        return 100.0 * x[0] * x[0];
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        g[0] = 200.0 * x[0];
        g[1] = 0.0;
    }

    void constraints(const Eigen::Vector2d& x, auto& c) const
    {
        c[0] = x[0] - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector2d&, auto& J) const
    {
        J(0, 0) = 1.0;
        J(0, 1) = 0.0;
    }
};

}

TEST_CASE("tr_sqp SOC retry fires on a rejected step whose violation "
          "did not decrease (Maratos regime)",
          "[sqp][tr_sqp][soc][trigger]")
{
    maratos_circle problem;
    Eigen::VectorXd x0(2);
    x0 << std::cos(0.1), std::sin(0.1);  // feasible: theta_k = 0

    solver_options opts;
    opts.max_iterations = 5;

    tr_sqp_policy_accurate<2> pol{};
    pol.options.soc_max_iterations = 2;
    step_budget_solver solver{pol, problem, x0, opts};

    // First composite step: tangential move along the circle, trial
    // violation sin^2(0.1) > 0 = theta_k, merit worsens -> rejected
    // with violation not decreased -> the SOC loop must be entered.
    auto sr = solver.step();
    INFO("soc_retry_count = " << sr.diagnostics.soc_retry_count);
    CHECK(sr.diagnostics.soc_retry_count >= 1);
}

TEST_CASE("tr_sqp SOC retry stays out on a rejected step whose "
          "violation decreased",
          "[sqp][tr_sqp][soc][trigger]")
{
    steep_objective_linear_eq problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    solver_options opts;
    opts.max_iterations = 5;

    tr_sqp_policy_accurate<2> pol{};
    pol.options.soc_max_iterations = 2;
    step_budget_solver solver{pol, problem, x0, opts};

    // First composite step: normal leg contracts the violation
    // 1 -> 0.2 but the trial objective explodes -> merit rejection
    // with violation DECREASED -> the SOC loop must NOT be entered.
    auto sr = solver.step();
    INFO("soc_retry_count = " << sr.diagnostics.soc_retry_count
         << "  is_null_step = " << sr.is_null_step);
    CHECK(sr.is_null_step);
    CHECK(sr.diagnostics.soc_retry_count == 0);
}

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
