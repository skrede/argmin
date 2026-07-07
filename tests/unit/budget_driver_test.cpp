// Behavior tests for the three budget drivers.
//
// Pins:
//   - trajectory identity: the shared solve loop and a manual step()-by-step()
//     loop over the same core reach a bit-identical iterate; the combined
//     step+time driver with an effectively-infinite deadline reproduces the
//     step-budget trajectory bit-identically.
//   - zero deadline: the time driver returns time_limit_reached within one
//     poll stride.
//   - poll cadence: a large stride defers deadline recognition (the deadline
//     is only sampled on stride boundaries), so a small deadline stops a
//     stride-1 run earlier than a stride-larger-than-budget run.
//   - concepts: all three drivers satisfy nlp_solver and steppable; only the
//     time drivers' result carries wall_time.
//   - warm-start boundary: a fixed-N MatrixBase x0 expression binds to every
//     ctor and reset without a heap materialization.
//
// Reference: N&W 2e Section 3.1; K&W 2e Section 4.4.

#include "mock_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#include "argmin/solver/step_and_time_budget_solver.h"
#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/timed_solve_result.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/formulation/concepts.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>
#include <vector>

using namespace argmin;

namespace
{

// Minimal policy whose step() sleeps a fixed duration, never converging, so
// the only stopping conditions are the iteration budget and the wall-clock
// deadline. Used to make the poll cadence observable in wall-clock time.
struct sleeping_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
    };

    std::chrono::microseconds per_step{1000};

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>&)
    {
        return state_type{.x = x0, .objective_value = 100.0};
    }

    // Never converges (mirrors non_converging_policy): a constant objective
    // with a claimed improvement and a large gradient norm keeps every
    // default convergence criterion off, so only the budget or the wall-clock
    // deadline stops the loop.
    step_result<double> step(state_type&)
    {
        std::this_thread::sleep_for(per_step);
        return step_result<double>{
            .objective_value = 100.0,
            .gradient_norm = 10.0,
            .step_size = 1.0,
            .objective_change = -1.0,
            .improved = true,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
    void reset_clear(state_type& s, const Eigen::VectorXd& x0) { reset(s, x0); }
};

// Detects whether a result type carries a wall_time member.
template <typename R>
concept has_wall_time = requires(R r) { r.wall_time; };

}

// -- Concept and result-shape static assertions --

static_assert(nlp_solver<step_budget_solver<test::mock_policy>>);
static_assert(nlp_solver<time_budget_solver<test::mock_policy>>);
static_assert(nlp_solver<step_and_time_budget_solver<test::mock_policy>>);

static_assert(steppable<step_budget_solver<test::mock_policy>>);
static_assert(steppable<time_budget_solver<test::mock_policy>>);
static_assert(steppable<step_and_time_budget_solver<test::mock_policy>>);

// The step-budget result is the chrono-free solve_result (no wall_time); the
// time drivers report a timed_solve_result that adds it.
static_assert(!has_wall_time<solve_result<double>>);
static_assert(has_wall_time<timed_solve_result<double>>);
static_assert(!has_wall_time<
    decltype(std::declval<step_budget_solver<test::mock_policy>&>().solve())>);
static_assert(has_wall_time<
    decltype(std::declval<time_budget_solver<test::mock_policy>&>().solve())>);
static_assert(has_wall_time<
    decltype(std::declval<step_and_time_budget_solver<test::mock_policy>&>().solve())>);

// -- Test: trajectory identity across the loop and a manual step loop --

TEST_CASE("step_budget loop matches a manual step() loop bit-identically",
          "[budget_driver]")
{
    rosenbrock<double, 4> prob;
    Eigen::Vector4d x0(-1.2, 1.0, -1.2, 1.0);

    // Kept below the lbfgsb/Rosenbrock convergence point (~23 iters) so the
    // loop path runs the full K without the convergence policy truncating it,
    // making the manual and loop iterate counts directly comparable.
    constexpr std::uint32_t K = 15;

    // Manual step() loop: record the per-step objective sequence and the
    // terminal iterate.
    step_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>> manual{prob, x0};
    std::vector<double> traj;
    traj.reserve(K);
    for(std::uint32_t i = 0; i < K; ++i)
        traj.push_back(manual.step().objective_value);

    // Loop path: step_n over the same setup must reach the same terminal x.
    step_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>> loop{prob, x0};
    auto loop_res = loop.step_n(K);

    for(int k = 0; k < 4; ++k)
        CHECK(loop.state().x(k) == manual.state().x(k));
    CHECK(loop_res.iterations == K);

    // A second independent manual loop reproduces the objective sequence
    // exactly (deterministic policy dynamics).
    step_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>> manual2{prob, x0};
    for(std::uint32_t i = 0; i < K; ++i)
        CHECK(manual2.step().objective_value == traj[i]);
}

TEST_CASE("step_and_time with an infinite deadline reproduces the step trajectory",
          "[budget_driver]")
{
    rosenbrock<double, 4> prob;
    Eigen::Vector4d x0(-1.2, 1.0, -1.2, 1.0);
    constexpr std::uint32_t K = 40;

    step_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>> step_only{prob, x0};
    auto step_res = step_only.step_n(K);

    time_budget_options<> opts;
    opts.max_time = std::chrono::hours(1); // effectively infinite
    step_and_time_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>>
        combined{prob, x0, opts};
    auto comb_res = combined.step_n(K, opts);

    for(int k = 0; k < 4; ++k)
    {
        CHECK(combined.state().x(k) == step_only.state().x(k));
        CHECK(comb_res.x(k) == step_res.x(k));
    }
    CHECK(comb_res.iterations == step_res.iterations);
    CHECK(comb_res.objective_value == step_res.objective_value);
    CHECK(comb_res.status == step_res.status);
}

// -- Test: zero deadline stops within one poll stride --

TEST_CASE("time budget with a zero deadline stops within one poll stride",
          "[budget_driver]")
{
    time_budget_options<> opts;
    opts.core.max_iterations = 1'000'000;
    opts.max_time = std::chrono::nanoseconds::zero();

    Eigen::VectorXd x0{{1.0, 1.0}};
    test::non_converging_policy pol;
    time_budget_solver<test::non_converging_policy> solver{pol, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::time_limit_reached);
    CHECK(result.iterations <= opts.time_poll_stride);
}

// -- Test: poll cadence defers deadline recognition --

TEST_CASE("large poll stride defers deadline recognition", "[budget_driver]")
{
    Eigen::VectorXd x0{{0.0}};
    const auto per_step = std::chrono::milliseconds(1);
    const std::uint32_t budget = 12;

    auto run = [&](std::uint32_t stride) -> std::uint32_t
    {
        sleeping_policy pol;
        pol.per_step = per_step;
        time_budget_options<> opts;
        opts.core.max_iterations = budget;
        // Deadline expires part-way through the budget (~3.5 steps in).
        opts.max_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            per_step * 7 / 2);
        opts.time_poll_stride = stride;
        time_budget_solver<sleeping_policy> solver{pol, x0, opts};
        return solver.step_n(budget, opts).iterations;
    };

    // Stride 1 polls every iteration and stops soon after the deadline.
    const std::uint32_t iters_stride1 = run(1);
    // A stride larger than the budget only polls on the first iteration, so
    // the deadline is never re-sampled and the full budget runs.
    const std::uint32_t iters_stride_big = run(budget + 100);

    CHECK(iters_stride1 < iters_stride_big);
    CHECK(iters_stride_big == budget);
}

// -- Test: fixed-N MatrixBase x0 binds to ctor and reset --

TEST_CASE("fixed-N MatrixBase x0 binds to ctor and reset", "[budget_driver]")
{
    rosenbrock<double, 4> prob;

    // CTAD from an un-rebound policy + a fixed-size Vector4d (no heap
    // materialization of a dynamic vector at the boundary).
    Eigen::Vector4d x0(-1.2, 1.0, -1.2, 1.0);
    auto solver = step_budget_solver{lbfgsb_policy<>{}, prob, x0};
    static_assert(std::is_same_v<decltype(solver)::scalar_type, double>);

    auto r0 = solver.step_n(30);
    CHECK(r0.iterations > 0);
    CHECK(r0.iterations <= 30);

    // reset with a fixed-size vector.
    solver.reset(x0);
    // reset with an Eigen expression (a float vector cast to double) -- binds
    // via MatrixBase<Derived> without an intermediate concrete vector.
    Eigen::Vector4f xf(-1.2f, 1.0f, -1.2f, 1.0f);
    solver.reset(xf.cast<double>());
    auto r1 = solver.step_n(30);
    CHECK(r1.iterations > 0);
    CHECK(r1.iterations <= 30);

    // The time drivers accept the same fixed-N x0 surface.
    time_budget_options<> topts;
    topts.max_time = std::chrono::hours(1);
    time_budget_solver<lbfgsb_policy<4>, 4, rosenbrock<double, 4>> tsolver{
        prob, x0, topts};
    tsolver.reset_clear(xf.cast<double>());
    auto r2 = tsolver.step_n(5, topts);
    CHECK(r2.iterations == 5);
}
