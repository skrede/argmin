#include "oracle/state_machine.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdint>

using namespace argmin;

namespace
{

using rosenbrock2 = rosenbrock<double, 2>;
using rosenbrock_stepper = stepper<lbfgsb_policy<double, 2>, 2, rosenbrock2>;

struct diverging_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::Vector2d x;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::Vector2d& x0, const solver_options<Convergence>&)
    {
        return {x0};
    }

    step_result<double> step(state_type& s) { return {.objective_value = 1.0, .x_norm = s.x.norm(), .policy_status = solver_status::diverged}; }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

rosenbrock_stepper rosenbrock_at_start(const rosenbrock2& problem)
{
    return {lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), solver_options<>{}, lbfgsb_policy<double, 2>::options_type{}};
}

using mma_hs076 = mma_policy<double, hs076<>::problem_dimension>;
using mma_solver = step_budget_solver<mma_hs076, hs076<>::problem_dimension, hs076<>>;

// A short stall window with best-seen tracking lets the criterion's history decide the end of an hs076 solve.
solver_options<> stall_tracking_options()
{
    solver_options<> opts;
    opts.set_stall_threshold(1e-6);
    opts.set_stall_window(5);
    std::get<stall_tolerance_criterion<>>(opts.convergence.criteria).track_best_seen_feasible = true;
    return opts;
}

mma_hs076::options_type short_stall_window()
{
    mma_hs076::options_type policy_opts;
    policy_opts.stall_window = 5;
    return policy_opts;
}

mma_solver mma_at_start(const hs076<>& problem)
{
    return {mma_hs076{}, problem, problem.initial_point(), stall_tracking_options(), short_stall_window()};
}

using mma_timed_solver = time_budget_solver<mma_hs076, hs076<>::problem_dimension, hs076<>>;

mma_timed_solver mma_timed_at_start(const hs076<>& problem)
{
    const time_budget_options<> opts{.core = stall_tracking_options(), .max_time = std::chrono::hours(1)};
    return {mma_hs076{}, problem, problem.initial_point(), opts, short_stall_window()};
}

// Each resumed call evaluates the objective once at the point it resumes from, to seed its best point.
template <typename Solver>
oracle::snapshot resumed_calls_snapshot(const Solver& whole, const solve_result<double, 4>& result, std::uint32_t resumed_calls)
{
    oracle::snapshot s = oracle::take_snapshot(whole, result);
    *s.evaluations += resumed_calls;
    return s;
}

template <typename Stepper>
void step_until_finished(Stepper& st)
{
    for(std::uint32_t i = 0; i < 1000 && !st.finished(); ++i)
        st.step();
    REQUIRE(st.finished());
}

template <typename Stepper>
std::string sticky_step_failure(Stepper& st)
{
    const oracle::snapshot before = oracle::take_snapshot(st);
    const step_result<double> r = st.step();
    return oracle::sticky_failure(before, oracle::take_snapshot(st), r);
}

}

TEST_CASE("an abort through step() ends the stepper aborted without running the policy", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    st.step();
    st.step();
    REQUIRE(st.status() == solver_status::running);
    const oracle::snapshot before = oracle::take_snapshot(st);
    st.abort();
    const step_result<double> r = st.step();
    CHECK(oracle::abort_failure(before, oracle::take_snapshot(st), r).empty());
}

TEST_CASE("a stepper stays aborted on every later step", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    st.abort();
    st.step();
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("a stepper keeps its claim status on a step after it", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    step_until_finished(st);
    REQUIRE(oracle::is_claim(st.status()));
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("a stepper keeps a policy-reported divergence on a step after it", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    stepper<diverging_policy, 2, rosenbrock2> st{problem, problem.initial_point()};
    st.step();
    REQUIRE(st.status() == solver_status::diverged);
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("stepper converged() and finished() follow the oracle's claim table", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper claimed = rosenbrock_at_start(problem);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(claimed)).empty());
    step_until_finished(claimed);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(claimed)).empty());
    stepper<diverging_policy, 2, rosenbrock2> diverged{problem, problem.initial_point()};
    diverged.step();
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(diverged)).empty());
    const rosenbrock<double> dynamic_problem{};
    stepper rejected{lbfgsb_policy<>{}, dynamic_problem, Eigen::VectorXd::Zero(3)};
    REQUIRE(rejected.status() == solver_status::invalid_problem);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(rejected)).empty());
}

TEST_CASE("a stepper reset clears its terminal status and steps again", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    step_until_finished(st);
    st.reset(problem.initial_point());
    CHECK(oracle::reset_failure(oracle::take_snapshot(st), problem.initial_point()).empty());
    const step_result<double> r = st.step();
    CHECK(r.evaluations > 0);
}

TEST_CASE("a reset_clear solve equals a solve on a freshly constructed solver", "[oracle][state-machine]")
{
    const hs076<> problem;
    mma_solver used = mma_at_start(problem);
    for(std::uint32_t i = 0; i < 1000 && used.status() == solver_status::running; ++i)
        used.step();
    REQUIRE(used.status() != solver_status::running);
    used.reset_clear(problem.initial_point());
    mma_solver fresh = mma_at_start(problem);
    const solve_result<double, 4> used_result = used.solve();
    const solve_result<double, 4> fresh_result = fresh.solve();
    CHECK(oracle::first_difference(oracle::take_snapshot(used, used_result), oracle::take_snapshot(fresh, fresh_result)).empty());
}

TEST_CASE("a defaulted step_n after step() calls equals the same number of step() calls", "[oracle][state-machine]")
{
    const hs076<> problem;
    mma_solver chunked = mma_at_start(problem);
    mma_solver stepped = mma_at_start(problem);
    for(std::uint32_t i = 0; i < 3; ++i)
    {
        chunked.step();
        stepped.step();
    }
    chunked.step_n(200);
    for(std::uint32_t i = 0; i < 200; ++i)
        stepped.step();
    REQUIRE(stepped.status() != solver_status::running);
    CHECK(oracle::first_difference(oracle::take_snapshot(chunked), oracle::take_snapshot(stepped)).empty());
}

TEST_CASE("an explicit step_n after step() calls clears the history its options carry", "[oracle][state-machine]")
{
    const hs076<> problem;
    const solver_options<> carried_opts = stall_tracking_options();
    const solver_options<> fresh_opts = stall_tracking_options();
    mma_solver other = mma_at_start(problem);
    REQUIRE(other.solve(carried_opts).status != solver_status::budget_exhausted);
    mma_solver carried = mma_at_start(problem);
    mma_solver fresh = mma_at_start(problem);
    for(std::uint32_t i = 0; i < 3; ++i)
    {
        carried.step();
        fresh.step();
    }
    const solve_result<double, 4> carried_result = carried.step_n(200, carried_opts);
    const solve_result<double, 4> fresh_result = fresh.step_n(200, fresh_opts);
    CHECK(oracle::first_difference(oracle::take_snapshot(carried, carried_result), oracle::take_snapshot(fresh, fresh_result)).empty());
}

TEST_CASE("two explicit step_n chunks on one options object equal one uninterrupted step_n", "[oracle][state-machine]")
{
    const hs076<> problem;
    const solver_options<> chunked_opts = stall_tracking_options();
    const solver_options<> whole_opts = stall_tracking_options();
    mma_solver chunked = mma_at_start(problem);
    mma_solver whole = mma_at_start(problem);
    REQUIRE(chunked.step_n(3, chunked_opts).status == solver_status::budget_exhausted);
    const solve_result<double, 4> chunked_result = chunked.step_n(200, chunked_opts);
    const solve_result<double, 4> whole_result = whole.step_n(203, whole_opts);
    REQUIRE(whole_result.status != solver_status::budget_exhausted);
    CHECK(oracle::first_difference(oracle::take_snapshot(chunked, chunked_result), resumed_calls_snapshot(whole, whole_result, 1)).empty());
}

TEST_CASE("a time_budget_solver call resumed after time_limit_reached equals one uninterrupted call", "[oracle][state-machine]")
{
    const hs076<> problem;
    time_budget_options<> resumed_opts{.core = stall_tracking_options(), .max_time = std::chrono::hours(1), .time_poll_stride = 1};
    const time_budget_options<> whole_opts = resumed_opts;
    mma_timed_solver resumed = mma_timed_at_start(problem);
    mma_timed_solver whole = mma_timed_at_start(problem);
    REQUIRE(resumed.step_n(3, resumed_opts).status == solver_status::budget_exhausted);
    resumed_opts.max_time = std::chrono::nanoseconds::zero();
    REQUIRE(resumed.step_n(200, resumed_opts).status == solver_status::time_limit_reached);
    resumed_opts.max_time = std::chrono::hours(1);
    const solve_result<double, 4> resumed_result = resumed.step_n(200, resumed_opts);
    const solve_result<double, 4> whole_result = whole.step_n(203, whole_opts);
    REQUIRE(whole_result.status != solver_status::budget_exhausted);
    CHECK(oracle::first_difference(oracle::take_snapshot(resumed, resumed_result), resumed_calls_snapshot(whole, whole_result, 2)).empty());
}
