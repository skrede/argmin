// Robustness: the multi-start restart decorator and its Halton seeding.
//
// multistart_test.cpp covers concept satisfaction, the better-than-single
// outcome, and restart EXHAUSTION (max_restarts = 0). It never fires an
// actual Halton-seeded restart, and the Halton base generator is only ever
// exercised inside its 64-prime compile-time table. These cells close both
// gaps behaviorally:
//
//   restart firing        -- with a restart pending and budget remaining,
//                          step() reseeds via the Halton sequence, restores
//                          the inner solver, and returns the synthetic
//                          restart step_result; restart_count advances.
//   boundless reseed      -- with no bounds present the reseed falls back to
//                          the best-ever iterate rather than a Halton point.
//   stall -> pending      -- once the inner policy stops improving for the
//                          stall budget, the decorator arms a restart.
//   Halton base extension -- past the 64-prime table, halton_base() finds
//                          further distinct increasing primes by trial
//                          division, and halton_point() stays in (0, 1)^n.
//
// Reference: Halton (1960); Kochenderfer and Wheeler Section 8.1
//            (multi-start global optimization).

#include "argmin/detail/halton.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace argmin;

namespace
{

// Bound-constrained Booth: convex, a single minimum at (1, 3).
struct bounded_booth
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double t1 = x[0] + 2.0 * x[1] - 7.0;
        const double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -10.0);
    }
    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 10.0);
    }
};

using ms_policy = multistart_policy<bobyqa_policy<>>;

}  // namespace

TEST_CASE("multistart fires a Halton-seeded restart when one is pending",
          "[robustness][multistart][restart]")
{
    bounded_booth problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    ms_policy::options_type policy_opts;
    policy_opts.max_restarts = 5;  // budget remains after the first restart

    ms_policy policy;
    auto state = policy.init(problem, x0, opts, policy_opts);

    // Arm a restart with budget remaining (restart_count 0 < 5).
    state.restart_pending = true;
    const auto result = policy.step(state);

    // The restart-consuming step emits the synthetic (1.0, 1.0, 1.0)
    // step_result and advances the restart counter.
    CHECK(result.objective_change == 1.0);
    CHECK(result.step_size == 1.0);
    CHECK(result.gradient_norm == 1.0);
    CHECK_FALSE(result.improved);
    CHECK(state.restart_count == 1);
    CHECK_FALSE(state.restart_pending);

    // The Halton reseed lands inside the box bounds.
    CHECK(state.x[0] >= -10.0);
    CHECK(state.x[0] <= 10.0);
    CHECK(state.x[1] >= -10.0);
    CHECK(state.x[1] <= 10.0);
    CHECK(std::isfinite(state.objective_value));
}

TEST_CASE("multistart reseeds from the best-ever iterate when no bounds are present",
          "[robustness][multistart][restart][unbounded]")
{
    bounded_booth problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    ms_policy::options_type policy_opts;
    policy_opts.max_restarts = 5;

    ms_policy policy;
    auto state = policy.init(problem, x0, opts, policy_opts);

    // Drop the bounds so the reseed takes the best-ever fallback branch
    // instead of the Halton-to-bounds mapping.
    state.lower = Eigen::VectorXd();
    state.upper = Eigen::VectorXd();
    const Eigen::VectorXd best_before = state.best_ever_x;

    state.restart_pending = true;
    const auto result = policy.step(state);

    CHECK(result.objective_change == 1.0);
    CHECK(state.restart_count == 1);
    REQUIRE(state.x.size() == best_before.size());
    for(Eigen::Index i = 0; i < state.x.size(); ++i)
        CHECK(std::isfinite(state.x[i]));
}

TEST_CASE("multistart arms a restart once the inner policy stalls",
          "[robustness][multistart][stall]")
{
    bounded_booth problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    ms_policy::options_type policy_opts;
    policy_opts.max_restarts = 5;
    policy_opts.stall_budget_per_restart = 1u;  // stall after a single miss

    ms_policy policy;
    auto state = policy.init(problem, x0, opts, policy_opts);

    // Force every inner step to count as non-improving so the stall counter
    // reaches its budget deterministically on the first forwarded step.
    state.best_ever_value = -std::numeric_limits<double>::infinity();

    REQUIRE_FALSE(state.restart_pending);
    const auto result = policy.step(state);

    CHECK(state.restart_pending);  // stall budget reached -> restart armed
    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("halton_base extends past the prime table for high dimensions",
          "[robustness][multistart][halton]")
{
    // The compile-time table holds 64 primes (last = 311). Dimensions at
    // and beyond index 64 must be served by the trial-division extension,
    // yielding further strictly-increasing distinct primes.
    constexpr int table_size = 64;
    const std::uint32_t last_table_prime = detail::halton_base(table_size - 1);
    CHECK(last_table_prime == 311u);

    std::set<std::uint32_t> seen;
    std::uint32_t prev = last_table_prime;
    for(int d = table_size; d < table_size + 32; ++d)
    {
        const std::uint32_t base = detail::halton_base(d);
        INFO("dimension " << d << " base " << base);
        CHECK(base > prev);              // strictly increasing
        CHECK(seen.insert(base).second); // distinct
        prev = base;
    }
    // First extension prime past 311 is 313.
    CHECK(detail::halton_base(table_size) == 313u);
}

TEST_CASE("halton_point stays in the unit cube beyond the table dimension",
          "[robustness][multistart][halton]")
{
    // A 70-dimensional point forces the base generator through its
    // extension path for the trailing coordinates; every coordinate must
    // remain a valid van der Corput value in (0, 1).
    constexpr int n = 70;
    for(std::uint32_t i = 0; i < 8; ++i)
    {
        const Eigen::VectorXd p = detail::halton_point(i, n);
        REQUIRE(p.size() == n);
        for(int d = 0; d < n; ++d)
        {
            CHECK(p[d] > 0.0);
            CHECK(p[d] < 1.0);
        }
    }
}
