// Robustness: the multi-start decorator's later runs, driven through the solve loop with default convergence options,
// and its Halton seeding past the compile-time prime table.
//
// Reference: Halton (1960); Kochenderfer and Wheeler Section 8.1
//            (multi-start global optimization).

#include "argmin/detail/halton.h"
#include "argmin/solver/options.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <cmath>
#include <vector>
#include <cstdint>

using namespace argmin;

namespace
{

double booth(const Eigen::VectorXd& x)
{
    const double t1 = x[0] + 2.0 * x[1] - 7.0;
    const double t2 = 2.0 * x[0] + x[1] - 5.0;
    return t1 * t1 + t2 * t2;
}

// Bound-constrained Booth: convex, a single minimum at (1, 3).
struct bounded_booth
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return booth(x); }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -10.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }
};

// Booth without a box.
struct free_booth
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return booth(x); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        const double t1 = x[0] + 2.0 * x[1] - 7.0;
        const double t2 = 2.0 * x[0] + x[1] - 5.0;
        g = Eigen::VectorXd{{2.0 * t1 + 4.0 * t2, 4.0 * t1 + 2.0 * t2}};
    }
};

// Records the start of every run the decorator begins after the first.
template <typename Base>
struct start_recorder : Base
{
    std::vector<Eigen::VectorXd> starts;

    template <typename State, typename X>
    solver_status reset_clear(State& s, const X& x0)
    {
        starts.emplace_back(x0);
        return Base::reset_clear(s, x0);
    }
};

// An objective that never improves on its first value and never claims, so a run ends only when the decorator's
// stagnation count runs out.
struct flat_policy
{
    using scalar_type = double;

    template <int>
    using rebind = flat_policy;

    struct options_type
    {
    };

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value;
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return {x0, problem.value(x0)};
    }

    step_result<double> step(state_type& s) { return {.objective_value = s.objective_value, .gradient_norm = 1.0, .step_size = 1.0, .objective_change = 0.0}; }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

template <typename Inner>
multistart_policy<Inner> with_restarts(std::uint16_t max_restarts)
{
    multistart_policy<Inner> policy;
    policy.options.max_restarts = max_restarts;
    return policy;
}

bool is_claim(solver_status status)
{
    return status == solver_status::converged || status == solver_status::ftol_reached || status == solver_status::xtol_reached;
}

}

TEST_CASE("multistart starts each later run at the next Halton point in the box", "[robustness][multistart][restart]")
{
    using ms_policy = multistart_policy<start_recorder<bobyqa_policy<>>>;
    const bounded_booth problem;
    const Eigen::VectorXd x0{{0.0, 0.0}};
    step_budget_solver<ms_policy, dynamic_dimension, bounded_booth> solver{with_restarts<start_recorder<bobyqa_policy<>>>(3), problem, x0};
    const auto result = solver.solve();
    const std::vector<Eigen::VectorXd>& starts = solver.policy().inner_policy_.starts;
    REQUIRE(starts.size() == 3);
    for(std::uint32_t k = 0; k < 3; ++k)
        CHECK(starts[k] == detail::halton_to_bounds<double>(k, 2, problem.lower_bounds(), problem.upper_bounds()));
    CHECK(solver.state().restart_count == 3);
    CHECK(is_claim(result.status));
}

TEST_CASE("multistart restarts from the best run's point when the problem has no box", "[robustness][multistart][restart][unbounded]")
{
    using ms_policy = multistart_policy<start_recorder<lbfgsb_policy<>>>;
    const free_booth problem;
    const Eigen::VectorXd x0{{0.0, 0.0}};
    step_budget_solver<ms_policy, dynamic_dimension, free_booth> solver{with_restarts<start_recorder<lbfgsb_policy<>>>(2), problem, x0};
    const auto result = solver.solve();
    const std::vector<Eigen::VectorXd>& starts = solver.policy().inner_policy_.starts;
    REQUIRE(starts.size() == 2);
    for(const Eigen::VectorXd& start : starts)
        CHECK((start - Eigen::VectorXd{{1.0, 3.0}}).norm() < 1e-4);
    CHECK(is_claim(result.status));
}

TEST_CASE("multistart ends a run that stops improving and starts the next", "[robustness][multistart][stall]")
{
    using ms_policy = multistart_policy<flat_policy>;
    const bounded_booth problem;
    const Eigen::VectorXd x0{{0.0, 0.0}};
    ms_policy policy = with_restarts<flat_policy>(2);
    policy.options.stall_budget_per_restart = 3u;
    step_budget_solver<ms_policy, dynamic_dimension, bounded_booth> solver{policy, problem, x0};
    const auto result = solver.solve();
    // Each run improves on its first step and stalls after three more.
    CHECK(result.iterations == 12);
    CHECK(solver.state().restart_count == 2);
    CHECK(result.status == solver_status::stalled);
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
        const Eigen::VectorXd p = detail::halton_point<double>(i, n);
        REQUIRE(p.size() == n);
        for(int d = 0; d < n; ++d)
        {
            CHECK(p[d] > 0.0);
            CHECK(p[d] < 1.0);
        }
    }
}
