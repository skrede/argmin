// restarting_policy<cmaes_policy<>> decorator unit tests.
//
// Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy
//            with Increasing Population Size", CEC 2005 (decorator
//            algorithm); Hansen (2023), "The CMA Evolution Strategy:
//            A Tutorial", arXiv:1604.00772 §B.5 (reproducibility).

#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/detail/cmaes_constants.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numbers>
#include <optional>
#include <algorithm>

using namespace argmin;

namespace
{

struct bounded_ackley
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int n{2};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * std::numbers::pi;
        double sum_sq = 0.0;
        double sum_cos = 0.0;
        for(int i = 0; i < n; ++i)
        {
            sum_sq += x[i] * x[i];
            sum_cos += std::cos(c * x[i]);
        }
        const double dn = static_cast<double>(n);
        return -a * std::exp(-b * std::sqrt(sum_sq / dn))
             - std::exp(sum_cos / dn) + a + std::exp(1.0);
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

static_assert(objective<bounded_ackley>);
static_assert(bound_constrained<bounded_ackley>);

namespace
{

// Records the mean every restart hands the inner policy.
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

struct restart_trace
{
    solve_result<double> result;
    std::vector<Eigen::VectorXd> restart_means;
};

template <typename Problem>
restart_trace trace_restarts(const Problem& problem, const Eigen::VectorXd& x0)
{
    using policy_type = restarting_policy<start_recorder<cmaes_policy<>>>;
    policy_type policy;
    policy.options.inner.seed = 42u;
    policy.options.stagnation_limit = 4u;
    policy.options.max_restarts = 4u;
    step_budget_solver<policy_type, dynamic_dimension, Problem> solver{policy, problem, x0};
    const solve_result<double> result = solver.solve();
    return {result, solver.policy().inner_policy().starts};
}

// A bowl in a box, so a run that never moves from its start still reports the objective at its point.
struct boxed_norm
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }
};

// Rastrigin in its usual box: a local minimum near every integer point traps a single CMA-ES run.
struct bounded_rastrigin
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double sum = 20.0;
        for(int i = 0; i < 2; ++i)
            sum += x[i] * x[i] - 10.0 * std::cos(2.0 * std::numbers::pi * x[i]);
        return sum;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.12); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.12); }
};

// Stays at its start and never improves on it, so a run ends only on the decorator's stagnation count. With
// fail_first_run, the first run's first step fails its line search and every later run claims a stationary point on
// its first step.
struct scripted_policy
{
    using scalar_type = double;

    template <int>
    using rebind = scripted_policy;

    struct options_type
    {
        bool fail_first_run{false};
    };

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value;
        std::uint32_t run;
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return {x0, x0.squaredNorm(), 0};
    }

    step_result<double> step(state_type& s)
    {
        if(options.fail_first_run && s.run == 0)
            return {.objective_value = s.objective_value, .gradient_norm = 1.0, .is_null_step = true, .policy_status = solver_status::line_search_failed};
        return {.objective_value = s.objective_value, .gradient_norm = options.fail_first_run ? 0.0 : 1.0, .step_size = 1.0};
    }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = x0.squaredNorm();
        ++s.run;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

template <typename Inner>
step_budget_solver<restarting_policy<Inner>, dynamic_dimension, boxed_norm> scripted_solver(const boxed_norm& problem, std::optional<std::uint32_t> max_restarts, bool fail_first_run)
{
    restarting_policy<Inner> policy;
    policy.options.inner.fail_first_run = fail_first_run;
    policy.options.max_restarts = max_restarts;
    policy.options.stagnation_limit = 2u;
    return {policy, problem, Eigen::VectorXd::Constant(2, 3.0)};
}

bool is_claim(solver_status status)
{
    return status == solver_status::converged || status == solver_status::ftol_reached || status == solver_status::xtol_reached;
}

}

TEST_CASE("restarting_policy: a fixed seed reproduces a restarting solve whose restarts start from different means",
          "[restarting_policy][cmaes]")
{
    const bounded_rastrigin problem;
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    const restart_trace first = trace_restarts(problem, x0);
    const restart_trace second = trace_restarts(problem, x0);
    REQUIRE(first.restart_means.size() >= 2);
    CHECK(first.result.objective_value == second.result.objective_value);
    CHECK(first.result.iterations == second.result.iterations);
    CHECK(first.restart_means == second.restart_means);
    for(std::size_t k = 0; k < first.restart_means.size(); ++k)
    {
        CHECK(first.restart_means[k] != x0);
        CHECK((first.restart_means[k].array().abs() <= 5.12).all());
        if(k > 0)
            CHECK(first.restart_means[k] != first.restart_means[k - 1]);
    }
}

template <typename State>
std::vector<std::uint64_t> seeds_of(const State& s)
{
    std::vector<std::uint64_t> seeds{s.base_seed};
    seeds.insert(seeds.end(), s.rng.state.begin(), s.rng.state.end());
    seeds.insert(seeds.end(), s.inner.rng.state.begin(), s.inner.rng.state.end());
    return seeds;
}

template <typename Policy, typename State>
void consume_stream(Policy& policy, State& s)
{
    const std::vector<std::uint64_t> before = seeds_of(s);
    for(std::uint32_t k = 0; k < 3; ++k)
        policy.step(s);
    CHECK(seeds_of(s) != before);
}

template <typename Policy, typename Problem>
std::vector<std::vector<std::uint64_t>> seeds_after_resets(Policy policy, const Problem& problem, const Eigen::VectorXd& x0)
{
    const auto requested = policy.options.inner.seed;
    auto s = policy.init(problem, x0, solver_options<>{});
    std::vector<std::vector<std::uint64_t>> seeds{seeds_of(s)};
    consume_stream(policy, s);
    policy.reset(s, x0);
    seeds.push_back(seeds_of(s));
    consume_stream(policy, s);
    policy.reset_clear(s, x0);
    seeds.push_back(seeds_of(s));
    CHECK(policy.options.inner.seed == requested);
    return seeds;
}

TEST_CASE("restarting_policy: an unset inner seed draws a new stream on init and every reset, a fixed one replays it", "[restarting_policy][cmaes]")
{
    const bounded_ackley problem{.n = 2, .lb = Eigen::VectorXd::Constant(2, -5.0), .ub = Eigen::VectorXd::Constant(2, 5.0)};
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    restarting_policy<cmaes_policy<>> policy;

    const auto unseeded = seeds_after_resets(policy, problem, x0);
    CHECK(unseeded[0].front() != unseeded[1].front());
    CHECK(unseeded[1].front() != unseeded[2].front());
    CHECK(unseeded[0] != seeds_after_resets(policy, problem, x0)[0]);

    policy.options.inner.seed = 42u;
    const auto seeded = seeds_after_resets(policy, problem, x0);
    CHECK(seeded[0] == seeded[1]);
    CHECK(seeded[1] == seeded[2]);
    CHECK(seeded == seeds_after_resets(policy, problem, x0));
}

TEST_CASE("restarting_policy seed plumbing through restarts", "[restarting_policy][cmaes]")
{
    // Sanity check on seed discrimination. Two decorators with
    // DIFFERENT explicit inner seeds run from the same x0 must
    // produce different sample paths, observable here as different
    // final objective values within the iteration budget. If a
    // future regression makes the decorator silently ignore the
    // inner seed (e.g. by clobbering it with a hard-coded constant
    // or routing through std::random_device), this assertion would
    // either pass by random chance OR fail deterministically -- in
    // either case the regression is detectable.
    //
    // On the chosen problem / x0 / iteration budget, the two seeds
    // exercise different IPOP-restart-decorator branches and converge
    // to numerically distinct final_objective values; if a future
    // change makes them coincide, that is itself signal worth
    // investigating.
    bounded_ackley problem{
        .n = 2,
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0),
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    auto run_with_seed = [&](std::uint64_t inner_seed) {
        restarting_policy<cmaes_policy<>> policy;
        policy.options.inner.seed = inner_seed;

        step_budget_solver solver{policy, problem, x0, opts};
        return solver.solve(opts);
    };

    auto result_42 = run_with_seed(42u);
    auto result_43 = run_with_seed(43u);

    // The two seeds must drive distinguishable trajectories. We do
    // NOT require any particular ordering of the two objective
    // values -- just that they are not bit-identical. A deterministic
    // tie here would indicate the seed is being discarded.
    const bool different =
        std::bit_cast<std::uint64_t>(result_42.objective_value)
        != std::bit_cast<std::uint64_t>(result_43.objective_value);
    REQUIRE(different);
}

// Auger & Hansen (2005), CEC 2005 §III (IPOP-CMA-ES); libcmaes ipopcmastrategy.cc::reset_search_state.
TEST_CASE("restarting_policy: an inner reset_clear after a population bump recomputes every CMA-ES strategy parameter",
          "[restarting_policy][cmaes]")
{
    bounded_ackley problem{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2,  5.0),
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 0.0);
    solver_options opts;
    opts.max_iterations = 1;

    // The bump and inner reset_clear mirror what a restart does.
    restarting_policy<cmaes_policy<>> rp;
    rp.options.inner.seed = 42u;
    rp.options.inner.lambda = 6u;
    auto state = rp.init(problem, x0, opts);
    const auto params_before = state.inner.params;
    REQUIRE(params_before.lambda == 6);
    REQUIRE(static_cast<std::uint32_t>(params_before.lambda)
            == state.initial_lambda);

    const std::uint32_t new_lambda = state.initial_lambda * 2u;
    rp.inner_policy().options.lambda = new_lambda;
    rp.inner_policy().reset_clear(state.inner, x0);

    const auto& params_after = state.inner.params;
    REQUIRE(params_after.lambda == static_cast<int>(new_lambda));
    REQUIRE(params_after.mu == params_after.lambda / 2);
    REQUIRE(params_after.weights.size() == params_after.lambda);

    const auto reference = argmin::detail::compute_constants<double>(
        static_cast<int>(state.dimension),
        static_cast<int>(new_lambda));
    REQUIRE(params_after.lambda == reference.lambda);
    REQUIRE(params_after.mu == reference.mu);
    REQUIRE(params_after.mu_eff == reference.mu_eff);
    REQUIRE(params_after.c_sigma == reference.c_sigma);
    REQUIRE(params_after.d_sigma == reference.d_sigma);
    REQUIRE(params_after.c_c == reference.c_c);
    REQUIRE(params_after.c_1 == reference.c_1);
    REQUIRE(params_after.c_mu == reference.c_mu);
    REQUIRE(params_after.chi_n == reference.chi_n);
    REQUIRE(params_after.weights.size() == reference.weights.size());
    for(int i = 0; i < params_after.weights.size(); ++i)
        REQUIRE(params_after.weights[i] == reference.weights[i]);

    REQUIRE(params_before.mu_eff != params_after.mu_eff);
    REQUIRE(params_before.c_1 != params_after.c_1);
}

TEST_CASE("restarting_policy: reset and reset_clear restore the caller's inner options after a restart changed them", "[restarting_policy][cmaes]")
{
    const bounded_ackley problem{.n = 2, .lb = Eigen::VectorXd::Constant(2, -5.0), .ub = Eigen::VectorXd::Constant(2, 5.0)};
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 0.0);
    restarting_policy<cmaes_policy<>> rp;
    rp.options.inner.seed = 42u;
    rp.options.inner.lambda = 6u;
    auto state = rp.init(problem, x0, solver_options<>{});
    for(const bool clearing : {false, true})
    {
        rp.inner_policy().options.lambda = 24u;
        rp.inner_policy().options.seed = 7u;
        rp.inner_policy().reset_clear(state.inner, x0);
        if(clearing)
            rp.reset_clear(state, x0);
        else
            rp.reset(state, x0);
        CHECK(rp.inner_policy().options.lambda == rp.options.inner.lambda);
        CHECK(rp.inner_policy().options.seed == rp.options.inner.seed);
        CHECK(state.inner.params.lambda == 6);
    }
}

TEST_CASE("restarting_policy restarts a CMA-ES run that stalls in a local minimum", "[restarting_policy][cmaes]")
{
    using policy_type = restarting_policy<cmaes_policy<>>;
    const bounded_rastrigin problem;
    std::uint32_t restarted = 0;
    for(std::uint64_t seed = 1; seed <= 8; ++seed)
    {
        policy_type policy;
        policy.options.inner.seed = seed;
        step_budget_solver<policy_type, dynamic_dimension, bounded_rastrigin> solver{policy, problem, Eigen::VectorXd::Constant(2, 3.0)};
        const solve_result<double> result = solver.solve();
        if(solver.state().restart_count == 0)
            continue;
        ++restarted;
        INFO("seed " << seed);
        CHECK(is_claim(result.status));
        CHECK(result.objective_value == problem.value(result.x));
    }
    CHECK(restarted > 0);
}

TEST_CASE("restarting_policy finishes on a CMA-ES run's claim without restarting", "[restarting_policy][cmaes]")
{
    const bounded_ackley problem{.n = 2, .lb = Eigen::VectorXd::Constant(2, -5.0), .ub = Eigen::VectorXd::Constant(2, 5.0)};
    restarting_policy<cmaes_policy<>> policy;
    policy.options.inner.seed = 42u;
    step_budget_solver<restarting_policy<cmaes_policy<>>, dynamic_dimension, bounded_ackley> solver{policy, problem, Eigen::VectorXd::Constant(2, 3.0)};
    const solve_result<double> result = solver.solve();
    CHECK(solver.state().restart_count == 0);
    CHECK(is_claim(result.status));
}

TEST_CASE("restarting_policy restarts after an inner failure status and finishes on the next run's claim", "[restarting_policy]")
{
    const boxed_norm problem;
    auto solver = scripted_solver<scripted_policy>(problem, std::nullopt, true);
    const solve_result<double> result = solver.solve();
    CHECK(solver.state().restart_count == 1);
    CHECK(result.iterations == 2);
    CHECK(result.status == solver_status::converged);
    CHECK(result.x == solver.state().x);
}

TEST_CASE("restarting_policy max_restarts bounds the solve and returns the best run", "[restarting_policy]")
{
    const boxed_norm problem;
    auto solver = scripted_solver<start_recorder<scripted_policy>>(problem, 2u, false);
    const solve_result<double> result = solver.solve();
    std::vector<Eigen::VectorXd> runs = solver.policy().inner_policy().starts;
    REQUIRE(runs.size() == 2);
    runs.push_back(Eigen::VectorXd::Constant(2, 3.0));
    const auto best = std::ranges::min_element(runs, {}, [](const Eigen::VectorXd& x) { return x.squaredNorm(); });
    // Each run improves on its first step and stalls after two more.
    CHECK(result.iterations == 9);
    CHECK(result.status == solver_status::stalled);
    CHECK(result.x == *best);
    CHECK(result.objective_value == best->squaredNorm());
}
