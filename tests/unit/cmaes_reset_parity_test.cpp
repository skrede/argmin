// restarting_policy, not the inner policy, draws a restart's mean, so reset_clear() keeps the start it is handed.
// reset() re-derives what init() derives from the configuration: the bounded population floor, the stagnation window,
// the eigendecomposition skip period and the boundary-axis cycle index. The covariance dirty flag means C changed since
// the last decomposition, so a generation that neither dirties C nor lands on the skip period reuses B and D.

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

// Bounded convex quadratic: objective && bound_constrained, no gradient.
struct bounded_quadratic
{
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return static_cast<int>(lb.size()); }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// Unbounded convex quadratic: objective only, no search box.
struct unbounded_quadratic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }
};

using pwq = alternative::cmaes::pwq_reparameterization_policy<>;
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<>;
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<>;

enum class reset_entry : std::uint8_t
{
    warm,
    clearing
};

// A narrow second coordinate, so the anisotropic initial scaling is part of what a reset must rebuild.
bounded_quadratic narrow_box()
{
    return {.lb = Eigen::VectorXd{{-5.0, -0.5}}, .ub = Eigen::VectorXd{{5.0, 0.5}}};
}

template <typename Policy>
Policy seeded(std::optional<std::uint64_t> seed)
{
    Policy policy;
    policy.options.seed = seed;
    return policy;
}

template <typename Solver>
std::vector<double> drain(Solver& solver)
{
    std::vector<double> trace;
    for(std::int32_t k = 0; k < 30; ++k)
    {
        trace.push_back(solver.step().objective_value);
        trace.insert(trace.end(), solver.state().x.begin(), solver.state().x.end());
    }
    return trace;
}

template <typename Policy>
bool reset_retraces_a_fresh_run(reset_entry entry)
{
    const bounded_quadratic problem = narrow_box();
    const Eigen::VectorXd x0{{1.0, -0.25}};
    const Eigen::VectorXd other{{-3.0, 0.4}};
    step_budget_solver fresh{seeded<Policy>(7u), problem, other, solver_options<>{}};
    step_budget_solver used{seeded<Policy>(7u), problem, x0, solver_options<>{}};
    drain(used);
    if(entry == reset_entry::warm)
        used.reset(other);
    else
        used.reset_clear(other);
    return drain(used) == drain(fresh);
}

template <typename Policy>
bool unset_seed_restreams_per_reset()
{
    const bounded_quadratic problem = narrow_box();
    const Eigen::VectorXd x0{{1.0, -0.25}};
    Policy policy;
    auto s = policy.init(problem, x0, solver_options<>{});
    const auto constructed = s.rng.state;
    policy.reset(s, x0);
    const auto after_reset = s.rng.state;
    policy.reset_clear(s, x0);
    return constructed != after_reset && after_reset != s.rng.state;
}

// Strongly anisotropic quadratic: the covariance deforms markedly, so a
// scheduled eigendecomposition visibly moves B and D.
struct anisotropic_quadratic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + 100.0 * x[1] * x[1];
    }
};

}

TEST_CASE("cmaes_policy: bounded reset_clear keeps the start it is handed as the mean",
          "[cmaes][restart]")
{
    bounded_quadratic problem{
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0)};

    Eigen::VectorXd x0{{0.0, 0.0}};
    const Eigen::VectorXd anchor{{2.5, -1.25}};
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.initial_sigma = 1.0;

    auto s = policy.init(problem, x0, opts);

    policy.reset_clear(s, anchor);
    CHECK(s.mean == anchor);

    policy.reset_clear(s, x0);
    CHECK(s.mean == x0);
}

TEST_CASE("cmaes_policy: unbounded restart keeps the initial-point anchor",
          "[cmaes][restart]")
{
    // With no finite search domain there is nothing to sample: the restart
    // mean stays at the user's initial point, so the randomization is a
    // pure no-op on unbounded problems (and reset_clear stays deterministic
    // there).
    unbounded_quadratic problem;

    Eigen::VectorXd x0{{0.3, -0.7}};
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.initial_sigma = 1.0;

    auto s = policy.init(problem, x0, opts);
    policy.reset_clear(s, x0);

    CHECK(s.mean == x0);
}

TEST_CASE("cmaes_policy: reset() re-derives the same quantities as init()",
          "[cmaes][reset]")
{
    // reset() must reproduce every configuration-derived quantity init()
    // computes, so a reset run behaves identically to a fresh run: the
    // population size, the stagnation window, the eigendecomposition skip
    // period, and the boundary-axis cycle index (which reset() previously
    // left stale).
    bounded_quadratic problem{
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0)};

    Eigen::VectorXd x0{{1.0, -2.0}};
    solver_options opts;
    opts.max_iterations = 50;
    cmaes_policy<> policy;
    policy.options.seed = 7u;
    policy.options.initial_sigma = 1.0;

    // Fresh init() derivations.
    auto s = policy.init(problem, x0, opts);
    const auto init_lambda = s.params.lambda;
    const auto init_stagnation = s.stagnation_window_min;
    const auto init_skip_k = s.decomposition_skip_k;
    const auto init_axis = s.axis_cycle_index;

    // Bounded problems use the Hansen population default 4 + floor(3 ln n)
    // = 6 at n = 2 (no widened bounded floor).
    REQUIRE(init_lambda == 6);

    // Advance several generations so the covariance state, generation
    // counter, and step-size move away from their init values, then force
    // the boundary-axis cycle index to a non-init position so reset()'s
    // restoration of it is observable independent of the mod-n wrap.
    for(int i = 0; i < 12; ++i)
        (void)policy.step(s);
    s.axis_cycle_index = init_axis + 1;
    REQUIRE(s.axis_cycle_index != init_axis);
    REQUIRE(s.generation != 0);

    // reset() must restore every derived quantity to its init value.
    policy.reset(s, x0);
    CHECK(s.params.lambda == init_lambda);
    CHECK(s.stagnation_window_min == init_stagnation);
    CHECK(s.decomposition_skip_k == init_skip_k);
    CHECK(s.axis_cycle_index == init_axis);

    // And the covariance state is a clean identity basis, matching init.
    CHECK(s.covariance_dirty == false);
    CHECK(s.C == Eigen::MatrixXd::Identity(2, 2));
    CHECK(s.D == Eigen::VectorXd::Ones(2));
    CHECK(s.initial_d_max == 1.0);
}

TEST_CASE("cmaes_policy: eigendecomposition cadence skips between refreshes",
          "[cmaes][cadence]")
{
    // The covariance moves every generation, but the eigenbasis (B, D) is
    // refreshed only once every K generations and reused unchanged in
    // between. With an explicit skip period of 3, the generations that do
    // not land on the cadence must leave B and D bit-identical, while a
    // generation that does land on the cadence refreshes them.
    anisotropic_quadratic problem;

    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 50;
    cmaes_policy<> policy;
    policy.options.seed = 3u;
    policy.options.initial_sigma = 1.0;
    policy.options.eigendecomposition_skip_generations = 3u;

    auto s = policy.init(problem, x0, opts);
    REQUIRE(s.decomposition_skip_k == 3u);

    // Generation 0 lands on the cadence (0 % 3 == 0) and decomposes.
    (void)policy.step(s);
    const Eigen::MatrixXd B_after_decomp = s.B;
    const Eigen::VectorXd D_after_decomp = s.D;

    // Generations 1 and 2 are off-cadence: B and D are reused unchanged
    // even though the covariance keeps moving.
    (void)policy.step(s);   // generation 1 at the gate -> skip
    CHECK(s.B == B_after_decomp);
    CHECK(s.D == D_after_decomp);

    (void)policy.step(s);   // generation 2 at the gate -> skip
    CHECK(s.B == B_after_decomp);
    CHECK(s.D == D_after_decomp);

    // Generation 3 lands on the cadence again and refreshes the basis; on
    // this anisotropic problem the covariance has moved, so D changes.
    (void)policy.step(s);   // generation 3 at the gate -> decompose
    CHECK(s.D != D_after_decomp);
}

TEST_CASE("cmaes variants: a fixed seed retraces a fresh run after reset", "[cmaes][reset]")
{
    CHECK(reset_retraces_a_fresh_run<pwq>(reset_entry::warm));
    CHECK(reset_retraces_a_fresh_run<repair_l2>(reset_entry::warm));
    CHECK(reset_retraces_a_fresh_run<no_repair>(reset_entry::warm));
}

TEST_CASE("cmaes variants: a fixed seed retraces a fresh run after reset_clear", "[cmaes][reset]")
{
    CHECK(reset_retraces_a_fresh_run<pwq>(reset_entry::clearing));
    CHECK(reset_retraces_a_fresh_run<repair_l2>(reset_entry::clearing));
    CHECK(reset_retraces_a_fresh_run<no_repair>(reset_entry::clearing));
}

TEST_CASE("cmaes variants: an unset seed starts a new stream on every reset", "[cmaes][reset]")
{
    CHECK(unset_seed_restreams_per_reset<pwq>());
    CHECK(unset_seed_restreams_per_reset<repair_l2>());
    CHECK(unset_seed_restreams_per_reset<no_repair>());
}
