// Seeded-determinism pins for the stochastic-ranking evolution strategy.
//
// Scope, deliberately narrow: these pins assert that one binary, given one
// seed, reproduces its own trajectory exactly. The policy draws through
// std::uniform_real_distribution and std::normal_distribution, whose output
// sequences are implementation-defined -- two standard libraries may
// legitimately produce different numbers from the same engine and the same
// seed. So no cross-platform or cross-toolchain bit-identity is claimed here,
// and these pins must not be extended into such a claim; that is a property
// the policy does not have and is not trying to have.
//
// A reset reseeds from the stored seed, so with a fixed seed a reset run
// retraces a fresh run of the same binary, for each variant.

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/solver/isres_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>
#include <cstdint>

using namespace argmin;

namespace
{

// Mirrors the problem the main stochastic-ranking suite exercises: minimize
// x0^2 + x1^2 subject to x0 + x1 >= 1 over [-10, 10]^2, optimum (0.5, 0.5).
// Restated here rather than shared so this translation unit stays independent.
struct simple_constrained
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

// A determinism assertion, not a convergence benchmark: the budget only has to
// be long enough that a seed difference has room to show up.
constexpr int trajectory_steps = 40;

solver_options<> trajectory_options()
{
    solver_options<> opts;
    opts.max_iterations = trajectory_steps;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    return opts;
}

// Drain the per-step trajectory (objective, mean step size, constraint violation
// and both iterate coordinates) coefficient by coefficient.
template <typename Solver>
std::vector<std::array<double, 5>> drain_trajectory(Solver& solver)
{
    std::vector<std::array<double, 5>> trace;
    for(int i = 0; i < trajectory_steps; ++i)
    {
        auto r = solver.step();
        const auto& s = solver.state();
        trace.push_back({r.objective_value, r.step_size, r.constraint_violation,
                         s.x(0), s.x(1)});
        if(r.policy_status)
            break;
    }
    return trace;
}

Eigen::VectorXd start_point()
{
    return Eigen::VectorXd{{2.0, 2.0}};
}

void require_bit_identical(const std::vector<std::array<double, 5>>& lhs,
                           const std::vector<std::array<double, 5>>& rhs)
{
    REQUIRE(lhs.size() == rhs.size());
    REQUIRE(lhs.size() > 5);
    for(std::size_t i = 0; i < lhs.size(); ++i)
        for(std::size_t k = 0; k < 5; ++k)
            CHECK(lhs[i][k] == rhs[i][k]); // exact, bit-for-bit
}

// True when the two trajectories disagree anywhere. Used to prove the pins
// above discriminate rather than passing on trajectories that never move.
bool differs_somewhere(const std::vector<std::array<double, 5>>& lhs,
                       const std::vector<std::array<double, 5>>& rhs)
{
    if(lhs.size() != rhs.size())
        return true;
    for(std::size_t i = 0; i < lhs.size(); ++i)
        for(std::size_t k = 0; k < 5; ++k)
            if(lhs[i][k] != rhs[i][k])
                return true;
    return false;
}


// Record the trajectory of a freshly constructed solver on the given seed.
template <typename Policy = isres_policy<>>
std::vector<std::array<double, 5>> record_trajectory(std::uint64_t seed)
{
    simple_constrained problem;
    Policy policy;
    policy.options.seed = seed;

    step_budget_solver solver{policy, problem, start_point(),
                              trajectory_options()};
    return drain_trajectory(solver);
}

enum class reset_entry : std::uint8_t
{
    warm,
    clearing
};

// Record the trajectory a solver takes after it has already been driven to
// exhaustion once and then reset back to the start point.
template <typename Policy = isres_policy<>>
std::vector<std::array<double, 5>> record_trajectory_after_reset(
    std::uint64_t seed, reset_entry entry = reset_entry::warm)
{
    simple_constrained problem;
    Policy policy;
    policy.options.seed = seed;

    step_budget_solver solver{policy, problem, start_point(),
                              trajectory_options()};

    (void)drain_trajectory(solver);
    if(entry == reset_entry::warm)
        solver.reset(start_point());
    else
        solver.reset_clear(start_point());
    return drain_trajectory(solver);
}

template <typename Policy>
void require_reset_retraces_a_fresh_run()
{
    const auto fresh_run = record_trajectory<Policy>(1u);
    require_bit_identical(fresh_run, record_trajectory_after_reset<Policy>(1u, reset_entry::warm));
    require_bit_identical(fresh_run, record_trajectory_after_reset<Policy>(1u, reset_entry::clearing));
}

template <typename Policy>
bool unset_seed_restreams_per_reset()
{
    simple_constrained problem;
    Policy policy;
    auto s = policy.init(problem, Eigen::Vector2d(2.0, 2.0), trajectory_options());
    const auto constructed = s.rng->state;
    policy.reset(s, Eigen::Vector2d(2.0, 2.0));
    const auto after_reset = s.rng->state;
    policy.reset_clear(s, Eigen::Vector2d(2.0, 2.0));
    return constructed != after_reset && after_reset != s.rng->state;
}

}

TEST_CASE("isres_policy: a fixed seed is bit-for-bit deterministic within one process",
          "[isres][determinism]")
{
    const auto run1 = record_trajectory(1u);
    const auto run2 = record_trajectory(1u);

    require_bit_identical(run1, run2);
}

// Discrimination control. Without it every pin in this file could pass while
// the policy ignored its seed outright, or while the trajectory recorded
// nothing that moves -- the check that cannot fail.
TEST_CASE("isres_policy: determinism is seed-specific, not accidental",
          "[isres][determinism]")
{
    const auto seed_1 = record_trajectory(1u);
    const auto seed_2 = record_trajectory(2u);

    CHECK(differs_somewhere(seed_1, seed_2));
}

TEST_CASE("isres variants: a fixed seed retraces a fresh run after reset and reset_clear",
          "[isres][determinism]")
{
    require_reset_retraces_a_fresh_run<alternative::isres::nlopt_faithful_policy<>>();
    require_reset_retraces_a_fresh_run<alternative::isres::original_argmin_policy<>>();
    require_reset_retraces_a_fresh_run<alternative::isres::runarsson_yao_paper_policy<>>();
}

TEST_CASE("isres variants: an unset seed starts a new stream on every reset",
          "[isres][determinism]")
{
    CHECK(unset_seed_restreams_per_reset<alternative::isres::nlopt_faithful_policy<>>());
    CHECK(unset_seed_restreams_per_reset<alternative::isres::original_argmin_policy<>>());
    CHECK(unset_seed_restreams_per_reset<alternative::isres::runarsson_yao_paper_policy<>>());
}
