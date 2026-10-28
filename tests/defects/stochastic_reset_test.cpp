#include "stochastic_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"

#include "argmin/detail/xoshiro256.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using defects::counting_box;

namespace
{

constexpr std::uint64_t seed = 7;
constexpr std::int32_t generations = 12;

template <typename Policy>
Policy seeded()
{
    Policy policy;
    policy.options.seed = seed;
    return policy;
}

const counting_box problem{Eigen::Vector2d::Constant(-5.0), Eigen::Vector2d::Constant(5.0), 0, 0};
const Eigen::Vector2d x0(1.0, -2.0);

template <typename Policy, typename Reset>
bool reset_restarts_the_stream(Reset reset)
{
    Policy policy = seeded<Policy>();
    const auto fresh = policy.init(problem, x0, solver_options<>{});
    auto used = policy.init(problem, x0, solver_options<>{});
    for(std::int32_t k = 0; k < generations; ++k)
        policy.step(used);
    reset(policy, used);
    return used.rng.state == fresh.rng.state;
}

template <typename Policy>
bool init_leaves_the_seeded_stream()
{
    return seeded<Policy>().init(problem, x0, solver_options<>{}).rng.state == detail::xoshiro256{seed}.state;
}

using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<>;

const auto clearing = [](auto& policy, auto& s) { policy.reset_clear(s, x0); };

const auto warm = [](auto& policy, auto& s) { policy.reset(s, x0); };

}

TEST_CASE("cmaes repair_l2_penalty restarts its random stream on reset_clear", "[known-defect][!shouldfail]")
{
    CHECK(reset_restarts_the_stream<repair_l2>(clearing));
}

TEST_CASE("cmaes restarts its random stream on reset", "[known-defect][!shouldfail]")
{
    CHECK(reset_restarts_the_stream<cmaes_policy<>>(warm));
}

TEST_CASE("a fresh cmaes init leaves its generator at the seeded state")
{
    CHECK(init_leaves_the_seeded_stream<repair_l2>());
    CHECK(init_leaves_the_seeded_stream<cmaes_policy<>>());
}
