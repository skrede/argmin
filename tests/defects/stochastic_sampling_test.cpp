#include "stochastic_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/isres_policy.h"

#include "argmin/detail/cmaes_covariance.h"

#include "argmin/detail/alternative/cmaes_sampling_marsaglia.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using defects::sphere_under_row;

namespace
{

using isres_state = isres_policy<>::state_type<sphere_under_row>;

isres_state isres_after_init(const sphere_under_row& problem, isres_policy<>& policy)
{
    policy.options.seed = 17;
    return policy.init(problem, Eigen::Vector2d(2.0, 2.0), solver_options<>{});
}

isres_state isres_after_reset(const sphere_under_row& problem)
{
    isres_policy<> policy;
    isres_state s = isres_after_init(problem, policy);
    for(std::int32_t k = 0; k < 10; ++k)
        policy.step(s);
    policy.reset(s, Eigen::Vector2d(2.0, 2.0));
    return s;
}

std::int32_t stale_fitnesses(const sphere_under_row& problem, const isres_state& s)
{
    std::int32_t stale = 0;
    for(std::int32_t j = 0; j < s.lambda; ++j)
        stale += s.fitnesses[j] != problem.value(s.population.col(j)) ? 1 : 0;
    return stale;
}

std::int32_t stale_violations(const sphere_under_row& problem, const isres_state& s)
{
    std::int32_t stale = 0;
    for(std::int32_t j = 0; j < s.lambda; ++j)
        stale += s.violations[j] != problem.violation(s.population.col(j)) ? 1 : 0;
    return stale;
}

}

TEST_CASE("the CMA-ES covariance update leaves C exactly symmetric", "[known-defect][!shouldfail]")
{
    Eigen::MatrixXd C(8, 8);
    for(std::int32_t i = 0; i < 8; ++i)
        for(std::int32_t j = 0; j < 8; ++j)
            C(i, j) = 1.0 + i + 2.0 * j;
    bool dirty = false;
    detail::update_covariance<double>(
        C, Eigen::VectorXd::Zero(8).eval(), 1.0, 0.0, 0.0, 0.5, Eigen::VectorXd::Zero(1).eval(), Eigen::MatrixXd::Zero(8, 1).eval(), Eigen::MatrixXd::Identity(8, 8).eval(), Eigen::VectorXd::Ones(8).eval(), 0, dirty);
    CHECK(C == C.transpose());
}

TEST_CASE("the Marsaglia sampler's uniform covers (-1, 1) from a 32-bit engine", "[known-defect][!shouldfail]")
{
    std::mt19937 engine(1u);
    double largest = -2.0;
    for(std::int32_t k = 0; k < 1000; ++k)
        largest = std::max(largest, detail::alternative::marsaglia::canonical_uniform_pm1<double>(engine));
    CHECK(largest > 0.0);
}

TEST_CASE("isres init evaluates the fitness and violation of every individual it draws", "[stochastic]")
{
    const sphere_under_row problem;
    isres_policy<> policy;
    const isres_state s = isres_after_init(problem, policy);
    CHECK(stale_fitnesses(problem, s) == 0);
    CHECK(stale_violations(problem, s) == 0);
}

TEST_CASE("isres reset evaluates the fitness of every individual it redraws", "[known-defect][!shouldfail]")
{
    const sphere_under_row problem;
    CHECK(stale_fitnesses(problem, isres_after_reset(problem)) == 0);
}

TEST_CASE("isres reset evaluates the violation of every individual it redraws", "[known-defect][!shouldfail]")
{
    const sphere_under_row problem;
    CHECK(stale_violations(problem, isres_after_reset(problem)) == 0);
}
