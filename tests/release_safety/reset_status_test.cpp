#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/detail/problem_shape.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <utility>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;
using capped_cmaes = cmaes_policy<double, dynamic_dimension, 8>;

// f = 0.5 |x - (1, 2)|^2 as residuals r = x - (1, 2), with `rows` copies of the inequality x0 + x1 >= 1, so one fixture
// fits every family: a box-only family gets it with zero rows.
struct boxed_residual_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    vector2 lower;
    vector2 upper;
    std::int32_t rows;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    std::int32_t num_residuals() const { return 2; }

    double value(const vector2& x) const { return 0.5 * (x - vector2(1.0, 2.0)).squaredNorm(); }

    void gradient(const vector2& x, vector2& g) const { g = x - vector2(1.0, 2.0); }

    void residuals(const vector2& x, auto& r) const { r << x[0] - 1.0, x[1] - 2.0; }

    void jacobian(const vector2&, auto& J) const { J << 1.0, 0.0, 0.0, 1.0; }

    void constraints(const vector2& x, auto& c) const { c.setConstant(x[0] + x[1] - 1.0); }

    void constraint_jacobian(const vector2&, auto& J) const { J.setOnes(); }

    vector2 lower_bounds() const { return lower; }

    vector2 upper_bounds() const { return upper; }
};

boxed_residual_bowl make_bowl(std::int32_t rows)
{
    return boxed_residual_bowl{.lower = vector2(-5.0, -5.0), .upper = vector2(5.0, 5.0), .rows = rows};
}

template <typename Policy, typename Problem>
std::pair<solver_status, solver_status> reset_after_init(Policy policy, const Problem& problem)
{
    const vector2 x0(2.0, 1.0);
    auto state = policy.init(problem, x0, solver_options<>{});
    const solver_status reset = policy.reset(state, x0);
    return {reset, policy.reset_clear(state, x0)};
}

struct latched_state
{
    Eigen::VectorXd x;
    bool invalid_problem;
};

struct bare_state
{
    Eigen::VectorXd x;
};

}

TEMPLATE_TEST_CASE("a policy reset on a valid state reports running",
                   "[release-safety][reset]",
                   augmented_lagrangian_policy<>,
                   nw_sqp_policy<>,
                   tr_sqp_policy<>,
                   mma_policy<>,
                   ccsa_quadratic_policy<>,
                   isres_policy<>,
                   cobyla_policy<>)
{
    const auto [reset, reset_clear] = reset_after_init(typename TestType::template rebind<2>{}, make_bowl(1));
    CHECK(reset == solver_status::running);
    CHECK(reset_clear == solver_status::running);
}

TEMPLATE_TEST_CASE("a box-only policy reset on a valid state reports running", "[release-safety][reset]", lbfgsb_policy<>, lm_policy<>, bobyqa_policy<>, capped_cmaes)
{
    const auto [reset, reset_clear] = reset_after_init(typename TestType::template rebind<2>{}, make_bowl(0));
    CHECK(reset == solver_status::running);
    CHECK(reset_clear == solver_status::running);
}

TEST_CASE("a CMA-ES reset over its population cap reports invalid_problem until the population fits", "[release-safety][reset]")
{
    capped_cmaes::rebind<2> policy;
    policy.options.lambda = 6u;
    policy.options.seed = 7u;
    const auto problem = make_bowl(0);
    const vector2 x0(2.0, 1.0);
    auto state = policy.init(problem, x0, solver_options<>{});
    policy.options.lambda = 16u;
    CHECK(policy.reset(state, x0) == solver_status::invalid_problem);
    CHECK(policy.reset_clear(state, x0) == solver_status::invalid_problem);
    policy.options.lambda = 6u;
    CHECK(policy.reset(state, x0) == solver_status::running);
}

TEST_CASE("a COBYLA reset on an inverted box reports invalid_problem", "[release-safety][reset]")
{
    const auto [reset, reset_clear] = reset_after_init(cobyla_policy<>{}, boxed_residual_bowl{.lower = vector2(1.0, -1.0), .upper = vector2(0.0, 1.0), .rows = 1});
    CHECK(reset == solver_status::invalid_problem);
    CHECK(reset_clear == solver_status::invalid_problem);
}

TEST_CASE("a restarting reset reports its inner policy's running status", "[release-safety][reset]")
{
    const auto [reset, reset_clear] = reset_after_init(restarting_policy<lbfgsb_policy<double, 2>>{}, make_bowl(0));
    CHECK(reset == solver_status::running);
    CHECK(reset_clear == solver_status::running);
}

TEST_CASE("latch_status reads the invalid_problem member and defaults to running", "[release-safety][reset]")
{
    CHECK(detail::latch_status(latched_state{.x = vector2(1.0, 0.0), .invalid_problem = true}) == solver_status::invalid_problem);
    CHECK(detail::latch_status(latched_state{.x = vector2(1.0, 0.0), .invalid_problem = false}) == solver_status::running);
    CHECK(detail::latch_status(bare_state{.x = vector2(1.0, 0.0)}) == solver_status::running);
}

TEST_CASE("rejected_step on a state without an objective or gradient reports NaN", "[release-safety][reset]")
{
    const auto step = detail::rejected_step<double>(bare_state{.x = vector2(3.0, 4.0)});
    CHECK(std::isnan(step.objective_value));
    CHECK(std::isnan(step.gradient_norm));
    CHECK(step.x_norm == 5.0);
    CHECK(step.is_null_step);
    REQUIRE(step.policy_status.has_value());
    CHECK(*step.policy_status == solver_status::invalid_problem);
}
