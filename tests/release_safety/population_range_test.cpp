#include "eigen_trap.h"
#include "isres_probe.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/result/status.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <limits>
#include <cstdint>
#include <optional>

using namespace argmin;
using namespace argmin::release_safety;

namespace
{

using isres_variants = std::tuple<alternative::isres::original_argmin_policy<double, 2>,
                                  alternative::isres::nlopt_faithful_policy<double, 2>,
                                  alternative::isres::runarsson_yao_paper_policy<double, 2>>;

using pwq = alternative::cmaes::pwq_reparameterization_policy<double, 2>;
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<double, 2>;
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<double, 2>;

constexpr std::uint32_t first_wrapped = 2147483648u;
constexpr std::uint32_t last_wrapped = 4294967295u;

template <typename Policy>
Policy requesting(std::optional<std::uint32_t> lambda)
{
    Policy policy;
    policy.options.seed = 7u;
    policy.options.lambda = lambda;
    return policy;
}

template <typename Policy>
void check_cmaes_rejected(const Policy& policy)
{
    const auto problem = plane(1);
    step_budget_solver solver{policy, problem, inside, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    step_budget_solver fresh{policy, problem, inside, solver_options<>{}};
    CHECK(fresh.solve(solver_options<>{}).status == solver_status::invalid_problem);
}

template <typename Policy>
std::int32_t requested_population(std::optional<std::uint32_t> lambda)
{
    auto policy = requesting<Policy>(lambda);
    const auto problem = plane(1);
    const auto state = policy.init(problem, inside, solver_options<>{});
    CHECK_FALSE(state.invalid_problem);
    step_budget_solver solver{policy, problem, inside, solver_options<>{}};
    for(std::int32_t k = 0; k < 3; ++k)
        CHECK(solver.step().policy_status != solver_status::invalid_problem);
    return state.params.lambda;
}

}

TEMPLATE_LIST_TEST_CASE("an ISRES population of zero is rejected as an invalid problem", "[release-safety][isres]", isres_variants)
{
    check_rejected(sized<TestType>(0u, std::nullopt), plane(1), inside);
}

TEMPLATE_LIST_TEST_CASE("an ISRES population request above INT32_MAX is rejected as an invalid problem", "[release-safety][isres]", isres_variants)
{
    SECTION("the first wrapped request")
    {
        check_rejected(sized<TestType>(first_wrapped, std::nullopt), plane(1), inside);
    }
    SECTION("the largest request")
    {
        check_rejected(sized<TestType>(last_wrapped, std::nullopt), plane(1), inside);
    }
}

TEMPLATE_LIST_TEST_CASE("an ISRES parent count above the population is rejected as an invalid problem", "[release-safety][isres]", isres_variants)
{
    check_rejected(sized<TestType>(std::nullopt, 2.0), plane(1), inside);
}

TEMPLATE_LIST_TEST_CASE("an ISRES parent fraction with no representable parent count is rejected as an invalid problem", "[release-safety][isres]", isres_variants)
{
    SECTION("a NaN fraction")
    {
        check_rejected(sized<TestType>(std::nullopt, std::numeric_limits<double>::quiet_NaN()), plane(1), inside);
    }
    SECTION("a fraction past the int range")
    {
        check_rejected(sized<TestType>(std::nullopt, 1e30), plane(1), inside);
    }
}

TEMPLATE_LIST_TEST_CASE("an ISRES population of one or a parent fraction of one runs", "[release-safety][isres]", isres_variants)
{
    SECTION("a population of one")
    {
        check_accepted(sized<TestType>(1u, std::nullopt), plane(1), inside);
    }
    SECTION("every individual a parent")
    {
        check_accepted(sized<TestType>(std::nullopt, 1.0), plane(1), inside);
    }
}

TEMPLATE_TEST_CASE("a CMA-ES population request above INT32_MAX is rejected as an invalid problem", "[release-safety][cmaes]", pwq, repair_l2, no_repair)
{
    for(const std::uint32_t lambda : {first_wrapped, last_wrapped})
        check_cmaes_rejected(requesting<TestType>(lambda));
}

TEMPLATE_TEST_CASE("a CMA-ES reset rejects a population request above INT32_MAX until it fits", "[release-safety][cmaes]", pwq, repair_l2, no_repair)
{
    auto policy = requesting<TestType>(6u);
    const auto problem = plane(1);
    auto state = policy.init(problem, inside, solver_options<>{});
    policy.options.lambda = first_wrapped;
    CHECK(policy.reset(state, inside) == solver_status::invalid_problem);
    CHECK(policy.reset_clear(state, inside) == solver_status::invalid_problem);
    policy.options.lambda = 6u;
    CHECK(policy.reset(state, inside) == solver_status::running);
}

TEMPLATE_TEST_CASE("a CMA-ES population request of zero or one keeps its documented population", "[release-safety][cmaes]", pwq, repair_l2, no_repair)
{
    CHECK(requested_population<TestType>(0u) == requested_population<TestType>(std::nullopt));
    CHECK(requested_population<TestType>(1u) == 2);
}
