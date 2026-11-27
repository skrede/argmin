#include "walled_bowl.h"

#include "argmin/detail/xoshiro256.h"
#include "argmin/detail/isres_operators.h"
#include "argmin/detail/stochastic_ranking.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>

using namespace argmin;
using defects::bowl_callback;

namespace
{

template <typename Policy>
Policy seeded(Policy policy)
{
    policy.options.seed = 7;
    return policy;
}

template <typename Policy>
solve_result<double> solve_from_origin(const Policy& policy, const defects::walled_bowl& problem)
{
    solver_options<> opts;
    opts.max_iterations = 300;
    step_budget_solver solver{policy, problem, Eigen::VectorXd::Zero(2).eval(), opts};
    return solver.solve(opts);
}

template <typename Check>
void for_each_isres(Check check)
{
    check(seeded(alternative::isres::nlopt_faithful_policy<>{}));
    check(seeded(alternative::isres::original_argmin_policy<>{}));
    check(seeded(alternative::isres::runarsson_yao_paper_policy<>{}));
}

}

TEST_CASE("cmaes without repair keeps a finite best objective after an offspring evaluates to -inf", "[stochastic]")
{
    const defects::walled_bowl problem = defects::glitching(bowl_callback::value, 3, true, -std::numeric_limits<double>::infinity(), 0);
    const solve_result<double> r = solve_from_origin(seeded(alternative::cmaes::no_repair_adaptive_penalty_policy<>{}), problem);
    CHECK(std::isfinite(r.objective_value));
}

TEST_CASE("isres never returns an individual whose rows evaluated NaN", "[stochastic]")
{
    for_each_isres(
        [](const auto& policy)
        {
            const solve_result<double> r = solve_from_origin(policy, defects::wall_at(1.0, 1, bowl_callback::constraints));
            CHECK(r.x[0] <= 1.0);
        });
}

TEST_CASE("isres ends non_finite_evaluation on a generation with no individual evaluated finite", "[stochastic]")
{
    for_each_isres(
        [](const auto& policy)
        {
            const solve_result<double> r = solve_from_origin(policy, defects::glitching(bowl_callback::value, 200, false, std::numeric_limits<double>::quiet_NaN(), 1));
            CHECK(r.status == solver_status::non_finite_evaluation);
            CHECK(std::isfinite(r.objective_value));
        });
}

TEST_CASE("isres reads a NaN inequality row as a NaN slack, not as satisfied", "[stochastic]")
{
    CHECK(std::isnan(detail::inequality_slack(std::numeric_limits<double>::quiet_NaN())));
    CHECK(detail::inequality_slack(-2.0) == 2.0);
    CHECK(detail::inequality_slack(2.0) == 0.0);
}

TEST_CASE("stochastic ranking sinks an individual with a NaN objective or violation below a finite one", "[stochastic]")
{
    constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd objectives(3);
    objectives << not_a_number, 1.0, 2.0;
    Eigen::VectorXd violations(3);
    violations << 0.0, not_a_number, 0.0;
    std::vector<std::uint32_t> indices{0, 1, 2};
    detail::xoshiro256 rng{1u};
    detail::stochastic_rank(indices, objectives, violations, 0.45, rng);
    CHECK(indices == std::vector<std::uint32_t>{2, 0, 1});
}
