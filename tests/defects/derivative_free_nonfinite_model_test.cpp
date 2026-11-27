#include "walled_bowl.h"

#include "argmin/solver/options.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;
using defects::bowl_callback;

namespace
{

template <typename Policy>
Policy with_radii(Policy policy, double initial, double final)
{
    policy.options.initial_trust_radius = initial;
    policy.options.final_trust_radius = final;
    return policy;
}

// Calls 2 to 2n + 1 = 5 evaluate the bootstrap's perturbed points; the solve starts at (0, 0), the minimum is (3, 3).
template <typename Policy>
solve_result<double> solve_with_nan_at_call(const Policy& policy, std::int32_t call)
{
    const defects::walled_bowl problem = defects::glitching(bowl_callback::value, call, true, std::numeric_limits<double>::quiet_NaN(), 0);
    const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    solver_options<> opts;
    opts.max_iterations = 5000;
    step_budget_solver solver{policy, problem, x0, opts};
    return solver.solve(opts);
}

}

TEST_CASE("bobyqa claims no convergence from a model bootstrapped on a NaN when its initial and final radii are equal", "[derivative_free]")
{
    for(std::int32_t call = 2; call <= 5; ++call)
        CHECK(solve_with_nan_at_call(with_radii(bobyqa_policy<>{}, 0.5, 0.5), call).status == solver_status::non_finite_evaluation);
}

TEST_CASE("bobyqa claims no convergence after a NaN in its bootstrap and a single contraction to the final radius", "[derivative_free]")
{
    for(std::int32_t call = 2; call <= 5; ++call)
        CHECK(solve_with_nan_at_call(with_radii(bobyqa_policy<>{}, 0.1, 0.1 / 16.0), call).status == solver_status::non_finite_evaluation);
}

// A NaN at call 2 or 4 has left the simplex before the models that stop the solve are built.
TEST_CASE("cobyla claims no step tolerance when the models that stopped it held a vertex that evaluated NaN", "[derivative_free]")
{
    for(const std::int32_t call : {3, 5})
        CHECK(solve_with_nan_at_call(with_radii(cobyla_policy<>{}, 0.1, 0.05), call).status == solver_status::non_finite_evaluation);
}

// The NaN lands on the clean run's last evaluation, a trial at the final radius that never enters the stopping models.
TEST_CASE("cobyla claims no step tolerance when its last trial at the final radius evaluated NaN", "[derivative_free]")
{
    const cobyla_policy<> policy = with_radii(cobyla_policy<>{}, 0.1, 0.05);
    const solve_result<double> clean = solve_with_nan_at_call(policy, std::numeric_limits<std::int32_t>::max());
    REQUIRE(clean.status == solver_status::xtol_reached);
    CHECK(solve_with_nan_at_call(policy, static_cast<std::int32_t>(clean.function_evaluations)).status == solver_status::non_finite_evaluation);
}
