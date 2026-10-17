#include "float_gate.h"
#include "sqp_float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

constexpr std::uint32_t max_iterations = 2000;
constexpr std::uint64_t seed = 42;

constexpr std::array lbfgsb_accepted_statuses{solver_status::converged};

// Under the default criteria a stalled exit is the step-tolerance stop, which only fires within the
// criterion's feasibility tolerance; both scalars end there on hs071.
constexpr std::array augmented_lagrangian_accepted_statuses{solver_status::converged, solver_status::stalled};

// CMA-ES reports ftol_reached on its own function-value stops; converged is a harness criterion.
constexpr std::array cmaes_accepted_statuses{solver_status::ftol_reached, solver_status::converged};

template <std::size_t K>
bool accepted(const std::array<solver_status, K>& statuses, solver_status status)
{
    return std::ranges::find(statuses, status) != statuses.end();
}

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive(const Problem& problem, Policy policy)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    step_budget_solver solver{policy, problem, problem.initial_point(), opts};
    return solver.solve();
}

template <scalar S>
augmented_lagrangian_policy<lbfgsb_policy<S>> augmented_lagrangian()
{
    return {};
}

template <scalar S>
multistart_policy<lbfgsb_policy<S>> multistart()
{
    return {};
}

template <scalar S>
restarting_policy<cmaes_policy<S>> restarting()
{
    restarting_policy<cmaes_policy<S>> policy{};
    policy.options.inner.seed = seed;
    return policy;
}

template <template <typename> typename Problem>
void check_augmented_lagrangian()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive(problem_d, augmented_lagrangian<double>());
    const auto result_f = drive(problem_f, augmented_lagrangian<float>());

    REQUIRE(accepted(augmented_lagrangian_accepted_statuses, result_d.status));
    REQUIRE(accepted(augmented_lagrangian_accepted_statuses, result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_wrapper));
    CHECK(max_violation(problem_f, result_f.x) <= augmented_lagrangian_defaults<float>::constraint_tolerance);
}

void check_multistart()
{
    const rosenbrock<double> problem_d;
    const rosenbrock<float> problem_f;
    const auto result_d = drive(problem_d, multistart<double>());
    const auto result_f = drive(problem_f, multistart<float>());

    REQUIRE(accepted(lbfgsb_accepted_statuses, result_d.status));
    REQUIRE(accepted(lbfgsb_accepted_statuses, result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_wrapper));
}

void check_restarting()
{
    const rosenbrock<double> problem_d;
    const rosenbrock<float> problem_f;
    const auto result_d = drive(problem_d, restarting<double>());
    const auto result_f = drive(problem_f, restarting<float>());

    REQUIRE(accepted(cmaes_accepted_statuses, result_d.status));
    REQUIRE(accepted(cmaes_accepted_statuses, result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_stochastic));
    CHECK(objective_tracks(result_f.objective_value, result_d.objective_value, k_stochastic));
}

}

TEST_CASE("wrappers over a float inner policy track double at their own defaults", "[float]")
{
    SECTION("augmented_lagrangian_policy over lbfgsb_policy hs035")
    {
        check_augmented_lagrangian<hs035>();
    }

    SECTION("augmented_lagrangian_policy over lbfgsb_policy hs071")
    {
        check_augmented_lagrangian<hs071>();
    }

    SECTION("multistart_policy over lbfgsb_policy rosenbrock")
    {
        check_multistart();
    }

    SECTION("restarting_policy over cmaes_policy rosenbrock")
    {
        check_restarting();
    }
}
