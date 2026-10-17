#include "float_gate.h"
#include "float_fixtures.h"

#include "argmin/detail/xoshiro256.h"

#include "argmin/detail/alternative/cmaes_sampling_marsaglia.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/test_functions/rosenbrock.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using namespace argmin::float_gate;
using namespace argmin::float_fixtures;

namespace
{

constexpr std::uint32_t max_iterations = 2000;
constexpr std::uint64_t seed = 42;

// CMA-ES reports ftol_reached on its own function-value stops; converged is a harness criterion.
constexpr std::array accepted_statuses{solver_status::ftol_reached, solver_status::converged};

bool accepted(solver_status status)
{
    return std::ranges::find(accepted_statuses, status) != accepted_statuses.end();
}

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    Policy policy{};
    policy.options.seed = seed;
    step_budget_solver solver{policy, problem, problem.initial_point(), opts};
    return solver.solve();
}

template <template <typename, int> typename Policy, typename ProblemD, typename ProblemF>
void check_tracks_double(const ProblemD& problem_d, const ProblemF& problem_f)
{
    const auto result_d = drive<Policy<double, ProblemD::problem_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, ProblemF::problem_dimension>>(problem_f);

    REQUIRE(accepted(result_d.status));
    REQUIRE(accepted(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_stochastic));
    CHECK(objective_tracks(result_f.objective_value, result_d.objective_value, k_stochastic));
    if constexpr(requires { problem_d.upper_bounds(); })
    {
        CHECK(std::abs(result_d.x[0] - problem_d.upper_bounds()[0]) <= tolerance(k_stochastic));
        CHECK(std::abs(static_cast<double>(result_f.x[0] - problem_f.upper_bounds()[0])) <= tolerance(k_stochastic));
    }
}

template <template <typename, int> typename Policy>
void check_rosenbrock()
{
    check_tracks_double<Policy>(rosenbrock<double>{}, rosenbrock<float>{});
}

template <template <typename, int> typename Policy>
void check_active_box_rosenbrock()
{
    check_tracks_double<Policy>(active_box_rosenbrock<double>{}, active_box_rosenbrock<float>{});
}

template <scalar S, int N>
using cmaes = cmaes_policy<S, N>;

template <scalar S, int N>
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<S, N>;

template <scalar S, int N>
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<S, N>;

}

TEST_CASE("the float uniform draw is the double draw at float precision", "[float]")
{
    detail::xoshiro256 rng_d{seed};
    detail::xoshiro256 rng_f{seed};
    for(int i = 0; i < 10000; ++i)
    {
        const double u_d = detail::alternative::marsaglia::canonical_uniform_pm1<double>(rng_d);
        const float u_f = detail::alternative::marsaglia::canonical_uniform_pm1<float>(rng_f);
        REQUIRE(u_f > -1.0f);
        REQUIRE(u_f < 1.0f);
        REQUIRE(u_f != 0.0f);
        REQUIRE(std::abs(static_cast<double>(u_f) - u_d) <= 0x1.0p-22);
    }
}

TEST_CASE("CMA-ES in float tracks double on Rosenbrock at its own defaults", "[float]")
{
    SECTION("cmaes_policy rosenbrock")
    {
        check_rosenbrock<cmaes>();
    }

    SECTION("cmaes_policy active_box_rosenbrock")
    {
        check_active_box_rosenbrock<cmaes>();
    }

    SECTION("repair_l2_penalty_policy rosenbrock")
    {
        check_rosenbrock<repair_l2>();
    }

    SECTION("repair_l2_penalty_policy active_box_rosenbrock")
    {
        check_active_box_rosenbrock<repair_l2>();
    }

    SECTION("no_repair_adaptive_penalty_policy rosenbrock")
    {
        check_rosenbrock<no_repair>();
    }

    SECTION("no_repair_adaptive_penalty_policy active_box_rosenbrock")
    {
        check_active_box_rosenbrock<no_repair>();
    }
}
