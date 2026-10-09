#include "float_gate.h"
#include "float_fixtures.h"
#include "sqp_float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <algorithm>
#include <type_traits>

using namespace argmin;
using namespace argmin::float_gate;
using namespace argmin::float_fixtures;

namespace
{

template <scalar S>
using bobyqa = bobyqa_policy<S>;

constexpr std::uint32_t max_iterations = 1000;

constexpr std::array bobyqa_accepted_statuses{solver_status::converged, solver_status::ftol_reached};

// COBYLA reports xtol_reached when its trust radius contracts to the final radius.
constexpr std::array cobyla_accepted_statuses{solver_status::xtol_reached, solver_status::converged};

template <std::size_t K>
bool accepted(const std::array<solver_status, K>& statuses, solver_status status)
{
    return std::ranges::find(statuses, status) != statuses.end();
}

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

// With the default criteria's thresholds at zero only the policy's own trust-radius stop ends the solve.
template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive_to_own_stop(const Problem& problem)
{
    using S = typename Policy::scalar_type;
    solver_options<default_convergence<S>> opts{};
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(S(0));
    opts.set_objective_threshold(S(0));
    opts.set_step_threshold(S(0));
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

template <template <scalar> typename Problem>
void check_cobyla_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<cobyla_policy<double>>(problem_d);
    const auto result_f = drive<cobyla_policy<float>>(problem_f);

    REQUIRE(accepted(cobyla_accepted_statuses, result_d.status));
    REQUIRE(accepted(cobyla_accepted_statuses, result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_derivative_free));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
}

template <template <scalar> typename Problem>
void check_bobyqa_experimental()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<bobyqa<double>>(problem_d);
    const auto result_f = drive<bobyqa<float>>(problem_f);
    const auto own_stop_f = drive_to_own_stop<bobyqa<float>>(problem_f);

    REQUIRE(accepted(bobyqa_accepted_statuses, result_d.status));
    REQUIRE(result_f.x.allFinite());
    CHECK_NOFAIL(accepted(bobyqa_accepted_statuses, result_f.status));
    CHECK_NOFAIL(tracks(result_f.x, result_d.x, k_derivative_free));
    REQUIRE(own_stop_f.status == solver_status::converged);
    CHECK(tracks(own_stop_f.x, result_d.x, k_derivative_free));
    if constexpr(std::is_same_v<Problem<float>, active_box_rosenbrock<float>>)
    {
        CHECK(result_d.x[0] == problem_d.upper_bounds()[0]);
        CHECK(own_stop_f.x[0] == problem_f.upper_bounds()[0]);
    }
}

}

TEST_CASE("derivative-free policies in float track double on bounded and constrained fixtures at their own defaults", "[float]")
{
    SECTION("cobyla_policy hs024")
    {
        check_cobyla_tracks_double<hs024>();
    }

    SECTION("cobyla_policy hs035")
    {
        check_cobyla_tracks_double<hs035>();
    }
}

// Under the default criteria the float solve ends early on the step-tolerance or gradient stop: once the
// trust radius reaches its final value, and after a geometry step that reports the length of the
// preceding short step, BOBYQA's reported step and model gradient fall below the float thresholds.
TEST_CASE("BOBYQA in float on bounded fixtures where the default criteria stop it before its own trust-radius stop", "[float]")
{
    SECTION("active_box_rosenbrock")
    {
        check_bobyqa_experimental<active_box_rosenbrock>();
    }

    SECTION("hs038")
    {
        check_bobyqa_experimental<hs038>();
    }
}
