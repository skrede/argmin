#include "float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/options/isres_defaults.h"

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

// ISRES reports ftol_reached on its own sigma-collapse stop; converged is a harness criterion.
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

// The feasibility gate bounds this sum of squared violations, evaluated from the problem rather than
// read from the solver's report.
template <typename Problem>
float squared_violation(const Problem& problem, const vector<float, Problem::problem_dimension>& x)
{
    const int n_eq = problem.num_equality();
    vector<float, dynamic_dimension> c(n_eq + problem.num_inequality());
    problem.constraints(x, c);
    float sum = 0.0f;
    for(int i = 0; i < static_cast<int>(c.size()); ++i)
    {
        const float v = i < n_eq ? c[i] : std::min(0.0f, c[i]);
        sum += v * v;
    }
    return sum;
}

template <template <typename, int> typename Policy, template <typename> typename Problem>
void check_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<Policy<double, Problem<double>::problem_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, Problem<float>::problem_dimension>>(problem_f);

    REQUIRE(accepted(result_d.status));
    REQUIRE(accepted(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_stochastic));
    CHECK(objective_tracks(result_f.objective_value, result_d.objective_value, k_stochastic));
    CHECK(squared_violation(problem_f, result_f.x) <= isres_defaults<float>::feasibility_gate);
}

// Both scalars stop after the first generation here: an infinite bound makes the bound-relative
// collapse threshold infinite, so the stop records where the first generation landed.
template <template <typename, int> typename Policy, template <typename> typename Problem>
void check_first_generation_stop()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<Policy<double, Problem<double>::problem_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, Problem<float>::problem_dimension>>(problem_f);

    REQUIRE(accepted(result_d.status));
    REQUIRE(result_f.x.allFinite());
    CHECK(accepted(result_f.status));
    CHECK(squared_violation(problem_f, result_f.x) <= isres_defaults<float>::feasibility_gate);
    CHECK_NOFAIL(tracks(result_f.x, result_d.x, k_stochastic));
}

template <scalar S, int N>
using nlopt_faithful = isres_policy<S, N>;

template <scalar S, int N>
using runarsson_yao = alternative::isres::runarsson_yao_paper_policy<S, N>;

template <scalar S, int N>
using original_argmin = alternative::isres::original_argmin_policy<S, N>;

}

TEST_CASE("ISRES in float tracks double on an inequality-constrained fixture at its own defaults", "[float]")
{
    SECTION("isres_policy hs023")
    {
        check_tracks_double<nlopt_faithful, hs023>();
    }

    SECTION("runarsson_yao_paper_policy hs023")
    {
        check_tracks_double<runarsson_yao, hs023>();
    }

    SECTION("original_argmin_policy hs023")
    {
        check_tracks_double<original_argmin, hs023>();
    }

    SECTION("original_argmin_policy hs024")
    {
        check_tracks_double<original_argmin, hs024>();
    }
}

TEST_CASE("ISRES in float on a half-bounded fixture where the collapse stop fires after one generation", "[float]")
{
    SECTION("isres_policy hs024")
    {
        check_first_generation_stop<nlopt_faithful, hs024>();
    }

    SECTION("runarsson_yao_paper_policy hs024")
    {
        check_first_generation_stop<runarsson_yao, hs024>();
    }
}
