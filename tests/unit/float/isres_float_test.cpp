#include "float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"

#include "argmin/options/convergence_defaults.h"

#include "argmin/test_functions/hock_schittkowski.h"

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

namespace
{

constexpr std::uint32_t max_iterations = 2000;
constexpr std::uint64_t seed = 42;

// ISRES reports xtol_reached on its own sigma-collapse stop; stalled is the harness step criterion,
// firing once the surviving population's mean step size falls below its threshold.
constexpr std::array accepted_statuses{solver_status::xtol_reached, solver_status::stalled};

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

// The largest row violation, evaluated from the problem rather than read from the solver's report.
template <typename Problem>
float row_violation(const Problem& problem, const vector<float, Problem::problem_dimension>& x)
{
    const int n_eq = problem.num_equality();
    vector<float, dynamic_dimension> c(n_eq + problem.num_inequality());
    problem.constraints(x, c);
    float largest = 0.0f;
    for(int i = 0; i < static_cast<int>(c.size()); ++i)
        largest = std::max(largest, i < n_eq ? std::abs(c[i]) : -c[i]);
    return largest;
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
    CHECK(row_violation(problem_f, result_f.x) <= convergence_defaults<float>::feasibility_tolerance);
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
    SECTION("runarsson_yao_paper_policy hs023")
    {
        check_tracks_double<runarsson_yao, hs023>();
    }

    SECTION("original_argmin_policy hs023")
    {
        check_tracks_double<original_argmin, hs023>();
    }

    SECTION("runarsson_yao_paper_policy hs024")
    {
        check_tracks_double<runarsson_yao, hs024>();
    }

    SECTION("original_argmin_policy hs024")
    {
        check_tracks_double<original_argmin, hs024>();
    }
}

TEST_CASE("ISRES nlopt_faithful in float tracks double on an inequality-constrained fixture at its own defaults", "[float]")
{
    SECTION("isres_policy hs023")
    {
        check_tracks_double<nlopt_faithful, hs023>();
    }

    SECTION("isres_policy hs024")
    {
        check_tracks_double<nlopt_faithful, hs024>();
    }
}
