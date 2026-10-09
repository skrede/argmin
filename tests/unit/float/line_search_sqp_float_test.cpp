#include "float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/small_dense.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

constexpr std::uint32_t max_iterations = 200;

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = max_iterations;
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

// Under the default criteria a stalled exit is the step-tolerance stop, which only fires within the
// criterion's feasibility tolerance. On hs071 the double run ends there too: its reported KKT residual
// stays near 1 at the solution, where the bound x0 >= 1 is active.
bool stopped_by_criterion(solver_status status)
{
    return status == solver_status::converged || status == solver_status::stalled;
}

// Evaluated from the problem rather than read from the solver's report, so a policy that drops the
// float problem's constraints cannot certify its own feasibility.
template <typename Problem>
float max_violation(const Problem& problem, const vector<float, Problem::problem_dimension>& x)
{
    const int n_eq = problem.num_equality();
    vector<float, dynamic_dimension> c(n_eq + problem.num_inequality());
    problem.constraints(x, c);
    float violation = 0.0f;
    for(int i = 0; i < static_cast<int>(c.size()); ++i)
        violation = std::max(violation, i < n_eq ? std::abs(c[i]) : -c[i]);
    return violation;
}

template <template <scalar, int> typename Policy, template <typename> typename Problem>
void check_tracks_double()
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive<Policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(stopped_by_criterion(result_d.status));
    REQUIRE(stopped_by_criterion(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k_line_search_sqp));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
}

template <template <scalar, int> typename Policy>
void check_family()
{
    SECTION("hs071")
    {
        check_tracks_double<Policy, hs071>();
    }

    SECTION("hs035")
    {
        check_tracks_double<Policy, hs035>();
    }

    SECTION("hs028")
    {
        check_tracks_double<Policy, hs028>();
    }

    SECTION("sd012")
    {
        check_tracks_double<Policy, sd012>();
    }
}

template <template <scalar, int> typename Policy>
void check_filter_family()
{
    SECTION("hs071")
    {
        check_tracks_double<Policy, hs071>();
    }

    SECTION("hs024")
    {
        check_tracks_double<Policy, hs024>();
    }

    SECTION("hs043")
    {
        check_tracks_double<Policy, hs043>();
    }

    SECTION("sd024")
    {
        check_tracks_double<Policy, sd024>();
    }
}

}

TEST_CASE("nw_sqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_family<nw_sqp_policy>();
}

TEST_CASE("kraft_slsqp in float tracks double on constrained Hock-Schittkowski fixtures at its own defaults", "[float]")
{
    check_family<kraft_slsqp_policy>();
}

TEST_CASE("filter SQP policies in float track double on constrained Hock-Schittkowski fixtures at their own defaults", "[float]")
{
    SECTION("filter_slsqp")
    {
        check_filter_family<filter_slsqp_policy>();
    }

    SECTION("filter_nw_sqp")
    {
        check_filter_family<filter_nw_sqp_policy>();
    }
}
