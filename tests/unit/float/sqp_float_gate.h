#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_SQP_FLOAT_GATE_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_SQP_FLOAT_GATE_H

#include "float_gate.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace argmin::float_gate
{

inline constexpr std::uint32_t sqp_max_iterations = 200;

template <typename Policy, typename Problem>
solve_result<typename Policy::scalar_type, Problem::problem_dimension> drive_sqp(const Problem& problem)
{
    solver_options<default_convergence<typename Policy::scalar_type>> opts{};
    opts.max_iterations = sqp_max_iterations;
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    return solver.solve();
}

// Under the default criteria a stalled exit is the step-tolerance stop, which only fires within the
// criterion's feasibility tolerance. On hs071 the double run ends there too: its reported KKT residual
// stays near 1 at the solution, where the bound x0 >= 1 is active.
inline bool stopped_by_criterion(solver_status status)
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
void check_sqp_tracks_double(double k)
{
    const Problem<double> problem_d;
    const Problem<float> problem_f;
    const auto result_d = drive_sqp<Policy<double, dynamic_dimension>>(problem_d);
    const auto result_f = drive_sqp<Policy<float, dynamic_dimension>>(problem_f);

    REQUIRE(stopped_by_criterion(result_d.status));
    REQUIRE(stopped_by_criterion(result_f.status));
    CHECK(tracks(result_f.x, result_d.x, k));
    CHECK(max_violation(problem_f, result_f.x) <= solver_options<default_convergence<float>>{}.feasibility_tolerance);
}

}

#endif
