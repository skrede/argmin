#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_OPTIONS_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_OPTIONS_H

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include <cstdint>
#include <string_view>

namespace argmin::cases
{

struct case_id
{
    std::string_view key;
    std::string_view problem_class;
    std::string_view problem;
};

inline constexpr std::uint64_t stochastic_seed = 20261028;

inline solver_options<> gradient_options()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    return opts;
}

inline solver_options<> least_squares_options()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);
    return opts;
}

// The visitor runs inside this call, so the problem outlives every solver make_solver returns.
template <typename Visitor, typename Policy, typename Problem>
void visit_case(Visitor& visitor, const case_id& id, const Policy& policy, const Problem& problem, const solver_options<>& options)
{
    const auto make_solver = [&]() { return step_budget_solver{policy, problem, problem.initial_point(), options}; };
    visitor(id, problem, options, make_solver);
}

}

#endif
