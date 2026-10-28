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

// The moving-asymptote family crawls past the saddle of the Wood function (hs038) for 8000 to 19000
// iterations before it converges, so its budget leaves room for that.
inline solver_options<> moving_asymptote_options()
{
    solver_options<> opts = gradient_options();
    opts.max_iterations = 30000;
    return opts;
}

// A derivative-free iteration spends one or a few evaluations, so these families need a budget sized in
// evaluations: BOBYQA takes about 500 on the Wood function (hs038).
inline solver_options<> derivative_free_options()
{
    solver_options<> opts = gradient_options();
    opts.max_iterations = 2000;
    return opts;
}

// The evolution strategies count a generation as an iteration; ISRES is specified with about 1750
// generations (Runarsson and Yao 2000).
inline solver_options<> stochastic_options()
{
    solver_options<> opts = gradient_options();
    opts.max_iterations = 2000;
    return opts;
}

// The visitor runs inside this call, so the problem outlives every solver make_solver returns. A visitor
// may pass its own start point, which the trajectory golden's perturbation controls use.
template <typename Visitor, typename Policy, typename Problem>
void visit_case(Visitor& visitor, const case_id& id, const Policy& policy, const Problem& problem, const solver_options<>& options)
{
    const auto make_solver = [&](const auto&... x0)
    {
        if constexpr(sizeof...(x0) == 0)
            return step_budget_solver{policy, problem, problem.initial_point(), options};
        else
            return step_budget_solver{policy, problem, x0..., options};
    };
    visitor(id, problem, options, make_solver);
}

}

#endif
