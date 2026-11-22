#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_RESET_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_RESET_SUPPORT_H

#include "same_bits.h"
#include "generated_values.h"
#include "property_support.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/detail/evaluation_count.h"

#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <rapidcheck.h>

#include <cmath>
#include <string>
#include <cstdint>
#include <utility>

namespace argmin::property
{

// A second start inside the problem's box, or anywhere in [-1, 1]^n when it has none.
template <typename Problem>
Eigen::VectorXd pick_other_start(const Problem& problem)
{
    Eigen::VectorXd x = pick_vector(problem.dimension());
    if constexpr(requires { problem.lower_bounds(); })
    {
        const Eigen::VectorXd lower = problem.lower_bounds();
        const Eigen::VectorXd width = problem.upper_bounds() - lower;
        for(Eigen::Index i = 0; i < x.size(); ++i)
            x[i] = std::isfinite(width[i]) ? lower[i] + 0.5 * (x[i] + 1.0) * width[i] : x[i];
    }
    return x;
}

enum class reset_kind : std::uint8_t
{
    clearing,
    warm
};

template <typename Policy, typename Problem>
void check_reset_equals_fresh(const Policy& policy, const Problem& problem, const solver_options<>& options, reset_kind kind)
{
    const Eigen::VectorXd other = pick_other_start(problem);
    step_budget_solver used{policy, problem, problem.initial_point(), options};
    used.solve(options);
    if(kind == reset_kind::clearing)
        used.reset_clear(other);
    else
        used.reset(other);
    const solve_result<double> after_reset = used.solve(options);
    step_budget_solver fresh{policy, problem, other, options};
    const solve_result<double> fresh_result = fresh.solve(options);
    RC_LOG() << "reset status " << static_cast<std::int32_t>(after_reset.status) << " after " << after_reset.iterations << ", fresh status " << static_cast<std::int32_t>(fresh_result.status) << " after "
             << fresh_result.iterations << '\n';
    RC_ASSERT(same_bits(after_reset, fresh_result));
}

template <typename Generator, typename Policy>
void check_reset(const std::string& key, const Generator& generator, const Policy& policy, const solver_options<>& options, reset_kind kind = reset_kind::clearing)
{
    const std::string description = key + (kind == reset_kind::clearing ? " reset_clear equals fresh" : " reset equals fresh");
    check(description, [&] { check_reset_equals_fresh(policy, *generator, options, kind); });
}

template <typename Generator, typename Policy>
void check_second_solve(const std::string& key, const Generator& generator, const Policy& policy, const solver_options<>& base)
{
    check(key + " second solve on one options object equals the first",
          [&]
          {
              const auto problem = *generator;
              solver_options<> options = base;
              step_budget_solver first{policy, problem, problem.initial_point(), options};
              const solve_result<double> first_result = first.solve(options);
              step_budget_solver second{policy, problem, problem.initial_point(), options};
              RC_ASSERT(same_bits(first_result, second.solve(options)));
          });
}

template <typename Policy, typename Problem>
using state_of = decltype(std::declval<Policy&>().init(std::declval<const Problem&>(), std::declval<const Eigen::VectorXd&>(), std::declval<const solver_options<>&>()));

template <typename Policy, typename State>
step_result<double> counted_step(Policy& policy, State& s)
{
    const std::uint32_t before = detail::evaluations_of(s);
    step_result<double> result = policy.step(s);
    detail::settle_step_evaluations(result, s, before);
    return result;
}

inline bool ends_run(const step_result<double>& result)
{
    return result.policy_status.has_value() && *result.policy_status != solver_status::running;
}

template <typename Policy, typename Problem>
state_of<Policy, Problem> stepped_then_reset(Policy& policy, const Problem& problem, const solver_options<>& options, std::int32_t steps, const Eigen::VectorXd& other)
{
    state_of<Policy, Problem> s = policy.init(problem, problem.initial_point(), options);
    for(std::int32_t k = 0; k < steps; ++k)
    {
        if(ends_run(counted_step(policy, s)))
            break;
    }
    policy.reset(s, other);
    return s;
}

template <typename Policy, typename State>
bool step_alike(Policy& used_policy, State& used, Policy& fresh_policy, State& fresh, std::int32_t steps)
{
    for(std::int32_t k = 0; k < steps; ++k)
    {
        const step_result<double> a = counted_step(used_policy, used);
        const step_result<double> b = counted_step(fresh_policy, fresh);
        const char* diverged = !same_bits(a, b) ? "step result" : !same_bits(used.x, fresh.x) ? "x" : counters_of(used) != counters_of(fresh) ? "counters" : nullptr;
        if(diverged != nullptr)
        {
            RC_LOG() << diverged << " diverged on step " << k + 1 << '\n';
            return false;
        }
        if(ends_run(a))
            break;
    }
    return true;
}

// Rebuilding the reset state a second time from the same steps gives a donor whose warm members can be moved, so a
// member that cannot be copied (an inner solver) still transplants.
template <typename Policy, typename Problem, typename Transplant>
void check_warm_reset_equals_fresh(const Policy& policy, const Problem& problem, const solver_options<>& options, const Transplant& transplant, std::int32_t steps)
{
    const Eigen::VectorXd other = pick_other_start(problem);
    const std::int32_t before = *rc::gen::inRange<std::int32_t>(0, steps + 1);
    Policy used_policy = policy;
    Policy donor_policy = policy;
    Policy fresh_policy = policy;
    state_of<Policy, Problem> used = stepped_then_reset(used_policy, problem, options, before, other);
    state_of<Policy, Problem> donor = stepped_then_reset(donor_policy, problem, options, before, other);
    state_of<Policy, Problem> fresh = fresh_policy.init(problem, other, options);
    transplant(donor, fresh);
    RC_LOG() << "reset after " << before << " steps\n";
    RC_ASSERT(step_alike(used_policy, used, fresh_policy, fresh, steps));
}

template <typename Generator, typename Policy, typename Transplant>
void check_warm_reset(const std::string& key, const Generator& generator, const Policy& policy, const solver_options<>& options, const Transplant& transplant, std::int32_t steps = 60)
{
    check(key + " reset equals fresh with its warm set carried over", [&] { check_warm_reset_equals_fresh(policy, *generator, options, transplant, steps); });
}

}

#endif
