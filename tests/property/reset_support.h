#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_RESET_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_RESET_SUPPORT_H

#include "generated_values.h"
#include "property_support.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <rapidcheck.h>

#include <bit>
#include <cmath>
#include <string>
#include <cstdint>

namespace argmin::property
{

inline bool same_bits(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

inline bool same_bits(const solve_result<double>& a, const solve_result<double>& b)
{
    if(a.status != b.status || a.iterations != b.iterations || a.function_evaluations != b.function_evaluations || a.x.size() != b.x.size())
        return false;
    if(!same_bits(a.objective_value, b.objective_value) || !same_bits(a.gradient_norm, b.gradient_norm) || !same_bits(a.constraint_violation, b.constraint_violation))
        return false;
    for(Eigen::Index i = 0; i < a.x.size(); ++i)
    {
        if(!same_bits(a.x[i], b.x[i]))
            return false;
    }
    return true;
}

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

}

#endif
