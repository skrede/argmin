#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_WARM_SETS_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_WARM_SETS_H

#include <utility>

// What each policy's reset keeps by contract, as a move from a reset state into a freshly built one at the same start.
// Any member a reset keeps that is not listed here makes the two runs diverge.
namespace argmin::property::warm
{

inline constexpr auto none = [](auto&, auto&) {};

inline constexpr auto curvature_pairs = [](auto& reset, auto& fresh) { fresh.B = std::move(reset.B); };

inline constexpr auto hessian = [](auto& reset, auto& fresh) { fresh.hessian = std::move(reset.hessian); };

inline constexpr auto hessian_and_penalty = [](auto& reset, auto& fresh)
{
    hessian(reset, fresh);
    fresh.sigma = reset.sigma;
};

inline constexpr auto hessian_and_multipliers = [](auto& reset, auto& fresh)
{
    hessian(reset, fresh);
    fresh.lambda = std::move(reset.lambda);
};

inline constexpr auto trust_radii = [](auto& reset, auto& fresh) { fresh.sigma = std::move(reset.sigma); };

// The inner context moves whole: the inner solver addresses the subproblem stored beside it.
inline constexpr auto multipliers_penalty_and_inner_solver = [](auto& reset, auto& fresh)
{
    fresh.lambda_eq = std::move(reset.lambda_eq);
    fresh.lambda_ineq = std::move(reset.lambda_ineq);
    fresh.mu = reset.mu;
    fresh.prev_viol = reset.prev_viol;
    fresh.mu_at_last_inner_solve = reset.mu_at_last_inner_solve;
    std::swap(fresh.ctx, reset.ctx);
};

template <typename Transplant>
constexpr auto of_inner(Transplant transplant)
{
    return [transplant](auto& reset, auto& fresh) { transplant(reset.inner, fresh.inner); };
}

}

#endif
