#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_GRADIENT_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_GRADIENT_CASES_H

#include "cases/class_cases.h"

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 4> gradient_keys{"lbfgsb", "byrd_lbfgsb", "augmented_lagrangian_lbfgsb", "multistart_lbfgsb"};

// The default multistart runs 1 + default_max_restarts starts, each given the budget of a single L-BFGS-B case.
inline solver_options<> multistart_options()
{
    solver_options<> opts = gradient_options();
    opts.max_iterations *= std::uint32_t{1} + multistart_policy<lbfgsb_policy<>>::default_max_restarts;
    return opts;
}

template <typename Visitor>
void visit_gradient_cases(Visitor& visitor)
{
    visit_unconstrained_and_bound(visitor, "lbfgsb", lbfgsb_policy<>{});
    visit_unconstrained_and_bound(visitor, "byrd_lbfgsb", byrd_lbfgsb_policy<>{});
    // The augmented Lagrangian requires constraint rows, so a bound-only problem is not one of its cases.
    visit_constraint_rows(visitor, "augmented_lagrangian_lbfgsb", augmented_lagrangian_policy<lbfgsb_policy<>>{});
    visit_unconstrained_and_bound(visitor, "multistart_lbfgsb", multistart_policy<lbfgsb_policy<>>{}, multistart_options());
}

}

#endif
