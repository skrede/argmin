#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_GRADIENT_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_GRADIENT_CASES_H

#include "cases/case_options.h"

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <string_view>

namespace argmin::cases
{

template <typename Visitor, typename Policy>
void visit_unconstrained_and_bound(Visitor& visitor, std::string_view key, const Policy& policy)
{
    const rosenbrock<double, 2> unconstrained;
    const hs038<> bound;
    visit_case(visitor, case_id{key, "unconstrained", "rosenbrock"}, policy, unconstrained, gradient_options());
    visit_case(visitor, case_id{key, "bound", "hs038"}, policy, bound, gradient_options());
}

// The augmented Lagrangian requires constraint rows, so a bound-only problem is not one of its cases.
template <typename Visitor>
void visit_augmented_lagrangian(Visitor& visitor)
{
    const std::string_view key = "augmented_lagrangian_lbfgsb";
    const augmented_lagrangian_policy<lbfgsb_policy<>> policy{};
    const hs076<> inequality;
    const hs028<> equality;
    const hs071<> mixed;
    visit_case(visitor, case_id{key, "inequality", "hs076"}, policy, inequality, gradient_options());
    visit_case(visitor, case_id{key, "equality", "hs028"}, policy, equality, gradient_options());
    visit_case(visitor, case_id{key, "mixed", "hs071"}, policy, mixed, gradient_options());
}

template <typename Visitor>
void visit_gradient_cases(Visitor& visitor)
{
    visit_unconstrained_and_bound(visitor, "lbfgsb", lbfgsb_policy<>{});
    visit_unconstrained_and_bound(visitor, "byrd_lbfgsb", byrd_lbfgsb_policy<>{});
    visit_augmented_lagrangian(visitor);
    visit_unconstrained_and_bound(visitor, "multistart_lbfgsb", multistart_policy<lbfgsb_policy<>>{});
}

}

#endif
