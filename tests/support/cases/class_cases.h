#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CLASS_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CLASS_CASES_H

#include "cases/case_options.h"
#include "cases/case_problems.h"

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

template <typename Inequality = hs076<>, typename Visitor, typename Policy>
void visit_constraint_rows(Visitor& visitor, std::string_view key, const Policy& policy, std::string_view inequality_name = "hs076")
{
    const Inequality inequality;
    const hs028<> equality;
    const hs071<> mixed;
    visit_case(visitor, case_id{key, "inequality", inequality_name}, policy, inequality, gradient_options());
    visit_case(visitor, case_id{key, "equality", "hs028"}, policy, equality, gradient_options());
    visit_case(visitor, case_id{key, "mixed", "hs071"}, policy, mixed, gradient_options());
}

template <typename Inequality = hs076<>, typename Visitor, typename Policy>
void visit_bound_and_constraint_rows(Visitor& visitor, std::string_view key, const Policy& policy, std::string_view inequality_name = "hs076")
{
    const no_constraint_rows<hs038<>> bound;
    visit_case(visitor, case_id{key, "bound", "hs038"}, policy, bound, gradient_options());
    visit_constraint_rows<Inequality>(visitor, key, policy, inequality_name);
}

template <typename Visitor, typename Policy>
void visit_every_class(Visitor& visitor, std::string_view key, const Policy& policy)
{
    visit_unconstrained_and_bound(visitor, key, policy);
    visit_constraint_rows(visitor, key, policy);
}

}

#endif
