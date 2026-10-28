#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_DERIVATIVE_FREE_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_DERIVATIVE_FREE_CASES_H

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <array>
#include <string_view>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 2> derivative_free_keys{"bobyqa", "cobyla"};

// BOBYQA takes bounds only. COBYLA requires the constrained interface and splits each equality row into
// two inequalities, so it takes every constraint class and its bound class has zero constraint rows.
template <typename Visitor>
void visit_derivative_free_cases(Visitor& visitor)
{
    const hs038<> bound;
    const no_constraint_rows<hs038<>> rowless_bound;
    const hs076<> inequality;
    const hs028<> equality;
    const hs071<> mixed;
    const solver_options<> options = derivative_free_options();
    visit_case(visitor, case_id{"bobyqa", "bound", "hs038"}, bobyqa_policy<>{}, bound, options);
    visit_case(visitor, case_id{"cobyla", "bound", "hs038"}, cobyla_policy<>{}, rowless_bound, options);
    visit_case(visitor, case_id{"cobyla", "inequality", "hs076"}, cobyla_policy<>{}, inequality, options);
    visit_case(visitor, case_id{"cobyla", "equality", "hs028"}, cobyla_policy<>{}, equality, options);
    visit_case(visitor, case_id{"cobyla", "mixed", "hs071"}, cobyla_policy<>{}, mixed, options);
}

}

#endif
