#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_SQP_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_SQP_CASES_H

#include "cases/class_cases.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <array>
#include <string_view>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 4> sqp_keys{"kraft_slsqp", "nw_sqp", "filter_slsqp", "filter_nw_sqp"};

// At the hs076 optimum the Newton-system policies end stalled under g++ Debug but run out of iterations
// under clang -march=native, and kraft_slsqp does the same under g++ -march=native, so their inequality
// case is hs024, whose verdict every build shares.
template <typename Visitor>
void visit_sqp_cases(Visitor& visitor)
{
    visit_unconstrained_and_bound(visitor, "kraft_slsqp", kraft_slsqp_policy<>{});
    visit_constraint_rows<hs024<>>(visitor, "kraft_slsqp", kraft_slsqp_policy<>{}, "hs024");
    visit_bound_and_constraint_rows<hs024<>>(visitor, "nw_sqp", nw_sqp_policy<>{}, "hs024");
    visit_every_class(visitor, "filter_slsqp", filter_slsqp_policy<>{});
    visit_bound_and_constraint_rows<hs024<>>(visitor, "filter_nw_sqp", filter_nw_sqp_policy<>{}, "hs024");
}

}

#endif
