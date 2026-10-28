#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_SQP_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_SQP_CASES_H

#include "cases/class_cases.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

namespace argmin::cases
{

// At the hs076 optimum the Newton-system policies end stalled under g++ Debug but run out of iterations
// under clang -march=native, so their inequality case is hs024, whose verdict both builds share.
template <typename Visitor>
void visit_sqp_cases(Visitor& visitor)
{
    visit_every_class(visitor, "kraft_slsqp", kraft_slsqp_policy<>{});
    visit_bound_and_constraint_rows<hs024<>>(visitor, "nw_sqp", nw_sqp_policy<>{}, "hs024");
    visit_every_class(visitor, "filter_slsqp", filter_slsqp_policy<>{});
    visit_bound_and_constraint_rows<hs024<>>(visitor, "filter_nw_sqp", filter_nw_sqp_policy<>{}, "hs024");
}

}

#endif
