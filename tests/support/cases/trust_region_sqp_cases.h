#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_TRUST_REGION_SQP_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_TRUST_REGION_SQP_CASES_H

#include "cases/class_cases.h"

#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/types.h"

namespace argmin::cases
{

template <sqp_mode Mode>
using tr_sqp = tr_sqp_policy<double, dynamic_dimension, Mode>;

template <sqp_mode Mode>
using filter_trsqp = filter_trsqp_policy<double, dynamic_dimension, Mode>;

// At the hs076 optimum accurate tr_sqp ends stalled under g++ Debug but trust_region_step_rejected
// under clang -march=native, so its inequality case is hs024, whose verdict both builds share.
template <typename Visitor>
void visit_trust_region_sqp_cases(Visitor& visitor)
{
    visit_bound_and_constraint_rows(visitor, "tr_sqp_fast", tr_sqp<sqp_mode::fast>{});
    visit_bound_and_constraint_rows<hs024<>>(visitor, "tr_sqp_accurate", tr_sqp<sqp_mode::accurate>{}, "hs024");
    visit_bound_and_constraint_rows(visitor, "filter_trsqp_fast", filter_trsqp<sqp_mode::fast>{});
    visit_bound_and_constraint_rows(visitor, "filter_trsqp_accurate", filter_trsqp<sqp_mode::accurate>{});
}

}

#endif
