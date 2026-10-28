#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_MOVING_ASYMPTOTE_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_MOVING_ASYMPTOTE_CASES_H

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "argmin/solver/mma_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 5> moving_asymptote_keys{"mma", "ccsa_quadratic", "gcmma_move_limit_shrink", "gcmma_raa_augmented", "gcmma_rho_wval"};

template <typename Visitor, typename Policy>
void visit_bound_and_inequality(Visitor& visitor, std::string_view key, const Policy& policy)
{
    const no_constraint_rows<hs038<>> bound;
    const hs076<> inequality;
    visit_case(visitor, case_id{key, "bound", "hs038"}, policy, bound, moving_asymptote_options());
    visit_case(visitor, case_id{key, "inequality", "hs076"}, policy, inequality, moving_asymptote_options());
}

// Converting a dynamic-dimension policy of this family to a fixed-dimension problem does not compile, so
// each policy is spelled at the dimension both fixtures share. The family rejects equality rows by
// design, so the equality and mixed classes are not its cases.
template <typename Visitor>
void visit_moving_asymptote_cases(Visitor& visitor)
{
    constexpr std::int32_t n = hs076<>::problem_dimension;
    static_assert(hs038<>::problem_dimension == n);
    visit_bound_and_inequality(visitor, "mma", mma_policy<double, n>{});
    visit_bound_and_inequality(visitor, "ccsa_quadratic", ccsa_quadratic_policy<double, n>{});
    visit_bound_and_inequality(visitor, "gcmma_move_limit_shrink", alternative::gcmma::move_limit_shrink_policy<double, n>{});
    visit_bound_and_inequality(visitor, "gcmma_raa_augmented", alternative::gcmma::raa_augmented_policy<double, n>{});
    visit_bound_and_inequality(visitor, "gcmma_rho_wval", alternative::gcmma::rho_wval_policy<double, n>{});
}

}

#endif
