#include "property_findings.h"

#include "cases/case_options.h"
#include "cases/linear_quadratic_problems.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

namespace
{

template <typename Policy>
void check_inside_box(const Policy& policy, const auto& problem, const solver_options<>& options)
{
    const oracle::verdict<double> v = defects::judge_finding(policy, problem, options);
    CHECK(v.legs.bound_violation == 0.0);
    CHECK(v.failing != "finite");
}

template <typename Policy>
void check_box_binding_claim(const Policy& policy)
{
    const oracle::verdict<double> v = defects::judge_finding(policy, defects::box_binding_least_squares(), cases::least_squares_options());
    CAPTURE(static_cast<int>(v.claimed), v.failing, v.legs.stationarity);
    CHECK(v.ok);
}

template <typename Policy>
void check_infeasible_start_claim(const Policy& policy)
{
    const oracle::verdict<double> v = defects::judge_finding(policy, defects::infeasible_start_quadratic(), cases::moving_asymptote_options());
    CAPTURE(static_cast<int>(v.claimed), v.failing, v.legs.stationarity, v.legs.inequality_violation);
    CHECK(v.ok);
}

void check_augmented_lagrangian_claim(const cases::linearly_constrained_quadratic& problem)
{
    const oracle::verdict<double> v = defects::judge_finding(augmented_lagrangian_policy<lbfgsb_policy<>>{}, problem, cases::gradient_options());
    CAPTURE(static_cast<int>(v.claimed), v.failing, v.legs.stationarity);
    CHECK(v.ok);
}

} // namespace

TEST_CASE("the projected Gauss-Newton solvers stay inside the box on a generated box-binding least-squares case", "[least_squares]")
{
    check_inside_box(projected_gn_policy<>{}, defects::box_binding_least_squares(), cases::least_squares_options());
    check_inside_box(projected_gradient_gn_policy<>{}, defects::box_binding_least_squares(), cases::least_squares_options());
}

TEST_CASE("projected_gn makes a claim the oracle accepts on a generated box-binding least-squares case", "[known-defect][!shouldfail]")
{
    check_box_binding_claim(projected_gn_policy<>{});
}

TEST_CASE("projected_gradient_gn makes a claim the oracle accepts on a generated box-binding least-squares case", "[known-defect][!shouldfail]")
{
    check_box_binding_claim(projected_gradient_gn_policy<>{});
}

TEST_CASE("the generated infeasible-start quadratic violates both rows at its start", "[moving_asymptote]")
{
    const cases::linearly_constrained_quadratic p = defects::infeasible_start_quadratic();
    CHECK(((p.G * p.start - p.h).array() < 0.0).all());
    check_inside_box(mma_policy<>{}, p, cases::moving_asymptote_options());
}

TEST_CASE("mma makes a claim the oracle accepts from a generated infeasible start", "[known-defect][!shouldfail]")
{
    check_infeasible_start_claim(mma_policy<>{});
}

TEST_CASE("gcmma move_limit_shrink makes a claim the oracle accepts from a generated infeasible start", "[known-defect][!shouldfail]")
{
    check_infeasible_start_claim(alternative::gcmma::move_limit_shrink_policy<>{});
}

TEST_CASE("gcmma raa_augmented makes a claim the oracle accepts from a generated infeasible start", "[known-defect][!shouldfail]")
{
    check_infeasible_start_claim(alternative::gcmma::raa_augmented_policy<>{});
}

TEST_CASE("gcmma rho_wval makes a claim the oracle accepts from a generated infeasible start", "[known-defect][!shouldfail]")
{
    check_infeasible_start_claim(alternative::gcmma::rho_wval_policy<>{});
}

TEST_CASE("gcmma move_limit_shrink stays inside the box on a generated two-row quadratic", "[moving_asymptote]")
{
    check_inside_box(alternative::gcmma::move_limit_shrink_policy<>{}, defects::two_row_quadratic(), cases::moving_asymptote_options());
}

TEST_CASE("gcmma move_limit_shrink makes a claim the oracle accepts on a generated two-row quadratic", "[known-defect][!shouldfail]")
{
    const oracle::verdict<double> v = defects::judge_finding(alternative::gcmma::move_limit_shrink_policy<>{}, defects::two_row_quadratic(), cases::moving_asymptote_options());
    CAPTURE(static_cast<int>(v.claimed), v.failing, v.legs.stationarity);
    CHECK(v.ok);
}

TEST_CASE("augmented_lagrangian_lbfgsb makes a claim the oracle accepts on a generated one-equality quadratic", "[known-defect][!shouldfail]")
{
    check_augmented_lagrangian_claim(defects::one_equality_quadratic());
}

TEST_CASE("augmented_lagrangian_lbfgsb makes a claim the oracle accepts on a generated mixed-row quadratic", "[known-defect][!shouldfail]")
{
    check_augmented_lagrangian_claim(defects::mixed_row_quadratic());
}
