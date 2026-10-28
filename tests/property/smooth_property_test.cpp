#include "property_judge.h"
#include "generated_problems.h"

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace argmin;

namespace
{

template <typename Policy>
void check_unconstrained_and_bound(const std::string& key, const Policy& policy)
{
    property::check_judged(key + " unconstrained", property::unconstrained_quadratic(), policy, cases::gradient_options());
    property::check_judged(key + " bound", property::bound_quadratic(), policy, cases::gradient_options());
}

template <typename Policy>
void check_constraint_rows(const std::string& key, const Policy& policy, const solver_options<>& options)
{
    property::check_judged(key + " inequality", property::inequality_quadratic(), policy, options);
    property::check_judged(key + " equality", property::equality_quadratic(), policy, options);
    property::check_judged(key + " mixed", property::mixed_quadratic(), policy, options);
}

template <typename Policy>
void check_bound_and_constraint_rows(const std::string& key, const Policy& policy)
{
    property::check_judged(key + " bound", property::rowless_bound_quadratic(), policy, cases::gradient_options());
    check_constraint_rows(key, policy, cases::gradient_options());
}

// Excluded region for the inequality class: an infeasible start. From one, mma and the gcmma variants can
// stay at the start and claim converged or stalled there (tests/defects/property_findings_test.cpp).
template <typename Policy>
void check_bound_and_inequality(const std::string& key, const Policy& policy, oracle::claim inequality_cap = oracle::claim::optimality)
{
    const solver_options<> options = property::moving_asymptote_property_options();
    property::check_judged(key + " bound", property::rowless_bound_quadratic(), policy, options);
    property::check_judged(key + " inequality", property::inequality_quadratic(property::start_kind::feasible), policy, options, inequality_cap);
}

template <sqp_mode Mode>
using tr_sqp = tr_sqp_policy<double, dynamic_dimension, Mode>;

template <sqp_mode Mode>
using filter_trsqp = filter_trsqp_policy<double, dynamic_dimension, Mode>;

}

TEST_CASE("the oracle accepts the gradient bound solvers on generated quadratics", "[property]")
{
    check_unconstrained_and_bound("lbfgsb", lbfgsb_policy<>{});
    check_unconstrained_and_bound("byrd_lbfgsb", byrd_lbfgsb_policy<>{});
    check_unconstrained_and_bound("multistart_lbfgsb", multistart_policy<lbfgsb_policy<>>{});
}

TEST_CASE("the oracle accepts the augmented Lagrangian on generated quadratics", "[property]")
{
    check_constraint_rows("augmented_lagrangian_lbfgsb", augmented_lagrangian_policy<lbfgsb_policy<>>{}, cases::gradient_options());
}

TEST_CASE("the oracle accepts the line-search SQP solvers on generated quadratics", "[property]")
{
    check_unconstrained_and_bound("kraft_slsqp", kraft_slsqp_policy<>{});
    check_constraint_rows("kraft_slsqp", kraft_slsqp_policy<>{}, cases::gradient_options());
    check_unconstrained_and_bound("filter_slsqp", filter_slsqp_policy<>{});
    check_constraint_rows("filter_slsqp", filter_slsqp_policy<>{}, cases::gradient_options());
    check_bound_and_constraint_rows("nw_sqp", nw_sqp_policy<>{});
    check_bound_and_constraint_rows("filter_nw_sqp", filter_nw_sqp_policy<>{});
}

TEST_CASE("the oracle accepts the trust-region SQP solvers on generated quadratics", "[property]")
{
    check_bound_and_constraint_rows("tr_sqp_fast", tr_sqp<sqp_mode::fast>{});
    check_bound_and_constraint_rows("tr_sqp_accurate", tr_sqp<sqp_mode::accurate>{});
    check_bound_and_constraint_rows("filter_trsqp_fast", filter_trsqp<sqp_mode::fast>{});
    check_bound_and_constraint_rows("filter_trsqp_accurate", filter_trsqp<sqp_mode::accurate>{});
}

TEST_CASE("the oracle accepts the moving-asymptote solvers on generated quadratics", "[property]")
{
    check_bound_and_inequality("mma", mma_policy<>{});
    check_bound_and_inequality("ccsa_quadratic", ccsa_quadratic_policy<>{});
    // Excluded claim: on inequality rows move_limit_shrink can claim converged with a multiplier on a row
    // at slack 1e-3, 1.6 times past the stationarity limit, so it is held to feasibility there.
    check_bound_and_inequality("gcmma_move_limit_shrink", alternative::gcmma::move_limit_shrink_policy<>{}, oracle::claim::feasibility);
    check_bound_and_inequality("gcmma_raa_augmented", alternative::gcmma::raa_augmented_policy<>{});
    check_bound_and_inequality("gcmma_rho_wval", alternative::gcmma::rho_wval_policy<>{});
}
