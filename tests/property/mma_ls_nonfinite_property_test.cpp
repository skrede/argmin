#include "nlp_policies.h"
#include "property_judge.h"
#include "nonfinite_support.h"
#include "generated_problems.h"
#include "generated_least_squares.h"

#include "cases/case_options.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace argmin;
using namespace argmin::property;

namespace
{

const std::vector<oracle::callback>& residual_callbacks()
{
    static const std::vector<oracle::callback> callbacks{oracle::callback::residuals, oracle::callback::jacobian};
    return callbacks;
}

template <typename Policy>
void check_moving_asymptote(const std::string& key, const Policy& policy)
{
    check_non_finite(key, inequality_quadratic(start_kind::feasible), policy, moving_asymptote_property_options());
}

template <typename Policy>
Policy with_dogleg(Policy policy)
{
    policy.options.use_dogleg = true;
    return policy;
}

template <typename Policy>
void check_projected(const std::string& key, const Policy& policy)
{
    const solver_options<> options = cases::least_squares_options();
    check_non_finite(key, boxed_least_squares(box_kind::around_minimizer), policy, options, residual_callbacks());
    check_non_finite(key + "_dogleg", boxed_least_squares(box_kind::around_minimizer), with_dogleg(policy), options, residual_callbacks());
}

}

TEST_CASE("the moving-asymptote solvers never report a non-finite value as success or as their point", "[property]")
{
    check_moving_asymptote("mma", mma_policy<>{});
    check_moving_asymptote("ccsa_quadratic", ccsa_quadratic_policy<>{});
    check_moving_asymptote("gcmma_move_limit_shrink", alternative::gcmma::move_limit_shrink_policy<>{});
    check_moving_asymptote("gcmma_raa_augmented", alternative::gcmma::raa_augmented_policy<>{});
    check_moving_asymptote("gcmma_rho_wval", alternative::gcmma::rho_wval_policy<>{});
}

TEST_CASE("the least-squares solvers never report a non-finite value as success or as their point", "[property]")
{
    check_non_finite("lm", unconstrained_least_squares(), lm_policy<>{}, cases::least_squares_options(), residual_callbacks());
    check_projected("projected_gn", projected_gn_policy<>{});
    check_projected("projected_gradient_gn", projected_gradient_gn_policy<>{});
}
