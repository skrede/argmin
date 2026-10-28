#include "property_judge.h"
#include "generated_least_squares.h"

#include "cases/case_options.h"

#include "argmin/solver/lm_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace argmin;

namespace
{

// The projected Gauss-Newton policies read the box unconditionally, so their unconstrained class has an
// infinite one. Excluded region: a box that binds at the solution. Both policies test convergence with the
// gradient of the previous iterate at the accepted one, so a step that lands on the box can claim
// converged at a non-stationary point (tests/defects/property_findings_test.cpp).
template <typename Policy>
void check_projected(const std::string& key, const Policy& policy)
{
    property::check_judged(key + " unconstrained", property::boxed_least_squares(property::box_kind::infinite), policy, cases::least_squares_options());
    property::check_judged(key + " bound", property::boxed_least_squares(property::box_kind::around_minimizer), policy, cases::least_squares_options());
}

}

TEST_CASE("the oracle accepts Levenberg-Marquardt on generated linear least squares", "[property]")
{
    property::check_judged("lm unconstrained", property::unconstrained_least_squares(), lm_policy<>{}, cases::least_squares_options());
}

TEST_CASE("the oracle accepts the projected Gauss-Newton solvers on generated linear least squares", "[property]")
{
    check_projected("projected_gn", projected_gn_policy<>{});
    check_projected("projected_gradient_gn", projected_gradient_gn_policy<>{});
}
