// Unit witnesses for the L1-merit penalty cold-bump helper
// bump_sigma_for_descent.
//
// The helper raises sigma so the L1-merit directional derivative becomes
// strictly negative (p a descent direction), capped at sigma_max. When the
// descent-restoring bump would exceed the cap it is clamped and the returned
// sigma_bump_result.saturated flag is set: the clamped sigma is provably too
// small to make p a descent direction, so the consuming SQP policies must
// short-circuit their line search into the recovery ladder rather than
// backtracking against an unsatisfiable Armijo test. These cases pin that
// signal with hand-computed values.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 18.3, eq. 18.36 (sufficient penalty for descent);
//            Kraft 1988 DFVLR-FB 88-28, Section 2.2.6 (sigma update);
//            Powell 1978, Section 6 (penalty cold-bump).

#include "argmin/detail/merit_function.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

TEST_CASE("bump_sigma_for_descent signals saturation when the bump hits sigma_max",
          "[detail][merit][sigma_bump]")
{
    // dphi = grad_f_dot_p - sigma_in * h4 * violation
    //      = 1e6 - 1.0 * 1.0 * 1.0 = 999999 >= 0  -> guard taken.
    // bumped = max(sigma_in, |grad_f_dot_p| / (violation * h4) + 1)
    //        = max(1.0, 1e6 / 1.0 + 1) = 1000001.
    // saturated = (1000001 > sigma_max = 100) = true (decided before clamp).
    // sigma = min(1000001, 100) = 100.
    const double sigma_in = 1.0;
    const double grad_f_dot_p = 1.0e6;
    const double violation = 1.0;
    const double h4 = 1.0;
    const double sigma_max = 100.0;

    const auto result = detail::bump_sigma_for_descent<double>(
        sigma_in, grad_f_dot_p, violation, h4, sigma_max);

    CHECK(result.saturated == true);
    CHECK(result.sigma == sigma_max);  // capped at 100.0 exactly
}

TEST_CASE("bump_sigma_for_descent does not saturate when the bump stays below sigma_max",
          "[detail][merit][sigma_bump]")
{
    // Same descent-restoring bump (1000001) but a cap well above it: the
    // clamp is inert, the uncapped sigma is returned, and saturated is false.
    const double sigma_in = 1.0;
    const double grad_f_dot_p = 1.0e6;
    const double violation = 1.0;
    const double h4 = 1.0;
    const double sigma_max = 1.0e10;

    const auto result = detail::bump_sigma_for_descent<double>(
        sigma_in, grad_f_dot_p, violation, h4, sigma_max);

    CHECK(result.saturated == false);
    CHECK(result.sigma == 1000001.0);  // |grad_f_dot_p| / violation + 1
}

TEST_CASE("bump_sigma_for_descent passes sigma through when p is already a descent direction",
          "[detail][merit][sigma_bump]")
{
    // dphi = -5.0 - 10.0 * 1.0 * 1.0 = -15 < 0 -> guard NOT taken; the helper
    // returns sigma_in bit-identically with saturated = false.
    const double sigma_in = 10.0;
    const double grad_f_dot_p = -5.0;
    const double violation = 1.0;

    const auto result = detail::bump_sigma_for_descent<double>(
        sigma_in, grad_f_dot_p, violation);

    CHECK(result.saturated == false);
    CHECK(result.sigma == sigma_in);  // 10.0 exactly, no bump
}

TEST_CASE("bump_sigma_for_descent passes sigma through at a feasible iterate",
          "[detail][merit][sigma_bump]")
{
    // violation == 0 disables the guard even with a non-negative slope: no
    // finite bump can restore descent at a feasible point, so sigma_in is
    // returned unchanged and unsaturated.
    const double sigma_in = 2.0;
    const double grad_f_dot_p = 5.0;
    const double violation = 0.0;

    const auto result = detail::bump_sigma_for_descent<double>(
        sigma_in, grad_f_dot_p, violation);

    CHECK(result.saturated == false);
    CHECK(result.sigma == sigma_in);  // 2.0 exactly, no bump
}
