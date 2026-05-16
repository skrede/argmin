// Unit tests for the L1-merit penalty calibration helpers.
//
// Validates calibrate_initial_penalty's two iter-0 floors:
//   1. Lambda-floor: sigma >= max_i |lambda_i| + delta (sufficient
//      penalty for L1-merit descent).
//   2. K-factor magnitude floor:
//      sigma >= K * max(1, |f_0| / (||c_0||_1 + eps))
//      (problem-scale floor guarding against under-weighted violation
//      term on objective-dominated initial points).
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Section 18.3, eq. 18.36 (sufficient penalty for descent);
//            Kraft 1988 DFVLR-FB 88-28, Section 2.2.6 (sigma update).

#include "argmin/detail/merit_function.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

TEST_CASE("calibrate_initial_penalty enforces lambda-floor", "[detail][merit]")
{
    // lambda_max = 5.0; lambda-floor = 5.0 + 1.0 = 6.0.
    // K-factor-floor = 10 * max(1, |1.0| / (100 + eps)) = 10 * 1 = 10.0.
    // Result = max(0.5, 6.0, 10.0) = 10.0 (K-factor dominates here).
    Eigen::VectorXd lambda(2);
    lambda << 5.0, -3.0;
    const double sigma_in = 0.5;
    const double f_0 = 1.0;
    const double c_0_l1 = 100.0;
    const double K_factor = 10.0;
    const double delta = 1.0;

    const double sigma = detail::calibrate_initial_penalty<double, Eigen::Dynamic>(
        sigma_in, lambda, f_0, c_0_l1, K_factor, delta);

    CHECK(sigma == Approx(10.0));
}

TEST_CASE("calibrate_initial_penalty enforces K-factor magnitude floor",
          "[detail][merit]")
{
    // lambda-floor = 0.1 + 1.0 = 1.1.
    // K-factor-floor = 10 * max(1, 100/1) = 1000.
    // Result = max(0.5, 1.1, 1000) = 1000 (K-factor dominates).
    Eigen::VectorXd lambda(1);
    lambda << 0.1;
    const double sigma_in = 0.5;
    const double f_0 = 100.0;
    const double c_0_l1 = 1.0;
    const double K_factor = 10.0;
    const double delta = 1.0;

    const double sigma = detail::calibrate_initial_penalty<double, Eigen::Dynamic>(
        sigma_in, lambda, f_0, c_0_l1, K_factor, delta);

    CHECK(sigma == Approx(1000.0));
}

TEST_CASE("calibrate_initial_penalty preserves sigma_in when both floors inactive",
          "[detail][merit]")
{
    // Empty lambda disables the lambda-floor.
    // K-factor-floor = 10 * max(1, 1/1) = 10.
    // Result = max(sigma_in=1e15, 10) = sigma_in (monotone preservation).
    Eigen::VectorXd lambda;  // empty
    const double sigma_in = 1e15;
    const double f_0 = 1.0;
    const double c_0_l1 = 1.0;

    const double sigma = detail::calibrate_initial_penalty<double, Eigen::Dynamic>(
        sigma_in, lambda, f_0, c_0_l1, 10.0, 1.0);

    CHECK(sigma == Approx(1e15));
}

TEST_CASE("calibrate_initial_penalty is monotone non-decreasing",
          "[detail][merit][property]")
{
    // For any sigma_in, the calibrated value must be >= sigma_in
    // (the function never decreases the input penalty).
    Eigen::VectorXd lambda(3);
    lambda << 1.0, 2.0, -3.0;

    for(double sigma_in : {0.0, 0.5, 1.0, 5.0, 100.0, 1e6, 1e12})
    {
        const double sigma = detail::calibrate_initial_penalty<double, Eigen::Dynamic>(
            sigma_in, lambda, 7.0, 2.0, 10.0, 1.0);
        CHECK(sigma >= sigma_in);
    }
}
