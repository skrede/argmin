// Regression tests for the SE(3) pose-batch IK test problem.
// Covers dimension, problem-class flags, joint bound symmetry, and
// central-difference gradient consistency at the initial point.
//
// Reference: Craig, Introduction to Robotics, 3rd ed., Section 3.6
//            (Denavit-Hartenberg convention) and Section 5.5
//            (cross-product form of the geometric Jacobian).
//            argmin::ik_pose_batch ships the synthetic in-tree analog
//            of the external pose-batch IK workload.

#include "argmin/test_functions/ik_pose_batch.h"
#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>

using namespace argmin;

TEST_CASE("ik_pose_batch dimensions and class flags", "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    REQUIRE(p.dimension() == 30);
    REQUIRE(p.num_equality() == 0);
    REQUIRE(p.num_inequality() == 0);

    REQUIRE(has_class(p.pclass, problem_class::application));
    REQUIRE(has_class(p.pclass, problem_class::bound_constrained));
    REQUIRE(!has_class(p.pclass, problem_class::equality));
}

TEST_CASE("ik_pose_batch bounds are symmetric, finite, and within +/- pi",
          "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    const auto lb = p.lower_bounds();
    const auto ub = p.upper_bounds();
    constexpr double pi = std::numbers::pi_v<double>;
    REQUIRE(lb.size() == 30);
    REQUIRE(ub.size() == 30);
    for(int i = 0; i < lb.size(); ++i)
    {
        REQUIRE(std::isfinite(lb[i]));
        REQUIRE(std::isfinite(ub[i]));
        REQUIRE(lb[i] == -ub[i]);
        REQUIRE(std::abs(lb[i]) <= pi);
        REQUIRE(std::abs(ub[i]) <= pi);
    }
}

TEST_CASE("ik_pose_batch gradient matches central-difference at the initial point",
          "[ik_pose_batch]")
{
    using prob_t = ik_pose_batch<6, 5>;
    prob_t p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, prob_t::problem_dimension, 1> g;
    p.gradient(z0, g);

    const double h = 1e-5;
    double max_err = 0.0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_plus = z0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_minus = z0;
    for(int i = 0; i < p.dimension(); ++i)
    {
        z_plus[i] = z0[i] + h;
        z_minus[i] = z0[i] - h;
        const double f_plus = p.value(z_plus);
        const double f_minus = p.value(z_minus);
        const double fd = (f_plus - f_minus) / (2.0 * h);
        const double err = std::abs(fd - g[i]);
        if(err > max_err)
            max_err = err;
        z_plus[i] = z0[i];
        z_minus[i] = z0[i];
    }
    REQUIRE(max_err < 1e-4);
}

TEST_CASE("ik_pose_batch value is finite and non-negative at the initial point",
          "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    const auto z0 = p.initial_point();
    const double f0 = p.value(z0);
    REQUIRE(std::isfinite(f0));
    REQUIRE(f0 >= 0.0);
}
