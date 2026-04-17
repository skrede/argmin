// Tests for detail::kkt_residual and detail::kkt_residual_bound.
//
// Validates the L-infinity KKT error returns zero at KKT points,
// captures stationarity and complementarity violations independently,
// and reduces to the projected-gradient norm for bound-constrained
// problems.
//
// Reference: N&W 2e Section 12.1 (KKT conditions);
//            N&W 2e Section 12.3 / eq. 12.34 (Lagrangian stationarity);
//            N&W 2e Section 16.7 (projected gradient optimality).

#include "nablapp/detail/kkt_residual.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace nablapp;

TEST_CASE("kkt_residual at KKT point returns near-zero", "[kkt_residual]")
{
    // 2D equality-constrained minimum of 0.5 ||x||^2 subject to
    //   x_0 + x_1 = 2
    // Lagrangian: L = 0.5 ||x||^2 - lambda * (x_0 + x_1 - 2)
    // At optimum x* = (1, 1), lambda* = 1.
    // grad_f(x*) = (1, 1); J_eq = [1 1]; J_eq^T lambda = (1, 1);
    // grad_L = 0.
    Eigen::Vector<double, 2> grad_f{{1.0, 1.0}};
    Eigen::Matrix<double, 1, 2> J_eq;
    J_eq << 1.0, 1.0;
    Eigen::Matrix<double, 0, 2> J_ineq;
    Eigen::Vector<double, 1> lambda_eq{{1.0}};
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 2, 1, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_ineq);
    CHECK(r < 1e-12);
}

TEST_CASE("kkt_residual away from KKT point reflects stationarity violation",
          "[kkt_residual]")
{
    // Same problem, but with wrong multiplier lambda = 0. Stationarity
    // residual becomes ||grad_f||_inf = 1.0.
    Eigen::Vector<double, 2> grad_f{{1.0, 1.0}};
    Eigen::Matrix<double, 1, 2> J_eq;
    J_eq << 1.0, 1.0;
    Eigen::Matrix<double, 0, 2> J_ineq;
    Eigen::Vector<double, 1> lambda_eq{{0.0}};
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 2, 1, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_ineq);
    CHECK(r == 1.0);
}

TEST_CASE("kkt_residual captures complementarity violation", "[kkt_residual]")
{
    // Contrive a case where stationarity is satisfied but complementarity
    // is not: mu_i > 0 and c_ineq_i > 0 simultaneously.
    Eigen::Vector<double, 2> grad_f{{0.0, 0.0}};
    Eigen::Matrix<double, 0, 2> J_eq;
    Eigen::Matrix<double, 1, 2> J_ineq;
    J_ineq << 0.0, 0.0;  // zero row so mu contribution to stationarity is zero
    Eigen::Vector<double, 0> lambda_eq;
    Eigen::Vector<double, 1> mu_ineq{{0.5}};
    Eigen::Vector<double, 1> c_ineq{{0.3}};

    double r = detail::kkt_residual<double, 2, 0, 1>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_ineq);
    // min(0.5, 0.3) = 0.3, stationarity is 0, so residual is 0.3.
    CHECK(r == 0.3);
}

TEST_CASE("kkt_residual with empty constraints reduces to grad_f norm",
          "[kkt_residual]")
{
    // No equality, no inequality. grad_L == grad_f and residual is
    // ||grad_f||_inf.
    Eigen::Vector<double, 3> grad_f{{-2.0, 1.0, 0.5}};
    Eigen::Matrix<double, 0, 3> J_eq;
    Eigen::Matrix<double, 0, 3> J_ineq;
    Eigen::Vector<double, 0> lambda_eq;
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 3, 0, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_ineq);
    CHECK(r == 2.0);
}

TEST_CASE("kkt_residual_bound interior point equals gradient infinity norm",
          "[kkt_residual][bound]")
{
    // x well inside bounds [-10, 10]; projection does nothing, so the
    // projected step equals -grad_f and the residual is ||grad_f||_inf.
    Eigen::Vector<double, 3> x{{0.0, 0.0, 0.0}};
    Eigen::Vector<double, 3> grad_f{{0.5, -1.5, 0.25}};
    Eigen::Vector<double, 3> lower{{-10.0, -10.0, -10.0}};
    Eigen::Vector<double, 3> upper{{10.0, 10.0, 10.0}};

    double r = detail::kkt_residual_bound<double, 3>(x, grad_f, lower, upper);
    CHECK(r == 1.5);
}

TEST_CASE("kkt_residual_bound at active bound with outward gradient is near-zero",
          "[kkt_residual][bound]")
{
    // x_0 at upper bound with grad_f[0] negative (pointing outward, i.e.
    // the unconstrained step would push beyond the bound). The projection
    // clips back to x itself, so the projected step is zero on that
    // component.
    Eigen::Vector<double, 2> x{{1.0, 0.0}};
    Eigen::Vector<double, 2> grad_f{{-2.0, 0.0}};
    Eigen::Vector<double, 2> lower{{-1.0, -1.0}};
    Eigen::Vector<double, 2> upper{{1.0, 1.0}};

    double r = detail::kkt_residual_bound<double, 2>(x, grad_f, lower, upper);
    CHECK(r == 0.0);
}

TEST_CASE("kkt_residual_bound inward gradient at active bound is captured",
          "[kkt_residual][bound]")
{
    // x_0 at upper bound with grad_f[0] positive (inward descent feasible).
    // The projected step is -grad_f[0] clipped onto the feasible side;
    // residual is |grad_f[0]|.
    Eigen::Vector<double, 2> x{{1.0, 0.0}};
    Eigen::Vector<double, 2> grad_f{{0.5, 0.0}};
    Eigen::Vector<double, 2> lower{{-1.0, -1.0}};
    Eigen::Vector<double, 2> upper{{1.0, 1.0}};

    double r = detail::kkt_residual_bound<double, 2>(x, grad_f, lower, upper);
    CHECK(r == 0.5);
}
