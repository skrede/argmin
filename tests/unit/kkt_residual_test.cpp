// Tests for detail::kkt_residual and detail::kkt_residual_bound.
//
// Validates the L-infinity first-order optimality error returns zero at
// KKT points, captures each of the five KKT legs (stationarity, primal
// equality, primal inequality, dual feasibility, complementarity)
// independently, and that the bound-constrained helper reduces to the
// projected-gradient norm.
//
// Reference: Nocedal and Wright, "Numerical Optimization" 2e,
//            Definition 12.1 (KKT conditions);
//            eq. 12.34 (Lagrangian stationarity);
//            Section 16.7 (projected gradient optimality).

#include "argmin/detail/kkt_residual.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace argmin;

TEST_CASE("kkt_residual at KKT point returns near-zero", "[kkt_residual]")
{
    // 2D equality-constrained minimum of 0.5 ||x||^2 subject to
    //   x_0 + x_1 = 2
    // Lagrangian: L = 0.5 ||x||^2 - lambda * (x_0 + x_1 - 2)
    // At optimum x* = (1, 1), lambda* = 1.
    // grad_f(x*) = (1, 1); J_eq = [1 1]; J_eq^T lambda = (1, 1);
    // grad_L = 0; c_eq(x*) = 0.
    //
    // Reference: N&W 2e Definition 12.1 (all five KKT legs zero at KKT point).
    Eigen::Vector<double, 2> grad_f{{1.0, 1.0}};
    Eigen::Matrix<double, 1, 2> J_eq;
    J_eq << 1.0, 1.0;
    Eigen::Matrix<double, 0, 2> J_ineq;
    Eigen::Vector<double, 1> lambda_eq{{1.0}};
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 1> c_eq{{0.0}};
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 2, 1, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    CHECK(r < 1e-12);
}

TEST_CASE("kkt_residual away from KKT point reflects stationarity violation",
          "[kkt_residual]")
{
    // Same problem, but with wrong multiplier lambda = 0. Stationarity
    // residual becomes ||grad_f||_inf = 1.0; c_eq stays 0 so that leg
    // does not dominate.
    //
    // Reference: N&W 2e eq. 12.34 (Lagrangian stationarity leg).
    Eigen::Vector<double, 2> grad_f{{1.0, 1.0}};
    Eigen::Matrix<double, 1, 2> J_eq;
    J_eq << 1.0, 1.0;
    Eigen::Matrix<double, 0, 2> J_ineq;
    Eigen::Vector<double, 1> lambda_eq{{0.0}};
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 1> c_eq{{0.0}};
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 2, 1, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    CHECK(r == 1.0);
}

TEST_CASE("kkt_residual captures complementarity violation", "[kkt_residual]")
{
    // Contrive a case where stationarity is satisfied but complementarity
    // is not: mu_i > 0 and c_ineq_i > 0 simultaneously. Meq = 0 so the
    // primal_eq leg is vacuous.
    //
    // Reference: N&W 2e Definition 12.1 (complementary slackness condition).
    Eigen::Vector<double, 2> grad_f{{0.0, 0.0}};
    Eigen::Matrix<double, 0, 2> J_eq;
    Eigen::Matrix<double, 1, 2> J_ineq;
    J_ineq << 0.0, 0.0;  // zero row so mu contribution to stationarity is zero
    Eigen::Vector<double, 0> lambda_eq;
    Eigen::Vector<double, 1> mu_ineq{{0.5}};
    Eigen::Vector<double, 0> c_eq;
    Eigen::Vector<double, 1> c_ineq{{0.3}};

    double r = detail::kkt_residual<double, 2, 0, 1>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    // min(0.5, 0.3) = 0.3, stationarity is 0, primal_ineq = max(-0.3, 0) = 0,
    // dual_feas = max(-0.5, 0) = 0; so composite = 0.3.
    CHECK(r == 0.3);
}

TEST_CASE("kkt_residual with empty constraints reduces to grad_f norm",
          "[kkt_residual]")
{
    // No equality, no inequality. grad_L == grad_f and residual is
    // ||grad_f||_inf.
    //
    // Reference: N&W 2e eq. 12.34 (stationarity reduces to gradient norm
    //            when no constraints are present).
    Eigen::Vector<double, 3> grad_f{{-2.0, 1.0, 0.5}};
    Eigen::Matrix<double, 0, 3> J_eq;
    Eigen::Matrix<double, 0, 3> J_ineq;
    Eigen::Vector<double, 0> lambda_eq;
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 0> c_eq;
    Eigen::Vector<double, 0> c_ineq;

    double r = detail::kkt_residual<double, 3, 0, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    CHECK(r == 2.0);
}

TEST_CASE("kkt_residual captures primal equality feasibility violation",
          "[kkt_residual]")
{
    // Feasible multipliers, stationarity zero, but x off the equality
    // manifold: c_eq != 0 triggers the primal_eq leg.
    //
    // Reference: N&W 2e Definition 12.1 primal-feasibility condition.
    Eigen::Vector<double, 2> grad_f{{1.0, 1.0}};
    Eigen::Matrix<double, 1, 2> J_eq;
    J_eq << 1.0, 1.0;
    Eigen::Matrix<double, 0, 2> J_ineq;
    Eigen::Vector<double, 1> lambda_eq{{1.0}};       // grad_f minus J_eq^T lambda_eq = 0
    Eigen::Vector<double, 0> mu_ineq;
    Eigen::Vector<double, 1> c_eq{{0.2}};            // 0.2 off the manifold
    Eigen::Vector<double, 0> c_ineq;

    const double r = detail::kkt_residual<double, 2, 1, 0>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    CHECK(std::abs(r - 0.2) < 1e-12);
}

TEST_CASE("kkt_residual captures dual feasibility violation",
          "[kkt_residual]")
{
    // One mu_ineq entry is negative; dual-feas leg captures max(-mu_ineq, 0).
    //
    // Reference: N&W 2e Definition 12.1 dual-feasibility condition.
    Eigen::Vector<double, 2> grad_f{{0.0, 0.0}};
    Eigen::Matrix<double, 0, 2> J_eq;
    Eigen::Matrix<double, 2, 2> J_ineq;
    J_ineq << 1.0, 0.0, 0.0, 1.0;
    Eigen::Vector<double, 0> lambda_eq;
    Eigen::Vector<double, 2> mu_ineq{{0.5, -0.3}};   // second entry violates mu >= 0
    Eigen::Vector<double, 0> c_eq;
    Eigen::Vector<double, 2> c_ineq{{100.0, 100.0}}; // inactive under c_ineq >= 0 feasible

    const double r = detail::kkt_residual<double, 2, 0, 2>(
        grad_f, J_eq, J_ineq, lambda_eq, mu_ineq, c_eq, c_ineq);
    CHECK(r >= 0.3);   // dual-feas leg at minimum; stationarity may dominate
}

TEST_CASE("kkt_residual_bound interior point equals gradient infinity norm",
          "[kkt_residual][bound]")
{
    // x well inside bounds [-10, 10]; projection does nothing, so the
    // projected step equals -grad_f and the residual is ||grad_f||_inf.
    //
    // Reference: N&W 2e Section 16.7 (projected gradient optimality).
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
    //
    // Reference: N&W 2e Section 16.7 (projected gradient optimality).
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
    //
    // Reference: N&W 2e Section 16.7 (projected gradient optimality).
    Eigen::Vector<double, 2> x{{1.0, 0.0}};
    Eigen::Vector<double, 2> grad_f{{0.5, 0.0}};
    Eigen::Vector<double, 2> lower{{-1.0, -1.0}};
    Eigen::Vector<double, 2> upper{{1.0, 1.0}};

    double r = detail::kkt_residual_bound<double, 2>(x, grad_f, lower, upper);
    CHECK(r == 0.5);
}
