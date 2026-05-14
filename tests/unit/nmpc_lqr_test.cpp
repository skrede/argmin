// Regression tests for the finite-horizon LQR-shaped NMPC test
// problem. Covers dimension reporting, problem-class flags, dynamics
// feasibility at the initial point, and central-difference gradient
// consistency.
//
// Reference: Anderson and Moore, Optimal Control: Linear Quadratic
//            Methods, 1990, Section 2.3 (LQR / Riccati);
//            argmin::nmpc_lqr<H> ships the synthetic in-tree analog
//            of the external NMPC h=10/20 workload.

#include "argmin/test_functions/problem_class.h"
#include "argmin/test_functions/nmpc_lqr.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace argmin;

TEST_CASE("nmpc_lqr dimensions and class flags", "[nmpc_lqr]")
{
    nmpc_lqr<10> p10{};
    REQUIRE(p10.dimension() == 60);
    REQUIRE(p10.num_equality() == 40);
    REQUIRE(p10.num_inequality() == 0);

    nmpc_lqr<20> p20{};
    REQUIRE(p20.dimension() == 120);
    REQUIRE(p20.num_equality() == 80);
    REQUIRE(p20.num_inequality() == 0);

    REQUIRE(has_class(p10.pclass, problem_class::application));
    REQUIRE(has_class(p10.pclass, problem_class::equality));
    REQUIRE(has_class(p10.pclass, problem_class::bound_constrained));

    REQUIRE(has_class(p20.pclass, problem_class::application));
    REQUIRE(has_class(p20.pclass, problem_class::equality));
    REQUIRE(has_class(p20.pclass, problem_class::bound_constrained));

    // application is the seventh atomic flag; compositions are
    // OR-clean across the existing six flags.
    constexpr auto composed =
        problem_class::bound_constrained | problem_class::application;
    REQUIRE(has_class(composed, problem_class::application));
    REQUIRE(has_class(composed, problem_class::bound_constrained));
    REQUIRE(!has_class(composed, problem_class::equality));
}

TEST_CASE("nmpc_lqr initial point is feasible for the dynamics equalities",
          "[nmpc_lqr]")
{
    nmpc_lqr<10> p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, nmpc_lqr<10>::constraint_count, 1> c_eq;
    p.constraints(z0, c_eq);
    REQUIRE(c_eq.norm() < 1e-10);
}

TEST_CASE("nmpc_lqr bound layout: states unbounded, controls box-bounded",
          "[nmpc_lqr]")
{
    nmpc_lqr<10> p{};
    const auto lb = p.lower_bounds();
    const auto ub = p.upper_bounds();
    constexpr double inf = std::numeric_limits<double>::infinity();
    // Each stage block has n_x = 4 state entries (indices 0..3 of
    // stage) and n_u = 2 control entries (indices 4..5 of stage).
    for(int k = 0; k < 10; ++k)
    {
        const int x_off = k * 6;
        for(int j = 0; j < 4; ++j)
        {
            REQUIRE(lb[x_off + j] == -inf);
            REQUIRE(ub[x_off + j] == inf);
        }
        const int u_off = k * 6 + 4;
        for(int j = 0; j < 2; ++j)
        {
            REQUIRE(lb[u_off + j] == -2.0);
            REQUIRE(ub[u_off + j] == 2.0);
        }
    }
}

TEST_CASE("nmpc_lqr gradient matches central-difference at the initial point",
          "[nmpc_lqr]")
{
    using prob_t = nmpc_lqr<10>;
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

TEST_CASE("nmpc_lqr constraint jacobian matches central-difference at z0",
          "[nmpc_lqr]")
{
    using prob_t = nmpc_lqr<10>;
    prob_t p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, prob_t::constraint_count, prob_t::problem_dimension> J;
    p.constraint_jacobian(z0, J);

    const double h = 1e-5;
    Eigen::Matrix<double, prob_t::constraint_count, 1> c_plus;
    Eigen::Matrix<double, prob_t::constraint_count, 1> c_minus;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_plus = z0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_minus = z0;
    double max_err = 0.0;
    for(int j = 0; j < p.dimension(); ++j)
    {
        z_plus[j] = z0[j] + h;
        z_minus[j] = z0[j] - h;
        p.constraints(z_plus, c_plus);
        p.constraints(z_minus, c_minus);
        for(int i = 0; i < p.num_equality(); ++i)
        {
            const double fd = (c_plus[i] - c_minus[i]) / (2.0 * h);
            const double err = std::abs(fd - J(i, j));
            if(err > max_err)
                max_err = err;
        }
        z_plus[j] = z0[j];
        z_minus[j] = z0[j];
    }
    REQUIRE(max_err < 1e-7);
}
