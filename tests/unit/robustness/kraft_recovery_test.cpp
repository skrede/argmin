// Robustness: the factored-Hessian path of the Kraft 1988 §3.4
// augmented-QP recovery wrapper.
//
// kraft_lsq_qp_recovery_test.cpp drives the augmented recovery through the
// plain solve() entry (B, g). The factored-Hessian entry
// (solve_with_factored_hessian, which the SQP hot path uses to skip the
// internal LLT) has its OWN recovery leg -- it reconstructs B = E^T E and
// runs the augmented loop -- and that leg is otherwise unexercised. These
// cells drive it:
//
//   direct-optimal passthrough -- on a feasible QP the factored path returns
//                               the direct result with relaxation_factor 0
//                               and never enters recovery.
//   equality recovery          -- an infeasible equality linearization is
//                               resolved by the augmented (n+1) QP built from
//                               the reconstructed Hessian.
//   inequality recovery        -- an infeasible inequality linearization is
//                               resolved, exercising the max(b_ineq, 0)
//                               augmentation column in the factored leg.
//
// Reference: Kraft, D. (1988), A Software Package for Sequential Quadratic
//            Programming, DFVLR-FB 88-28, §3.4 (inconsistent linearization).

#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;
using namespace argmin::detail;

TEST_CASE("kraft factored-Hessian path is direct-optimal on a feasible QP",
          "[robustness][kraft-recovery][factored]")
{
    // Unconstrained QP with B = I => E = I, and E^T f = -g => f = -g. The
    // direct factored solve is optimal, so recovery never fires and
    // relaxation_factor stays 0.
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{0.1, -0.2}};
    Eigen::Vector<double, n> f = -g;

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(0, n);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);
    Eigen::Vector<double, n> p_lo{{-1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0}};

    kraft_lsq_qp_recovery_solver<double, n> recovery;
    recovery.resize(n, 0, 0, n, n);
    auto res = recovery.solve_with_factored_hessian(
        E, f, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    REQUIRE(res.status == qp_status::optimal);
    CHECK(res.relaxation_factor == Approx(0.0));
    // Unconstrained optimum p = -B^-1 g = -g for B = I.
    CHECK(res.x[0] == Approx(-0.1).margin(1e-9));
    CHECK(res.x[1] == Approx( 0.2).margin(1e-9));
}

TEST_CASE("kraft factored-Hessian recovery resolves an infeasible equality",
          "[robustness][kraft-recovery][factored]")
{
    // B = I (E = I), g = 0 (f = 0). Equality A_eq = [1, 0], b_eq = [2]
    // needs p[0] = 2 but the box forces p in [-1, 1]^2: infeasible direct,
    // resolved by the augmented recovery at s = 0.5 (mirrors the plain
    // solve() equality-recovery cell, but through the factored leg).
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> g = Eigen::Vector<double, n>::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(1, n);
    A_eq << 1.0, 0.0;
    Eigen::VectorXd b_eq(1);
    b_eq << 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);
    Eigen::Vector<double, n> p_lo{{-1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0}};

    kraft_lsq_qp_recovery_solver<double, n> recovery;
    recovery.resize(n, 1, 0, n, n);
    auto res = recovery.solve_with_factored_hessian(
        E, f, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    REQUIRE(res.status == qp_status::optimal);
    CHECK(res.relaxation_factor == Approx(0.5).margin(1e-6));
    CHECK(res.x[0] == Approx(1.0).margin(1e-6));
    CHECK(res.x[1] == Approx(0.0).margin(1e-9));
}

TEST_CASE("kraft factored-Hessian recovery resolves an infeasible inequality",
          "[robustness][kraft-recovery][factored]")
{
    // Inequality A_ineq = [1, 0], b_ineq = [10] needs p[0] >= 10; the box
    // forces p in [-1, 1]^2. The factored recovery's max(b_ineq, 0)
    // augmentation column resolves it with the box active (p[0] = 1,
    // s = 0.9).
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> g = Eigen::Vector<double, n>::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(0, n);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(1, n);
    A_ineq << 1.0, 0.0;
    Eigen::VectorXd b_ineq(1);
    b_ineq << 10.0;
    Eigen::Vector<double, n> p_lo{{-1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0}};

    kraft_lsq_qp_recovery_solver<double, n> recovery;
    recovery.resize(n, 0, 1, n, n);
    auto res = recovery.solve_with_factored_hessian(
        E, f, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    REQUIRE(res.status == qp_status::optimal);
    CHECK(res.relaxation_factor > 0.0);
    CHECK(res.x[0] == Approx(1.0).margin(1e-6));
    CHECK(res.relaxation_factor == Approx(0.9).margin(1e-6));
}
