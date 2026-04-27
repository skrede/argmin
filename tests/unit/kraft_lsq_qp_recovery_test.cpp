#include "nablapp/detail/kraft_lsq_qp.h"
#include "nablapp/detail/kraft_lsq_qp_recovery.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;
using namespace nablapp::detail;

// ---------------------------------------------------------------------------
// Bit-identity with direct kraft_lsq_qp_solver on feasible QPs.
//
// The recovery wrapper must not perturb the optimal-path output. On any
// QP where the direct solver returns optimal, the wrapper must return the
// same x, lambda, and status. relaxation_factor must be zero.
// ---------------------------------------------------------------------------

TEST_CASE("kraft_lsq_qp_recovery is bit-identical to direct solver on feasible QPs",
          "[kraft_lsq_qp_recovery]")
{
    // Mixed feasible QP: n=4, equality + inactive inequality + finite box.
    // Mirrors the existing 'kraft_lsq_qp_solver mixed HS071-like subproblem'
    // shape so the comparison covers a representative SQP regime.
    constexpr int n = 4;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{0.1, -0.2, 0.05, -0.1}};

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(1, n);
    A_eq << 1.0, 1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq(1);
    b_eq << 0.0;

    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(1, n);
    A_ineq << 1.0, -1.0, 0.0, 0.0;
    Eigen::VectorXd b_ineq(1);
    b_ineq << -10.0;

    Eigen::Vector<double, n> p_lo{{-1.0, -1.0, -1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0,  1.0,  1.0}};

    kraft_lsq_qp_solver<double, n> direct;
    direct.resize(n, 1, 1, n, n);
    auto direct_res = direct.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
    REQUIRE(direct_res.status == qp_status::optimal);

    kraft_lsq_qp_recovery_solver<double, n> recovery;
    recovery.resize(n, 1, 1, n, n);
    auto recovery_res = recovery.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(recovery_res.status == qp_status::optimal);
    CHECK(recovery_res.relaxation_factor == Approx(0.0));

    // Bit-identical x. Use exact equality, not Approx -- the recovery
    // wrapper must invoke the direct solver and return its result
    // unchanged on the optimal path.
    REQUIRE(recovery_res.x.size() == direct_res.x.size());
    for(int i = 0; i < n; ++i)
        CHECK(recovery_res.x[i] == direct_res.x[i]);

    REQUIRE(recovery_res.lambda.size() == direct_res.lambda.size());
    for(int i = 0; i < direct_res.lambda.size(); ++i)
        CHECK(recovery_res.lambda[i] == direct_res.lambda[i]);
}

TEST_CASE("kraft_lsq_qp_recovery preserves dynamic-N direct path",
          "[kraft_lsq_qp_recovery]")
{
    // Same as above but with N = Dynamic, exercising the Dynamic
    // augmented_dim<Dynamic> specialization. The direct path must still
    // be invoked exactly once and its result forwarded unchanged.
    const int n = 3;
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd g(n); g << 0.5, -0.3, 0.2;

    Eigen::MatrixXd A_eq(0, n);
    Eigen::VectorXd b_eq(0);

    Eigen::MatrixXd A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    Eigen::VectorXd p_lo = -Eigen::VectorXd::Ones(n);
    Eigen::VectorXd p_hi =  Eigen::VectorXd::Ones(n);

    kraft_lsq_qp_recovery_solver<double, Eigen::Dynamic> recovery;
    recovery.resize(n, 0, 0, n, n);
    auto res = recovery.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);
    CHECK(res.relaxation_factor == Approx(0.0));
    // Unconstrained QP: optimum is p = -B^-1 g = -g for B = I.
    CHECK(res.x[0] == Approx(-0.5));
    CHECK(res.x[1] == Approx( 0.3));
    CHECK(res.x[2] == Approx(-0.2));
}

// ---------------------------------------------------------------------------
// Augmented-QP recovery on infeasible linearization (Kraft 1988 §3.4).
// ---------------------------------------------------------------------------

TEST_CASE("kraft_lsq_qp_recovery resolves infeasible equality via augmentation",
          "[kraft_lsq_qp_recovery]")
{
    // n=2, B=I, g=0. Single equality A_eq = [1, 0], b_eq = [2] requires
    // p[0] = 2; box bounds force p ∈ [-1, 1]^2. Linearization infeasible.
    //
    // Augmented problem (per Kraft §3.4):
    //   min 0.5 (p[0]² + p[1]² + rho s²) + 0
    //   s.t. p[0] + 2s = 2     (augmentation column = b_eq[0] = 2)
    //        p ∈ [-1, 1]², s ∈ [0, 1]
    //
    // Substituting p[0] = 2(1 - s) and minimizing over p[1] (free; opt at 0):
    //   f(s) = 0.5 (4(1-s)² + rho s²) = 2(1-s)² + 0.5 rho s²
    // Constraint p[0] ∈ [-1, 1] requires 2(1-s) ∈ [-1, 1] -> s ∈ [0.5, 1.5].
    // Combined with s ∈ [0, 1]: s ∈ [0.5, 1].
    //
    // Unconstrained min of f: df/ds = -4(1-s) + rho s = (4+rho)s - 4 = 0
    //   -> s* = 4/(4+rho). At rho=100: s* ≈ 0.0385, below the 0.5 floor.
    // So the constrained optimum is s = 0.5, p[0] = 1, p[1] = 0,
    // relaxation_factor = 0.5.
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g = Eigen::Vector<double, n>::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(1, n);
    A_eq << 1.0, 0.0;
    Eigen::VectorXd b_eq(1);
    b_eq << 2.0;

    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    Eigen::Vector<double, n> p_lo{{-1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0}};

    // Direct solver should fail (infeasible linearization).
    kraft_lsq_qp_solver<double, n> direct;
    direct.resize(n, 1, 0, n, n);
    auto direct_res = direct.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);
    CHECK(direct_res.status != qp_status::optimal);

    // Recovery solver should succeed via augmentation.
    kraft_lsq_qp_recovery_solver<double, n> recovery;
    recovery.resize(n, 1, 0, n, n);
    auto rec_res = recovery.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    REQUIRE(rec_res.status == qp_status::optimal);

    // Expected: s = 0.5, p[0] = 1.0, p[1] = 0.0.
    CHECK(rec_res.relaxation_factor == Approx(0.5).margin(1e-6));
    CHECK(rec_res.x[0] == Approx(1.0).margin(1e-6));
    CHECK(rec_res.x[1] == Approx(0.0).margin(1e-9));
}

TEST_CASE("kraft_lsq_qp_recovery resolves infeasible inequality via augmentation",
          "[kraft_lsq_qp_recovery]")
{
    // n=2, B=I, g=0. Inequality A_ineq = [1, 0], b_ineq = [10] requires
    // p[0] >= 10; box bounds force p ∈ [-1, 1]^2. Infeasible.
    //
    // Augmented problem:
    //   min 0.5 (p[0]² + p[1]² + rho s²)
    //   s.t. p[0] + max(b_ineq[0], 0) s = p[0] + 10 s >= 10
    //        p ∈ [-1, 1]², s ∈ [0, 1]
    //
    // At s=1, the constraint becomes p[0] + 10 >= 10, i.e. p[0] >= 0,
    // which is satisfiable at p[0] = 0. Unconstrained min: p = 0, s = 0.
    // But the constraint at s=0 is p[0] >= 10, infeasible.
    // The augmented optimum picks the smallest s such that the relaxed
    // constraint can be satisfied at minimum (objective + penalty) cost.
    //
    // With p[0] = 0 (free interior of box), constraint becomes
    // 10 s >= 10, i.e. s >= 1. So s = 1 forced. relaxation_factor = 1.
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
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
    auto res = recovery.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    REQUIRE(res.status == qp_status::optimal);
    CHECK(res.relaxation_factor > 0.0);
    // The augmented QP gives p[0] some value in [0, 1] depending on the
    // penalty/p balance. At rho=100 the optimum balances p[0]² + 100 s²
    // subject to p[0] + 10 s >= 10. Lagrangian: p[0] = lambda,
    // 100 s = 10 lambda -> s = lambda/10. Constraint active:
    // lambda + lambda = 10 -> lambda = 5. So p[0] = 5? But box bound
    // says p[0] <= 1. So box is active: p[0] = 1, then constraint
    // 1 + 10 s = 10 -> s = 0.9.
    CHECK(res.x[0] == Approx(1.0).margin(1e-6));
    CHECK(res.x[1] == Approx(0.0).margin(1e-9));
    CHECK(res.relaxation_factor == Approx(0.9).margin(1e-6));
}
