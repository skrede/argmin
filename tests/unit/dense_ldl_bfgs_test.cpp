#include "argmin/detail/dense_ldl_bfgs.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>

using Catch::Approx;
using namespace argmin::detail;

namespace
{

// Materialize the Cholesky factor of B via Eigen's LLT for cross-check.
template <int N>
Eigen::Matrix<double, N, N> chol_upper_(const Eigen::Matrix<double, N, N>& B)
{
    Eigen::LLT<Eigen::Matrix<double, N, N>> llt(B);
    REQUIRE(llt.info() == Eigen::Success);
    return llt.matrixL().transpose();
}

}

// ---------------------------------------------------------------------------
// dense_ldl_bfgs invariants
// ---------------------------------------------------------------------------

TEST_CASE("dense_ldl_bfgs reset gives identity Hessian", "[dense_ldl_bfgs]")
{
    constexpr int n = 5;
    dense_ldl_bfgs<double, n> bfgs(n);
    const auto& B = bfgs.hessian();
    REQUIRE(B.isApprox(Eigen::Matrix<double, n, n>::Identity(), 1e-14));
}

TEST_CASE("dense_ldl_bfgs single PD push satisfies secant equation", "[dense_ldl_bfgs]")
{
    // After one BFGS update with positive curvature (s^T y > 0.2 * s^T B s,
    // so no Powell damping kicks in), the secant equation B_new * s = y
    // must hold to roundoff. N&W eq. 8.18.
    constexpr int n = 4;
    dense_ldl_bfgs<double, n> bfgs(n);

    Eigen::Vector<double, n> s{{1.0, 0.5, -0.3, 0.7}};
    Eigen::Vector<double, n> y{{2.0, 1.1,  0.4, 1.5}};  // s.dot(y) = 2 + 0.55 - 0.12 + 1.05 = 3.48
                                                        // s.dot(B s) = s.dot(s) = 1.83 (B = I)
                                                        // 0.2 * 1.83 = 0.366 << 3.48 -> no damping

    bfgs.push(s, y);

    Eigen::Vector<double, n> Bs = bfgs.hessian() * s;
    REQUIRE(Bs.isApprox(y, 1e-12));

    // multiply() (packed-factor path) must agree with the materialized B.
    Eigen::Vector<double, n> Bs_packed = bfgs.multiply(s);
    REQUIRE(Bs_packed.isApprox(y, 1e-12));
}

TEST_CASE("dense_ldl_bfgs Powell damping engages for low s^T y", "[dense_ldl_bfgs]")
{
    // Construct a pair where s^T y < 0.2 * s^T B s with B = I, so
    // s^T y < 0.2 * s^T B s. The Powell-damped update should produce
    // s^T (B_new * s) = 0.2 * s^T B_old s (positive by construction).
    //
    // Use a warm-up push first to bypass the Shanno initial-Hessian
    // rescale (which would absorb the curvature-magnitude mismatch
    // before the second push could trigger damping).
    constexpr int n = 3;
    dense_ldl_bfgs<double, n> bfgs(n);

    // Warm-up push: well-conditioned positive curvature, no damping.
    Eigen::Vector<double, n> s0{{0.5, 0.5, 0.5}};
    Eigen::Vector<double, n> y0{{1.0, 1.0, 1.0}};
    bfgs.push(s0, y0);

    // Capture B after warm-up so we can verify the damping invariant.
    const Eigen::Matrix<double, n, n> B_pre = bfgs.hessian();

    // Damped pair: s^T y arranged below 0.2 * s^T B s.
    Eigen::Vector<double, n> s{{1.0, 1.0, 1.0}};
    Eigen::Vector<double, n> y{{0.05, 0.05, 0.05}};
    const double sBs_pre = s.dot(B_pre * s);
    REQUIRE(s.dot(y) < 0.2 * sBs_pre);  // damping must engage

    bfgs.push(s, y);

    Eigen::LLT<Eigen::Matrix<double, n, n>> llt(bfgs.hessian());
    REQUIRE(llt.info() == Eigen::Success);

    const double sBs_new = s.dot(bfgs.hessian() * s);
    REQUIRE(sBs_new == Approx(0.2 * sBs_pre).margin(1e-10));
}

TEST_CASE("dense_ldl_bfgs first-push Shanno init scale", "[dense_ldl_bfgs]")
{
    // The first accepted (s, y) pair should rescale the identity
    // baseline by theta = y^T y / s^T y (N&W eq. 6.20) before the
    // BFGS rank-1 updates fire. Subsequent pushes leave D unscaled.
    constexpr int n = 3;
    dense_ldl_bfgs<double, n> bfgs(n);

    // Choose s, y so the rescaled-identity contribution dominates the
    // rank-1 corrections: s = e_1 ensures the rank-2 BFGS update only
    // touches the (1, 1) block, leaving the other diagonals at theta.
    Eigen::Vector<double, n> s{{1.0, 0.0, 0.0}};
    Eigen::Vector<double, n> y{{4.0, 0.0, 0.0}};   // theta = 16 / 4 = 4

    bfgs.push(s, y);
    const Eigen::Matrix<double, n, n>& B = bfgs.hessian();

    // Untouched diagonals must equal theta exactly.
    REQUIRE(B(1, 1) == Approx(4.0).margin(1e-12));
    REQUIRE(B(2, 2) == Approx(4.0).margin(1e-12));
    // Off-diagonals across rows/cols 1, 2 must vanish.
    REQUIRE(B(1, 2) == Approx(0.0).margin(1e-12));
    REQUIRE(B(0, 2) == Approx(0.0).margin(1e-12));
}

TEST_CASE("dense_ldl_bfgs rejects pairs with non-positive damping target",
          "[dense_ldl_bfgs]")
{
    // Tiny || s || -> reject without modifying B.
    constexpr int n = 3;
    dense_ldl_bfgs<double, n> bfgs(n);
    Eigen::Vector<double, n> s = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> y{{1.0, 1.0, 1.0}};
    bfgs.push(s, y);
    REQUIRE(bfgs.hessian().isApprox(Eigen::Matrix<double, n, n>::Identity(), 1e-14));
}

TEST_CASE("dense_ldl_bfgs factor_to_E_and_f matches Cholesky-derived E, f",
          "[dense_ldl_bfgs]")
{
    // After multiple pushes, factor_to_E_and_f must agree with what
    // kraft_lsq_qp_solver currently computes via LLT(B): E = L^T (where
    // B = L L^T) and f = -L^{-1} g.
    constexpr int n = 4;
    dense_ldl_bfgs<double, n> bfgs(n);

    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Generate a few well-conditioned PD curvature pairs.
    for(int k = 0; k < 5; ++k)
    {
        Eigen::Vector<double, n> s, y;
        for(int i = 0; i < n; ++i)
        {
            s[i] = nd(rng);
            y[i] = nd(rng);
        }
        // Force s^T y > 0 with margin.
        if(s.dot(y) <= 0.5 * s.squaredNorm())
            y = s + 0.5 * y;
        bfgs.push(s, y);
    }

    Eigen::Vector<double, n> g{{0.4, -0.2, 1.1, -0.7}};

    Eigen::Matrix<double, n, n> E_pkt;
    Eigen::Vector<double, n> f_pkt;
    bfgs.factor_to_E_and_f(E_pkt, g, f_pkt);

    // Cross-check against LLT(B).
    const auto& B = bfgs.hessian();
    Eigen::LLT<Eigen::Matrix<double, n, n>> llt(B);
    REQUIRE(llt.info() == Eigen::Success);
    const Eigen::Matrix<double, n, n> L_chol = llt.matrixL();
    Eigen::Matrix<double, n, n> E_chol = L_chol.transpose();
    Eigen::Vector<double, n> f_chol = -L_chol.template triangularView<Eigen::Lower>().solve(g);

    // E and E_chol must agree as factors of B (E^T E = B in both cases).
    // The factor itself is not unique up to sign; cross-check via E^T E.
    Eigen::Matrix<double, n, n> EtE_pkt = E_pkt.transpose() * E_pkt;
    Eigen::Matrix<double, n, n> EtE_chol = E_chol.transpose() * E_chol;
    REQUIRE(EtE_pkt.isApprox(B, 1e-10));
    REQUIRE(EtE_chol.isApprox(B, 1e-10));

    // f must satisfy E^T f = -g in both representations.
    REQUIRE((E_pkt.transpose() * f_pkt).isApprox(-g, 1e-10));
    REQUIRE((E_chol.transpose() * f_chol).isApprox(-g, 1e-10));
}

TEST_CASE("dense_ldl_bfgs sequence of pushes preserves SPD", "[dense_ldl_bfgs]")
{
    // Stress test: 100 random PD pushes; B must remain SPD throughout.
    constexpr int n = 6;
    dense_ldl_bfgs<double, n> bfgs(n);

    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);

    for(int k = 0; k < 100; ++k)
    {
        Eigen::Vector<double, n> s, y;
        for(int i = 0; i < n; ++i)
        {
            s[i] = nd(rng);
            y[i] = nd(rng);
        }
        if(s.dot(y) <= 0.0) y = -y;
        bfgs.push(s, y);
    }

    Eigen::LLT<Eigen::Matrix<double, n, n>> llt(bfgs.hessian());
    REQUIRE(llt.info() == Eigen::Success);

    // Symmetry to roundoff.
    const auto& B = bfgs.hessian();
    REQUIRE(B.isApprox(B.transpose(), 1e-10));
}

TEST_CASE("dense_ldl_bfgs dynamic-N constructor", "[dense_ldl_bfgs]")
{
    // Dynamic-dimension instantiation must work the same as fixed-N.
    dense_ldl_bfgs<double, argmin::dynamic_dimension> bfgs(4);

    Eigen::Vector<double, Eigen::Dynamic> s(4), y(4);
    s << 1.0, 0.5, -0.3, 0.7;
    y << 2.0, 1.1,  0.4, 1.5;
    bfgs.push(s, y);

    Eigen::Vector<double, Eigen::Dynamic> Bs = bfgs.hessian() * s;
    REQUIRE(Bs.isApprox(y, 1e-12));
}
