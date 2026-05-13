#include "argmin/detail/steihaug_cg.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using argmin::detail::cg_exit_status;
using argmin::detail::steihaug_cg;

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

// Helper: build a closure that multiplies by a dense Eigen matrix B
// and returns a freshly-allocated Eigen::Vector. The hessian-op
// signature accepts a const Eigen::Ref to the search direction.
struct dense_hessian_op
{
    Eigen::Matrix2d B;
    Eigen::Vector2d operator()(
        const Eigen::Ref<const Eigen::Vector2d>& v) const
    {
        return (B * v).eval();
    }
};

}

TEST_CASE("steihaug_cg unconstrained quadratic exits via forcing tolerance",
          "[steihaug_cg][forcing]")
{
    // q(p) = g^T p + 0.5 p^T B p with B = I, g = [1, 2].
    // Unconstrained minimizer is p* = -B^{-1} g = [-1, -2], norm = sqrt(5) ~ 2.236 < 100.
    Eigen::Vector2d g;
    g << 1.0, 2.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::Vector2d lower;
    lower << -kInf, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/100.0, /*eps=*/1e-8,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    CHECK(status == cg_exit_status::forcing);
    CHECK(p_out[0] == Approx(-1.0).margin(1e-10));
    CHECK(p_out[1] == Approx(-2.0).margin(1e-10));
}

TEST_CASE("steihaug_cg negative-curvature direction returns boundary step",
          "[steihaug_cg][negative_curvature]")
{
    // B = diag(1, -1) is indefinite. Seed g into the negative-curvature
    // subspace directly: with g = [0, 1] the initial direction d_0 =
    // [0, -1] has curvature d_0^T B d_0 = -1 < 0, so the helper takes
    // the negative-curvature exit on the first iteration and returns
    // a step on the TR boundary along d_0.
    Eigen::Vector2d g;
    g << 0.0, 1.0;
    Eigen::Matrix2d B;
    B << 1.0, 0.0,
         0.0, -1.0;

    Eigen::Vector2d lower;
    lower << -kInf, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/1.0, /*eps=*/1e-12,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    CHECK(status == cg_exit_status::negative_curvature);
    CHECK(p_out.norm() == Approx(1.0).margin(1e-12));
}

TEST_CASE("steihaug_cg TR-boundary truncation", "[steihaug_cg][boundary]")
{
    // B = 0.01 * I is SPD but ill-scaled: unconstrained minimizer is
    // p* = -100 * g, far outside any reasonable TR. With g = [1, 0]
    // and delta = 1, the first CG step lands at p = -100*[1,0] which
    // is truncated to the TR boundary along d_0 = -g.
    Eigen::Vector2d g;
    g << 1.0, 0.0;
    Eigen::Matrix2d B = 0.01 * Eigen::Matrix2d::Identity();

    Eigen::Vector2d lower;
    lower << -kInf, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/1.0, /*eps=*/1e-12,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    CHECK(status == cg_exit_status::boundary);
    CHECK(p_out.norm() == Approx(1.0).margin(1e-12));
    // Direction must align with -g (positive descent direction along x).
    CHECK(p_out[0] == Approx(-1.0).margin(1e-12));
    CHECK(p_out[1] == Approx(0.0).margin(1e-12));
}

TEST_CASE("steihaug_cg max_iterations exit", "[steihaug_cg][max_iter]")
{
    // Pick a 2-D quadratic whose CG needs more than one step. Using a
    // moderately ill-conditioned diagonal B = diag(1, 10) with off-axis
    // gradient g = [1, 1]: CG converges in at most 2 iterations on a
    // 2-D SPD system but the first iterate is not yet the minimizer.
    // With max_iter = 1, the helper exhausts the budget and must return
    // max_iterations with a finite step inside the TR.
    Eigen::Vector2d g;
    g << 1.0, 1.0;
    Eigen::Matrix2d B;
    B << 1.0, 0.0,
         0.0, 10.0;

    Eigen::Vector2d lower;
    lower << -kInf, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/100.0, /*eps=*/1e-14,
        lower, upper, /*max_iter=*/1,
        p_out, r_buf, d_buf, Bd_buf);

    CHECK(status == cg_exit_status::max_iterations);
    CHECK(std::isfinite(p_out[0]));
    CHECK(std::isfinite(p_out[1]));
    CHECK(p_out.norm() <= 100.0 + 1e-12);
}

TEST_CASE("steihaug_cg restart on lower-bound activation",
          "[steihaug_cg][bounds]")
{
    // q(p) = g^T p + 0.5 p^T p with g = [2, 0] and B = I has
    // unconstrained minimizer p* = -g = [-2, 0]. With the displaced
    // lower bound lower_displaced = [-1, -inf] (i.e., x_lower - z_k
    // = -1 - 0 = -1), the constrained minimizer is at p = [-1, 0].
    //
    // The first CG iterate is p_trial = [-2, 0] (alpha = 1, d = -g);
    // detail::project clips this to p_proj = [-1, 0]; the Lin-More
    // restart recomputes the residual at the projected iterate. The
    // residual at p_proj is g + B p_proj = [2, 0] + [-1, 0] = [1, 0]
    // — non-zero because the Lagrange multiplier for the active
    // constraint is +1; without an explicit free-variable subspace
    // restriction the unconstrained-CG forcing criterion cannot
    // recognize the active-face KKT condition. The helper therefore
    // cycles in the active-set face and exits via max_iterations
    // with the projected iterate. The load-bearing behavior the
    // outer ratio test relies on is that p respects the bound; that
    // is what is asserted here.
    //
    // Reference: Lin and More 1999 §3 — the free-variable restart
    //            documented here is "re-seed the residual at the
    //            projected iterate"; the stronger "restrict CG to
    //            the free face" variant is intentionally out of
    //            scope for this helper.
    Eigen::Vector2d g;
    g << 2.0, 0.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::Vector2d lower;
    lower << -1.0, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/10.0, /*eps=*/1e-12,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    // Lower bound must be respected to within floating-point.
    CHECK(p_out[0] >= -1.0 - 1e-12);
    CHECK(p_out[0] == Approx(-1.0).margin(1e-10));
    CHECK(p_out[1] == Approx(0.0).margin(1e-10));
    // Either forcing (rare; only if the active-face residual happens
    // to be smaller than eps*||r_0||) or max_iterations is acceptable
    // for this geometry without free-variable subspace restriction.
    CHECK((status == cg_exit_status::forcing
        || status == cg_exit_status::max_iterations));
}
