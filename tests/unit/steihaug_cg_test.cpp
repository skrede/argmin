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

// Helper: build a closure that multiplies by a dense Eigen matrix B,
// writing the product into the caller-provided output Ref. The
// hessian-op signature is the output-parameter form steihaug_cg
// consumes (no return-by-value temporary on the inner CG iteration).
struct dense_hessian_op
{
    Eigen::Matrix2d B;
    void operator()(const Eigen::Ref<const Eigen::Vector2d>& v,
                    Eigen::Ref<Eigen::Vector2d> out) const
    {
        out.noalias() = B * v;
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

// ─── Kernel pin: model decrease under box bounds ───────────────────
//
// Reference: Conn, Gould, Toint 2000 "Trust-Region Methods" Chapter 7
//            (Steihaug-Toint: the truncated iterate stays on the CG
//             descent ray up to the blocking face, which preserves the
//             closed-form model-decrease guarantee). A post-step box
//             PROJECTION (finding 16) pushes the exit point OFF the
//             descent ray, forfeiting that guarantee.
//
// Instance (hand-derived): g = (1, -1), B = [[1, 0.9], [0.9, 1]],
// upper box on coordinate 1 at 0.5 (lower unbounded), delta = 1. The
// initial CG direction is d0 = -g = (-1, 1). The unconstrained CG
// minimizer along d0 is at tau = 10; the trust-region sphere is at
// tau = 1/sqrt(2) ~ 0.707; the box face on coordinate 1 is at
// tau = 0.5. Truncate-before-step stops at tau = 0.5, giving the
// on-ray exit p = 0.5 * d0 = (-0.5, 0.5) with q(p) = -0.975 <= 0.
//
// Pre-fix code steps to the sphere first (p = (-0.7071, 0.7071)) then
// clips coordinate 1 to 0.5, yielding p = (-0.7071, 0.5) -- OFF the
// d0 ray (the coordinate-0 leg overran the blocking face). Recorded
// pre-fix red: exit not colinear with d0 (cross-product -0.2071).
TEST_CASE("steihaug_cg box-active exit stays on the CG descent ray",
          "[kernel-pin][steihaug][model-decrease]")
{
    Eigen::Vector2d g;
    g << 1.0, -1.0;
    Eigen::Matrix2d B;
    B << 1.0, 0.9,
         0.9, 1.0;

    Eigen::Vector2d lower;
    lower << -kInf, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, 0.5;

    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    steihaug_cg<double, 2>(
        g, op, /*delta=*/1.0, /*eps=*/1e-12,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    const Eigen::Vector2d d0 = -g;
    const double cross = p_out[0] * d0[1] - p_out[1] * d0[0];
    const double q = g.dot(p_out) + 0.5 * p_out.dot(B * p_out);
    INFO("p_out = (" << p_out[0] << ", " << p_out[1] << ")  cross = "
         << cross << "  q = " << q);
    // On the descent ray: exit colinear with d0.
    CHECK(cross == Approx(0.0).margin(1e-12));
    // Model decrease from p = 0 (q(0) = 0).
    CHECK(q <= 1e-12);
    // Box respected.
    CHECK(p_out[1] <= 0.5 + 1e-12);
}

TEST_CASE("steihaug_cg TR boundary hit at CG iteration >= 2",
          "[steihaug_cg][boundary][cross-terms]")
{
    // Ill-conditioned SPD B = diag(1, 10) with an off-axis gradient so
    // the first CG step lands strictly inside the trust region and the
    // SECOND step crosses the boundary -- exercising the tau-quadratic
    // cross-terms (p . d != 0) that a first-iteration boundary hit
    // never touches.
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
        g, op, /*delta=*/0.9, /*eps=*/1e-14,
        lower, upper, /*max_iter=*/50,
        p_out, r_buf, d_buf, Bd_buf);

    const double q = g.dot(p_out) + 0.5 * p_out.dot(B * p_out);
    INFO("status boundary=" << (status == cg_exit_status::boundary)
         << "  ||p||=" << p_out.norm() << "  q=" << q);
    CHECK(status == cg_exit_status::boundary);
    CHECK(p_out.norm() == Approx(0.9).margin(1e-10));
    CHECK(q <= 1e-12);
}

TEST_CASE("steihaug_cg negative-curvature exit decreases the model",
          "[steihaug_cg][negative_curvature][sign]")
{
    // B = diag(1, -1) indefinite; g = (0, 1) seeds d0 = (0, -1) into
    // the negative-curvature subspace. The exit must follow the
    // DESCENT sense of d0 (not its negation), so q(p_exit) < 0.
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

    const double q = g.dot(p_out) + 0.5 * p_out.dot(B * p_out);
    INFO("p_out = (" << p_out[0] << ", " << p_out[1] << ")  q = " << q);
    CHECK(status == cg_exit_status::negative_curvature);
    CHECK(q < 0.0);
}

TEST_CASE("steihaug_cg truncates at the blocking box face",
          "[steihaug_cg][bounds]")
{
    // q(p) = g^T p + 0.5 p^T p with g = [2, 0] and B = I has
    // unconstrained minimizer p* = -g = [-2, 0]. With the displaced
    // lower bound lower_displaced = [-1, -inf], the box-constrained
    // minimizer is at p = [-1, 0].
    //
    // With truncate-before-step handling, the first CG direction is
    // d0 = -g = [-2, 0]; the unconstrained step length is alpha = 1
    // (reaching [-2, 0]) but the box face on coordinate 0 is at
    // tau = 0.5, so the step is truncated at the face BEFORE it is
    // applied, giving p = 0.5 * d0 = [-1, 0] and a boundary-class exit.
    // No post-step projection is performed, so the returned iterate
    // lies exactly on the box face along the descent ray.
    //
    // Reference: scipy qp_subproblem.projected_cg (box truncation);
    //            Conn, Gould, Toint 2000 Chapter 7 (model decrease on
    //            truncation).
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

    // Lower bound respected exactly; iterate on the box face.
    CHECK(p_out[0] >= -1.0 - 1e-12);
    CHECK(p_out[0] == Approx(-1.0).margin(1e-12));
    CHECK(p_out[1] == Approx(0.0).margin(1e-12));
    CHECK(status == cg_exit_status::boundary);
}
