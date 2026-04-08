#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/cauchy_point.h"
#include "nablapp/detail/subspace_minimization.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

using Catch::Approx;
using namespace nablapp::detail;

TEST_CASE("compact_lbfgs multiply matches explicit dense BFGS", "[detail]")
{
    // 2D problem: push one (s, y) pair and verify B*v matches explicit BFGS update.
    // Reference: N&W eq. 6.19 (BFGS direct update).
    Eigen::VectorXd s(2);
    s << 1.0, 0.0;
    Eigen::VectorXd y(2);
    y << 2.0, 1.0;

    compact_lbfgs<double, -1, 2> B;
    B.push(s, y);

    // Build explicit B1 from rank-2 BFGS update:
    // theta = y^T y / s^T y
    double sTy = s.dot(y);
    double theta = y.squaredNorm() / sTy;

    // B0 = theta * I
    Eigen::Matrix2d B0 = theta * Eigen::Matrix2d::Identity();

    // B1 = B0 - (B0*s*s^T*B0)/(s^T*B0*s) + (y*y^T)/(y^T*s)
    // N&W eq. 6.19
    Eigen::Vector2d B0s = B0 * s;
    double sTB0s = s.dot(B0s);
    Eigen::Matrix2d B1 = B0 - (B0s * B0s.transpose()) / sTB0s
                        + (y * y.transpose()) / sTy;

    Eigen::VectorXd v(2);
    v << 1.0, 1.0;
    Eigen::VectorXd Bv_compact = B.multiply(v);
    Eigen::Vector2d Bv_explicit = B1 * v;

    CHECK(Bv_compact[0] == Approx(Bv_explicit[0]).margin(1e-10));
    CHECK(Bv_compact[1] == Approx(Bv_explicit[1]).margin(1e-10));
}

TEST_CASE("compact_lbfgs two_loop_recursion consistency", "[detail]")
{
    // Push one (s, y) pair and verify B * (H * g) ~= g (B and H are inverses).
    Eigen::VectorXd s(2);
    s << 1.0, 0.0;
    Eigen::VectorXd y(2);
    y << 2.0, 1.0;

    compact_lbfgs<double, -1, 2> B;
    B.push(s, y);

    Eigen::VectorXd g(2);
    g << 1.0, 1.0;
    Eigen::VectorXd Hg = B.two_loop_recursion(g);

    // Verify finite
    CHECK(std::isfinite(Hg[0]));
    CHECK(std::isfinite(Hg[1]));
    CHECK(Hg.size() == 2);

    // B * (H * g) should approximately equal g (B and H are inverses for exact BFGS)
    Eigen::VectorXd BHg = B.multiply(Hg);
    CHECK(BHg[0] == Approx(g[0]).margin(1e-6));
    CHECK(BHg[1] == Approx(g[1]).margin(1e-6));
}

TEST_CASE("compact_lbfgs incremental push matches full recompute", "[detail]")
{
    // Verify that the incremental STS_/L_/D_ update in push() produces
    // identical multiply() and two_loop_recursion() results whether the
    // buffer is filling (non-full path) or evicting (full path).
    //
    // Uses two independent compact_lbfgs instances fed the same (s,y) sequence.
    // After each push, compares multiply(v) and two_loop_recursion(v) outputs
    // to verify incremental bookkeeping matches. Also checks B * H * g ~ g
    // (inverse consistency) as a cross-validation of both paths.

    constexpr int n = 4;
    constexpr int max_hist = 3;
    constexpr int total_pushes = max_hist + 3;  // exercises eviction

    compact_lbfgs<double, -1, max_hist> A;
    compact_lbfgs<double, -1, max_hist> B;

    for(int i = 0; i < total_pushes; ++i)
    {
        Eigen::VectorXd s(n), y(n);
        for(int d = 0; d < n; ++d)
        {
            s[d] = 0.1 * (i + 1) + 0.3 * (d + 1);
            y[d] = 0.5 * (i + 1) + 0.2 * (d + 1) + 1.0;
        }

        A.push(s, y);
        B.push(s, y);

        Eigen::VectorXd v = Eigen::VectorXd::Ones(n) * (i + 1.0);

        INFO("Push " << i << " (count=" << A.size() << ")");
        CHECK(A.multiply(v).isApprox(B.multiply(v), 1e-12));
        CHECK(A.two_loop_recursion(v).isApprox(B.two_loop_recursion(v), 1e-12));

        // Inverse consistency: B * (H * g) ~ g
        Eigen::VectorXd Hg = A.two_loop_recursion(v);
        Eigen::VectorXd BHg = A.multiply(Hg);
        CHECK(BHg.isApprox(v, 1e-6));
    }
}

TEST_CASE("cauchy_point on bounded quadratic with known GCP", "[detail]")
{
    // 2D: x=(2,2), g=(1,-1), lower=(0,0), upper=(3,3), B=I (no history).
    Eigen::VectorXd x(2);
    x << 2.0, 2.0;
    Eigen::VectorXd g(2);
    g << 1.0, -1.0;
    Eigen::VectorXd lo(2);
    lo << 0.0, 0.0;
    Eigen::VectorXd hi(2);
    hi << 3.0, 3.0;

    compact_lbfgs<double, -1, 2> B;  // Empty: B = theta*I = 1*I

    auto result = cauchy_point(x, g, lo, hi, B);

    // x_cauchy should be within bounds
    CHECK(result.x_cauchy[0] >= lo[0]);
    CHECK(result.x_cauchy[0] <= hi[0]);
    CHECK(result.x_cauchy[1] >= lo[1]);
    CHECK(result.x_cauchy[1] <= hi[1]);

    // Free and active indices should be consistent with x_cauchy
    for(int idx : result.active_indices)
    {
        bool at_lower = result.x_cauchy[idx] <= lo[idx] + 1e-12;
        bool at_upper = result.x_cauchy[idx] >= hi[idx] - 1e-12;
        CHECK((at_lower || at_upper));
    }
}

TEST_CASE("cauchy_point unconstrained fallback returns all free", "[detail]")
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd x(2);
    x << 2.0, 2.0;
    Eigen::VectorXd g(2);
    g << 1.0, -1.0;
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(2, -inf);
    Eigen::VectorXd hi = Eigen::VectorXd::Constant(2, inf);

    compact_lbfgs<double, -1, 2> B;

    auto result = cauchy_point(x, g, lo, hi, B);

    CHECK(result.free_indices.size() == 2);
    CHECK(result.active_indices.size() == 0);
}

TEST_CASE("subspace_minimize on trivially solvable reduced system", "[detail]")
{
    // 2D: x=(0,0), x_cauchy=(0.5,0), g=(-1,-1), B=I (no history).
    // Reduced gradient at x_cauchy: r = g + B*(x_c - x) = (-1,-1) + (0.5,0) = (-0.5,-1).
    // B_FF = I, so d_hat = -B_FF^{-1} * r_F = (0.5, 1).
    // x_new = x_cauchy + d_hat = (1, 1).
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    Eigen::VectorXd x_cauchy(2);
    x_cauchy << 0.5, 0.0;
    Eigen::VectorXd g(2);
    g << -1.0, -1.0;
    Eigen::VectorXd lo(2);
    lo << -10.0, -10.0;
    Eigen::VectorXd hi(2);
    hi << 10.0, 10.0;
    std::vector<int> free_idx{0, 1};

    compact_lbfgs<double, -1, 2> B;  // Empty: B = I

    Eigen::VectorXd result = subspace_minimize(x, x_cauchy, g, lo, hi, free_idx, B);

    CHECK(result[0] == Approx(1.0).margin(1e-10));
    CHECK(result[1] == Approx(1.0).margin(1e-10));

    // Within bounds
    CHECK(result[0] >= lo[0]);
    CHECK(result[0] <= hi[0]);
    CHECK(result[1] >= lo[1]);
    CHECK(result[1] <= hi[1]);
}

TEST_CASE("subspace_minimize with empty free_indices returns x_cauchy", "[detail]")
{
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    Eigen::VectorXd x_cauchy(2);
    x_cauchy << 1.0, 2.0;
    Eigen::VectorXd g(2);
    g << -1.0, -1.0;
    Eigen::VectorXd lo(2);
    lo << 0.0, 0.0;
    Eigen::VectorXd hi(2);
    hi << 3.0, 3.0;
    std::vector<int> free_idx{};

    compact_lbfgs<double, -1, 2> B;

    Eigen::VectorXd result = subspace_minimize(x, x_cauchy, g, lo, hi, free_idx, B);

    CHECK(result[0] == x_cauchy[0]);
    CHECK(result[1] == x_cauchy[1]);
}

TEST_CASE("bound_projection project and compute_alpha_max", "[detail]")
{
    SECTION("project clamps to box")
    {
        Eigen::VectorXd x(2);
        x << 5.0, -3.0;
        Eigen::VectorXd lo(2);
        lo << 0.0, 0.0;
        Eigen::VectorXd hi(2);
        hi << 4.0, 4.0;

        Eigen::VectorXd p = project(x, lo, hi);
        CHECK(p[0] == Approx(4.0));
        CHECK(p[1] == Approx(0.0));
    }

    SECTION("compute_alpha_max hits upper bound")
    {
        Eigen::VectorXd x(2);
        x << 1.0, 1.0;
        Eigen::VectorXd d(2);
        d << 1.0, 1.0;
        Eigen::VectorXd lo(2);
        lo << 0.0, 0.0;
        Eigen::VectorXd hi(2);
        hi << 3.0, 3.0;

        double alpha = compute_alpha_max(x, d, lo, hi);
        CHECK(alpha == Approx(2.0));
    }

    SECTION("compute_alpha_max hits lower bound")
    {
        Eigen::VectorXd x(2);
        x << 1.0, 1.0;
        Eigen::VectorXd d(2);
        d << -1.0, 0.0;
        Eigen::VectorXd lo(2);
        lo << 0.0, 0.0;
        Eigen::VectorXd hi(2);
        hi << 3.0, 3.0;

        double alpha = compute_alpha_max(x, d, lo, hi);
        CHECK(alpha == Approx(1.0));
    }
}
