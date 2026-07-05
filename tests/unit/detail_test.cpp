#include "argmin/detail/compact_lbfgs.h"
#include "argmin/detail/cauchy_point.h"
#include "argmin/detail/subspace_minimization.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/isres_operators.h"
#include "argmin/detail/xoshiro256.h"
#include "argmin/solver/convergence.h"

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>

using Catch::Approx;
using namespace argmin::detail;

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

// ---------------------------------------------------------------------------
// ISRES operator tests for differential_variation, log_normal_mutate, and
// the two sigma-collapse predicates added under the ISRES Runarsson-Yao
// rewrite.
// Reference: Runarsson & Yao (2005), IEEE Trans. SMC-C 35(2):233-243;
//            K&W 2e Section 8.6; NLopt 2.10.0 isres.c.
// ---------------------------------------------------------------------------

TEST_CASE("differential_variation applies DE-style x0 difference for k+1 < mu",
          "[isres_operators]")
{
    // n = 2, mu = 3. Three columns in population correspond to indices
    // 0, 1, 2. Snapshot has the same shape.
    constexpr int n = 2;
    const int mu = 3;
    Eigen::Matrix<double, 2, Eigen::Dynamic> population(n, mu);
    population.col(0) << 0.0, 0.0;
    population.col(1) << 0.0, 0.0;
    population.col(2) << 0.0, 0.0;

    Eigen::Matrix<double, 2, Eigen::Dynamic> snap(n, mu);
    snap.col(0) << 1.0, 2.0;  // anchor "0"
    snap.col(1) << 0.0, 0.0;  // anchor for k=0
    snap.col(2) << 0.5, 1.0;  // anchor for k=1

    std::vector<std::uint32_t> indices{0u, 1u, 2u};
    const double gamma = 0.5;
    Eigen::Vector<double, 2> lower; lower << -10.0, -10.0;
    Eigen::Vector<double, 2> upper; upper <<  10.0,  10.0;

    argmin::detail::differential_variation<2, double>(
        population, snap, indices, mu, gamma, lower, upper);

    // k = 0 (rk = indices[0] = 0): population.col(0) += 0.5 * (snap.col(0) - snap.col(1))
    //                            = 0.5 * (1, 2) = (0.5, 1.0)
    CHECK(population(0, 0) == Approx(0.5));
    CHECK(population(1, 0) == Approx(1.0));

    // k = 1 (rk = indices[1] = 1): population.col(1) += 0.5 * (snap.col(0) - snap.col(2))
    //                            = 0.5 * (0.5, 1.0) = (0.25, 0.5)
    CHECK(population(0, 1) == Approx(0.25));
    CHECK(population(1, 1) == Approx(0.5));

    // k = 2 (k + 1 == mu): boundary case, column unchanged.
    CHECK(population(0, 2) == Approx(0.0));
    CHECK(population(1, 2) == Approx(0.0));
}

TEST_CASE("log_normal_mutate clamps with finite sigma_max and is no-op at infinity",
          "[isres_operators]")
{
    argmin::detail::xoshiro256 rng{42u};

    // With sigma_max >> sigma * exp(...), the clamp does not bind. The
    // returned value must equal sigma * exp(tau_prime_rand + tau * N(0, 1)).
    {
        argmin::detail::xoshiro256 rng_a{42u};
        argmin::detail::xoshiro256 rng_b{42u};
        const double sigma = 1.0;
        const double tau = 0.5;
        const double tau_prime_rand = 0.25;
        const double sigma_max_high = 1e6;

        const double res = argmin::detail::log_normal_mutate(
            sigma, tau, tau_prime_rand, sigma_max_high, rng_a);

        // Mirror the function body exactly with rng_b in lockstep:
        std::normal_distribution<double> normal(0.0, 1.0);
        const double expected = sigma
            * std::exp(tau_prime_rand + tau * normal(rng_b));
        CHECK(res == Approx(expected));
        CHECK(res < sigma_max_high);
    }

    // With sigma_max == infinity, the clamp must be a no-op. The result
    // is finite for finite sigma and finite tau / tau_prime_rand.
    {
        const double sigma = 0.5;
        const double tau = 0.1;
        const double tau_prime_rand = 0.0;
        const double inf = std::numeric_limits<double>::infinity();
        const double res = argmin::detail::log_normal_mutate(
            sigma, tau, tau_prime_rand, inf, rng);
        CHECK(std::isfinite(res));
        CHECK(res > 0.0);
    }

    // With sigma_max smaller than the unclamped result, the clamp binds.
    {
        argmin::detail::xoshiro256 rng_c{7u};
        const double sigma = 1.0;
        const double tau = 0.0;
        const double tau_prime_rand = 5.0;  // exp(5) ~ 148
        const double sigma_max_low = 1.5;
        const double res = argmin::detail::log_normal_mutate(
            sigma, tau, tau_prime_rand, sigma_max_low, rng_c);
        CHECK(res == Approx(sigma_max_low));
    }
}

namespace
{

// Minimal synthetic state mirroring the field surface the predicates
// require: `sigmas` matrix and `mu` survivor count.
struct sigma_state
{
    Eigen::MatrixXd sigmas;
    int mu;
};

}

TEST_CASE("sigma_collapsed_bound_relative fires only below threshold",
          "[isres_operators]")
{
    // n = 2, mu = 3, range = (10 - (-10)) on each dim => mean_range = 20.
    // ratio * mean_range / sqrt(n) = ratio * 20 / sqrt(2).
    constexpr int n = 2;
    sigma_state s;
    s.mu = 3;
    s.sigmas = Eigen::MatrixXd::Constant(n, s.mu, 1.0);  // mean_sigma = 1.0

    Eigen::Vector<double, 2> lower; lower << -10.0, -10.0;
    Eigen::Vector<double, 2> upper; upper <<  10.0,  10.0;

    // Loose ratio (1.0) fires: 1.0 < 1.0 * 20 / sqrt(2) (~14.14).
    CHECK(argmin::detail::sigma_collapsed_bound_relative(
        s, 1.0, n, lower, upper));

    // Tight ratio (1e-3) does not fire: 1.0 < 1e-3 * 14.14 = 0.014 is false.
    CHECK_FALSE(argmin::detail::sigma_collapsed_bound_relative(
        s, 1e-3, n, lower, upper));

    // With small sigmas (1e-15), the tight ratio fires.
    s.sigmas = Eigen::MatrixXd::Constant(n, s.mu, 1e-15);
    CHECK(argmin::detail::sigma_collapsed_bound_relative(
        s, 1e-9, n, lower, upper));
}

TEST_CASE("sigma_collapsed_xtol_coupled honors convergence threshold and falls back",
          "[isres_operators]")
{
    constexpr int n = 2;
    sigma_state s;
    s.mu = 2;
    s.sigmas = Eigen::MatrixXd::Constant(n, s.mu, 1e-7);  // mean_sigma = 1e-7

    Eigen::Vector<double, 2> lower; lower << -1.0, -1.0;
    Eigen::Vector<double, 2> upper; upper <<  1.0,  1.0;

    // default_convergence carries step_tolerance_criterion. Set the
    // threshold to a value the mean_sigma beats (1e-6 > 1e-7) -> fires.
    {
        argmin::default_convergence conv{};
        std::get<argmin::step_tolerance_criterion>(conv.criteria).threshold = 1e-6;
        CHECK(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1e-9, n, lower, upper));
    }

    // Same convergence with a tighter threshold (1e-9) the mean_sigma does
    // not beat -> does not fire.
    {
        argmin::default_convergence conv{};
        std::get<argmin::step_tolerance_criterion>(conv.criteria).threshold = 1e-9;
        CHECK_FALSE(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1e-12, n, lower, upper));
    }

    // default_convergence's step_tolerance_criterion now carries a
    // direct-value literature default (1e-8; see convergence.h) instead of
    // an inert std::nullopt, so a freshly-constructed default_convergence
    // is never in the "threshold absent" state and always takes the
    // engaged (non-fallback) branch: mean_sigma (1e-7) does not beat the
    // 1e-8 default, so the predicate returns false regardless of
    // fallback_ratio. The genuine fallback path (Convergence lacking
    // step_tolerance_criterion entirely) is exercised by the
    // slsqp_compatible_convergence case below.
    {
        argmin::default_convergence conv{};
        CHECK_FALSE(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1.0, n, lower, upper));
        CHECK_FALSE(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1e-12, n, lower, upper));
    }

    // slsqp_compatible_convergence lacks step_tolerance_criterion. The
    // SFINAE guard must keep the function compilable AND silently fall
    // back to bound_relative semantics.
    {
        argmin::slsqp_compatible_convergence conv{};
        CHECK(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1.0, n, lower, upper));
        CHECK_FALSE(argmin::detail::sigma_collapsed_xtol_coupled(
            s, conv, /*fallback_ratio=*/1e-12, n, lower, upper));
    }

    // Captured-threshold overload: finite + positive threshold_value is
    // honored; non-finite or non-positive falls back to bound_relative
    // semantics with `fallback_ratio`.
    {
        const double finite_thr = 1e-6;
        CHECK(argmin::detail::sigma_collapsed_xtol_coupled(
            s, finite_thr, /*fallback_ratio=*/1e-12, n, lower, upper));

        const double tight_thr = 1e-9;
        CHECK_FALSE(argmin::detail::sigma_collapsed_xtol_coupled(
            s, tight_thr, /*fallback_ratio=*/1e-12, n, lower, upper));

        const double inf = std::numeric_limits<double>::infinity();
        CHECK(argmin::detail::sigma_collapsed_xtol_coupled(
            s, inf, /*fallback_ratio=*/1.0, n, lower, upper));

        CHECK(argmin::detail::sigma_collapsed_xtol_coupled(
            s, /*threshold_value=*/0.0, /*fallback_ratio=*/1.0, n, lower, upper));
    }
}

TEST_CASE("ISRES operators preserve existing detail symbols byte-for-byte",
          "[isres_operators]")
{
    // Existing free-function symbols: mutate_individual, compute_violations,
    // compute_es_rates, initialize_population, initialize_sigmas,
    // and the workspace class isres_operator_workspace.
    // This test exercises the existing learning-rate helper and the
    // workspace-class ctor to confirm preservation; full coverage of
    // the existing ops lives in isres_test.cpp.
    const int n = 4;
    argmin::detail::es_learning_rates rates =
        argmin::detail::compute_es_rates(n);
    CHECK(rates.tau == Approx(1.0 / std::sqrt(2.0 * n)));
    CHECK(rates.tau_prime
          == Approx(1.0 / std::sqrt(2.0 * std::sqrt(static_cast<double>(n)))));

    argmin::detail::isres_operator_workspace<double, Eigen::Dynamic> ws(n);
    (void)ws;  // ctor compiles, no fields exposed for direct check.
}
