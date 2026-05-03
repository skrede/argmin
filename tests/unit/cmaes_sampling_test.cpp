// CMA-ES sampling variants (detail/alternative/) -- chi-square
// goodness-of-fit and tail-path correctness tests.
//
// The chi-square outcome is INFORMATIONAL per CONTEXT D-06; surfacing
// distribution-shape regressions in future code changes (e.g., a typo
// in lookup-table indices). At first introduction here the
// implementation is required to be correct, so the CHECK lines are
// written as gating; a future change that deliberately re-shapes the
// distribution should re-baseline rather than preserve bias (per
// `feedback_correctness_over_compat`).
//
// References:
//   Marsaglia & Bray (1964), SIAM Review 6(3) -- polar method.
//   Marsaglia & Tsang (2000), J. Statistical Software 5(8) -- Ziggurat.
//   Doornik (2005) -- Ziggurat tail-handling refinements.
//   Knuth, TAOCP Vol. 2, §3.3.1 -- chi-square goodness-of-fit.

#include "argmin/detail/alternative/cmaes_sampling_marsaglia.h"
#include "argmin/detail/alternative/cmaes_sampling_ziggurat.h"
#include "argmin/detail/xoshiro256.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace
{

// 20 bins on [-3.5, 3.5] sigma (per RESEARCH §"Recommended choice: chi-square").
constexpr int K_BINS = 20;
constexpr double X_LOW = -3.5;
constexpr double X_HIGH = 3.5;
constexpr int N_SAMPLES = 100'000;
// 99.9% confidence threshold for K-1 = 19 dof per Knuth TAOCP Vol. 2 §3.3.1
// table: chi^2_{0.001, 19} ~ 43.82.
constexpr double CHI2_99_9_PCT_19DOF = 43.82;

int bin_index(double z)
{
    if(z < X_LOW || z >= X_HIGH) return -1;
    const double width = (X_HIGH - X_LOW) / K_BINS;
    const int idx = static_cast<int>((z - X_LOW) / width);
    return (idx < 0 || idx >= K_BINS) ? -1 : idx;
}

// Probability mass in bin k under standard normal, computed via erf.
double normal_bin_prob(int k)
{
    const double width = (X_HIGH - X_LOW) / K_BINS;
    const double lo = X_LOW + k * width;
    const double hi = lo + width;
    const double sqrt2 = std::sqrt(2.0);
    return 0.5 * (std::erf(hi / sqrt2) - std::erf(lo / sqrt2));
}

double chi_square_statistic(const std::array<int, K_BINS>& counts, int n)
{
    double chi2 = 0.0;
    for(int k = 0; k < K_BINS; ++k)
    {
        const double e = normal_bin_prob(k) * static_cast<double>(n);
        const double diff = static_cast<double>(counts[k]) - e;
        chi2 += diff * diff / e;
    }
    return chi2;
}

}

TEST_CASE("cmaes_sampling: marsaglia_polar passes chi-square goodness-of-fit",
          "[cmaes_sampling]")
{
    argmin::detail::xoshiro256 rng{42u};

    std::array<int, K_BINS> counts{};
    int outliers = 0;
    for(int i = 0; i < N_SAMPLES; ++i)
    {
        const double z = argmin::detail::alternative::marsaglia::marsaglia_normal<double>(rng);
        const int bin = bin_index(z);
        if(bin >= 0) ++counts[bin];
        else ++outliers;
    }

    const int in_range = std::accumulate(counts.begin(), counts.end(), 0);
    const double chi2 = chi_square_statistic(counts, in_range);

    INFO("chi^2 statistic: " << chi2 << " (threshold for 99.9% confidence at 19 dof: "
         << CHI2_99_9_PCT_19DOF << ")");
    INFO("outliers (|z| >= 3.5): " << outliers);
    CHECK(chi2 < CHI2_99_9_PCT_19DOF);
    // Outliers should be ~ 2 * (1 - Phi(3.5)) * N ~ 47 in 100k draws;
    // a wide [10, 200] band catches a gross distribution corruption
    // without false-positiving on run-to-run variation.
    CHECK(outliers > 10);
    CHECK(outliers < 200);
}

TEST_CASE("cmaes_sampling: ziggurat_normal passes chi-square goodness-of-fit",
          "[cmaes_sampling]")
{
    argmin::detail::xoshiro256 rng{42u};

    std::array<int, K_BINS> counts{};
    int outliers = 0;
    for(int i = 0; i < N_SAMPLES; ++i)
    {
        const double z = argmin::detail::alternative::ziggurat::ziggurat_normal<double>(rng);
        const int bin = bin_index(z);
        if(bin >= 0) ++counts[bin];
        else ++outliers;
    }

    const int in_range = std::accumulate(counts.begin(), counts.end(), 0);
    const double chi2 = chi_square_statistic(counts, in_range);

    INFO("chi^2 statistic: " << chi2 << " (threshold: " << CHI2_99_9_PCT_19DOF << ")");
    INFO("outliers (|z| >= 3.5): " << outliers);
    CHECK(chi2 < CHI2_99_9_PCT_19DOF);
    CHECK(outliers > 10);
    CHECK(outliers < 200);
}

TEST_CASE("cmaes_sampling: ziggurat_normal tail draws are finite and nonzero",
          "[cmaes_sampling]")
{
    // Per RESEARCH "Caveats per feedback_correctness_over_compat": the
    // Ziggurat tail-handling has historically had edge-case bugs in
    // published implementations. This test exercises tail draws
    // (|x| > 3.5) and verifies they are finite + nonzero.
    argmin::detail::xoshiro256 rng{42u};

    int tail_count = 0;
    int positive_tail = 0;
    int negative_tail = 0;
    for(int i = 0; i < 1'000'000; ++i)
    {
        const double z = argmin::detail::alternative::ziggurat::ziggurat_normal<double>(rng);
        if(std::abs(z) > 3.5)
        {
            ++tail_count;
            if(z > 0) ++positive_tail;
            else ++negative_tail;
            REQUIRE(std::isfinite(z));
            REQUIRE(z != 0.0);
        }
    }
    // Expected tail count: 2 * (1 - Phi(3.5)) per draw ~ 4.65e-4 per
    // draw, ~ 465 per million. Wide band [100, 2000] catches gross
    // tail-handling regressions without false-positiving on
    // run-to-run variation.
    INFO("tail samples (|z| > 3.5) per million: " << tail_count
         << " (positive: " << positive_tail << ", negative: " << negative_tail << ")");
    CHECK(tail_count > 100);
    CHECK(tail_count < 2000);
    // Sanity check: both signs should be represented in the tail.
    CHECK(positive_tail > 0);
    CHECK(negative_tail > 0);
}
