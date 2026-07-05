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
#include <cstddef>
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

TEST_CASE("cmaes_sampling: ziggurat_normal produces draws beyond the truncation sentinel R",
          "[cmaes_sampling][!shouldfail]")
{
    // Expected to FAIL against the current production ziggurat_normal (a
    // known, not-yet-fixed truncation defect -- see the comment below).
    // [!shouldfail] records this as the expected disposition, per the
    // project's established fix-detector convention (see
    // filter_trsqp_test.cpp): once the sampler is corrected this case
    // will start passing, which Catch2 reports as an unexpected success,
    // forcing the tag's removal at that point.
    // cmaes_sampling_ziggurat.h's strip-0 common-path acceptance test uses
    // the ratio x_table[0]/x_table[1], where x_table[0] = v/f(R) ~ 3.911
    // (the sentinel) and x_table[1] << R. That ratio is >> 1, so
    // "u < ratio" (u in [0,1)) is true for every strip-0 draw -- the tail
    // fallback beneath it is never reached, and |z| is truncated at the
    // sentinel ~3.911 rather than following the true unbounded normal tail.
    // The chi-square/tail-band tests above (fixed at 3.5 sigma, well
    // inside the truncation point) cannot see this: they only exercise
    // draws already known to lie below 3.911. This statistic looks past
    // the truncation point itself.
    argmin::detail::xoshiro256 rng{42u};
    constexpr int N_DRAWS = 400'000;
    constexpr double TAIL_BOUND = 3.911;
    int beyond = 0;
    for(int i = 0; i < N_DRAWS; ++i)
    {
        const double z = argmin::detail::alternative::ziggurat::ziggurat_normal<double>(rng);
        if(std::abs(z) > TAIL_BOUND) ++beyond;
    }
    // Expected count for a correct (untruncated) standard-normal sampler:
    // 2 * (1 - Phi(3.911)) * 400000 ~ 37; a [15, 70] band absorbs
    // run-to-run sampling variation (Poisson-ish, sd ~ 6) without
    // false-positiving.
    INFO("draws with |z| > " << TAIL_BOUND << " in " << N_DRAWS << ": " << beyond
         << " (expected ~37 for an untruncated sampler)");
    CHECK(beyond > 15);
    CHECK(beyond < 70);
}

TEST_CASE("cmaes_sampling: marsaglia_normal is bit-identical across "
          "identically-seeded odd-consumption sequences",
          "[cmaes_sampling][!shouldfail]")
{
    // Expected to FAIL against the current production marsaglia_normal (a
    // known, not-yet-fixed thread_local pair-cache leak -- see the
    // comment below). [!shouldfail] records this as the expected
    // disposition, per the same fix-detector convention as above.
    // cmaes_sampling_marsaglia.h's pair cache is thread_local -- scoped to
    // the process/thread, not to an RNG instance or a draw sequence. An
    // EVEN total consumption (the historical n=2 determinism test) always
    // starts and ends each sequence with an empty cache, so it is
    // structurally blind to this. An ODD total consumption leaves one
    // spare cached when a sequence ends; the next identically-seeded
    // sequence should still reproduce the first sequence bit-for-bit (same
    // seed => same raw stream => same output), but instead its first draw
    // is silently satisfied from the stale spare left over by the PRIOR
    // sequence.
    constexpr int n_draws = 5; // odd lambda*n
    constexpr std::uint64_t seed = 123u;

    // Normalize the entering cache state so this test is independent of
    // whatever earlier TEST_CASEs (or Catch2 execution order/shuffling)
    // left behind: two draws is an even consumption, which always leaves
    // the cache empty afterward.
    {
        argmin::detail::xoshiro256 warmup_rng{999u};
        argmin::detail::alternative::marsaglia::marsaglia_normal<double>(warmup_rng);
        argmin::detail::alternative::marsaglia::marsaglia_normal<double>(warmup_rng);
    }

    std::array<double, n_draws> seq1{};
    {
        argmin::detail::xoshiro256 rng{seed};
        for(int i = 0; i < n_draws; ++i)
            seq1[static_cast<std::size_t>(i)]
                = argmin::detail::alternative::marsaglia::marsaglia_normal<double>(rng);
    }

    std::array<double, n_draws> seq2{};
    {
        argmin::detail::xoshiro256 rng{seed}; // identically-seeded, freshly rebuilt
        for(int i = 0; i < n_draws; ++i)
            seq2[static_cast<std::size_t>(i)]
                = argmin::detail::alternative::marsaglia::marsaglia_normal<double>(rng);
    }

    INFO("seq1[0]=" << seq1[0] << " seq2[0]=" << seq2[0]);
    CHECK(seq1 == seq2);
}
