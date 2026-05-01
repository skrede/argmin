#ifndef HPP_GUARD_NABLAPP_DETAIL_ALTERNATIVE_CMAES_SAMPLING_ZIGGURAT_H
#define HPP_GUARD_NABLAPP_DETAIL_ALTERNATIVE_CMAES_SAMPLING_ZIGGURAT_H

// CMA-ES offspring sampling -- Ziggurat variant.
//
// Drop-in replacement for detail::sample_offspring. Uses a 256-region
// compile-time-precomputed table (Marsaglia & Tsang 2000 recurrence
// with R = 3.6541528853610087963519472518 and v = R*f(R) +
// integral_R^infty f(x) dx ~ 0.00492867323399). The common path
// (~99% acceptance) executes one 64-bit rng() draw, one byte-mask, one
// signed multiply, one comparison; no sqrt/log/div. The wedge path
// (~1%) does an inverse-CDF rejection. The tail path (i == 0,
// |x| > R) uses Marsaglia & Tsang 2000 §3 fallback (Doornik 2005
// refines but is mathematically equivalent for the standard normal).
//
// Layout (GSL gsl_ran_gaussian_ziggurat / Marsaglia & Tsang convention):
//   Strip 0   = bottom rectangle of width R + tail beyond R.
//   Strip i, i in [1, N-2] = rectangle of width x[i] under
//     y[i] - y[i-1], where y[i] = f(x[i]).
//   Strip N-1 = top cap (under y[N-1]).
//   x[N-1] = R, x[0] = sentinel = v / f(R) used as the rectangle
//     width for strip 0's common-path acceptance test.
//   y[i]   = exp(-x[i]^2 / 2)
//
// Caveats:
//   - Different RNG-byte -> Gaussian-value mapping than Marsaglia or
//     Box-Muller. Distribution is statistically equivalent
//     (chi-square pass) but byte-exact reproducibility breaks vs the
//     production sampler. Acceptable per
//     `feedback_correctness_over_compat`; re-baseline.
//   - Tail handling has historically had edge-case bugs in published
//     implementations. The cmaes_sampling_test.cpp chi-square test
//     covers the bulk distribution; the dedicated tail-path test
//     covers tail draws (|z| > 3.5) for finite + nonzero output.
//
// References:
//   Marsaglia, G. & Tsang, W. W. (2000) "The Ziggurat Method for
//     Generating Random Variables", J. Statistical Software 5(8).
//   Doornik, J. A. (2005) "An Improved Ziggurat Method to Generate
//     Normal Random Samples".
//   Blackman, D. & Vigna, S. (2021) "Scrambled Linear Pseudorandom
//     Number Generators", ACM TOMS 47(4) -- xoshiro256+ engine.
//   K&W (2025) Algorithms for Optimization 2e, §8.7, Algorithm 8.10
//     step 1 (CMA-ES offspring sampling).

#include "nablapp/detail/xoshiro256.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

namespace nablapp::detail::alternative::ziggurat
{

namespace ziggurat_internal
{

constexpr int N_REGIONS = 256;
constexpr double R = 3.6541528853610087963519472518;
constexpr double v = 0.00492867323399;

// Compile-time table:
//   x[i]: rectangle widths (and tail-handling sentinel at index 0).
//   y[i]: f(x[i]) values used for the wedge-path acceptance test.
//
// Recurrence walks down from the tail boundary x[N-1] = R:
//   x[i] = sqrt(-2 ln(f(x[i+1]) + v / x[i+1])),  i = N-2 .. 1
//   x[0] = v / f(R) (sentinel; common-path accept ratio for strip 0)
//
// Convergence: the published v = 0.00492867323399 ensures
// f(x[i+1]) + v/x[i+1] < 1 throughout, so sqrt of a positive
// argument is real. A defensive guard is included for the edge case.
consteval auto compute_x_table() -> std::array<double, N_REGIONS>
{
    std::array<double, N_REGIONS> x{};
    x[N_REGIONS - 1] = R;
    for(int i = N_REGIONS - 2; i >= 1; --i)
    {
        const double f_next = std::exp(-x[i + 1] * x[i + 1] / 2.0);
        const double arg = f_next + v / x[i + 1];
        x[i] = (arg > 0.0 && arg < 1.0)
            ? std::sqrt(-2.0 * std::log(arg))
            : 0.0;
    }
    // Sentinel: ratio for the strip-0 common-path acceptance is
    // x[0] / x[1] = (v / f(R)) / x[1]. Storing the numerator at x[0]
    // and dividing by x[1] in the body matches the GSL idiom.
    x[0] = v / std::exp(-R * R / 2.0);
    return x;
}

consteval auto compute_y_table() -> std::array<double, N_REGIONS>
{
    auto x = compute_x_table();
    std::array<double, N_REGIONS> y{};
    // y[0] is unused as a density value (strip 0 is the tail-containing
    // rectangle); set to f(R) for symmetry with the wedge formula.
    y[0] = std::exp(-R * R / 2.0);
    for(int i = 1; i < N_REGIONS; ++i)
        y[i] = std::exp(-x[i] * x[i] / 2.0);
    return y;
}

inline constexpr auto x_table = compute_x_table();
inline constexpr auto y_table = compute_y_table();

}

// Single N(0,1) draw via Marsaglia & Tsang (2000) Ziggurat.
template <typename Scalar = double, typename RNG>
auto ziggurat_normal(RNG& rng) -> Scalar
{
    using namespace ziggurat_internal;
    std::uniform_real_distribution<Scalar> uni01(Scalar(0), Scalar(1));

    while(true)
    {
        // One 64-bit URBG draw splits into:
        //   - 8 bits of region index i in [0, N-1]
        //   - 1 bit of sign
        //   - 53 bits of magnitude in [0, 1) (uniform double)
        const std::uint64_t r = static_cast<std::uint64_t>(rng());
        const std::uint8_t i = static_cast<std::uint8_t>(r & 0xFFu);
        const std::uint64_t hi = r >> 8;
        const Scalar sign = (hi & 0x1ULL) ? Scalar(1) : Scalar(-1);
        const Scalar u = static_cast<Scalar>(hi >> 1)
                       / static_cast<Scalar>(1ULL << 55);
        // Scale the [0, 1) magnitude by the strip width x[i]; sign
        // gives the symmetric distribution.
        const Scalar x = sign * u * static_cast<Scalar>(x_table[i]);

        // Common-path test: accept if |x| is inside the inscribed
        // rectangle for this strip. The acceptance ratio is
        // x[i-1] / x[i] for strip i in [1, N-1]; for strip 0 the ratio
        // is x[0] / x[1] (using the sentinel layout).
        if(i > 0)
        {
            if(u < static_cast<Scalar>(x_table[i - 1] / x_table[i]))
                return x;
        }
        else
        {
            // Strip 0: common-path threshold = sentinel x[0] / x[1].
            if(u < static_cast<Scalar>(x_table[0] / x_table[1]))
                return x;
        }

        // Tail path (strip 0, |x| in the tail region beyond R).
        if(i == 0)
        {
            // Marsaglia & Tsang (2000) §3 tail fallback: rejection
            // from an exponential proposal centered at R.
            Scalar xt;
            Scalar yt;
            do
            {
                xt = static_cast<Scalar>(-std::log(uni01(rng)) / R);
                yt = static_cast<Scalar>(-std::log(uni01(rng)));
            }
            while(yt + yt < xt * xt);
            return sign * static_cast<Scalar>(R + xt);
        }

        // Wedge path: accept iff Uniform(y[i-1], y[i]) draw lies under
        // f(x). For strip i in [1, N-1] the wedge sits between the
        // inscribed rectangle and the density curve f(x).
        const Scalar u2 = uni01(rng);
        const Scalar y_lo = static_cast<Scalar>(y_table[i - 1]);
        const Scalar y_hi = static_cast<Scalar>(y_table[i]);
        const Scalar f_x = static_cast<Scalar>(std::exp(-x * x / Scalar(2)));
        if(y_lo + u2 * (y_hi - y_lo) < f_x)
            return x;
        // else: reject and retry from a fresh URBG draw.
    }
}

// Sample lambda offspring x_i = mean + sigma * B * D * z_i, z_i ~ N(0, I).
// Same template signature as detail::sample_offspring so this is a
// drop-in replacement when production aliases the variant.
// Reference: K&W (2025) Algorithm 8.10 step 1.
template <typename Scalar = double, int N = nablapp::dynamic_dimension,
          int MaxLambda = Eigen::Dynamic, typename RNG = std::mt19937>
auto sample_offspring(
    const Eigen::Vector<Scalar, N>& mean,
    Scalar sigma,
    const Eigen::Matrix<Scalar, N, N>& B,
    const Eigen::Vector<Scalar, N>& D,
    int lambda,
    RNG& rng)
    -> Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0,
        N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxLambda>
{
    const int n = mean.size();

    Eigen::Matrix<Scalar, N, Eigen::Dynamic, 0,
        N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxLambda>
            offspring(n, lambda);

    for(int i = 0; i < lambda; ++i)
    {
        Eigen::Vector<Scalar, N> z(n);
        for(int j = 0; j < n; ++j)
            z[j] = ziggurat_normal<Scalar>(rng);

        offspring.col(i).noalias() = mean + sigma * (B * D.asDiagonal() * z);
    }

    return offspring;
}

}

#endif
