#ifndef HPP_GUARD_ARGMIN_DETAIL_ALTERNATIVE_CMAES_SAMPLING_MARSAGLIA_H
#define HPP_GUARD_ARGMIN_DETAIL_ALTERNATIVE_CMAES_SAMPLING_MARSAGLIA_H

// CMA-ES offspring sampling -- Marsaglia polar Gaussian variant.
//
// Drop-in replacement for detail::sample_offspring. Replaces the
// per-coord std::normal_distribution<double>::operator() draw with
// the Marsaglia polar method on top of xoshiro256+. Generates two
// Gaussians per pair of uniforms with one log + one sqrt; no trig
// calls. Roughly 2-3x faster than std::normal_distribution per
// Marsaglia & Bray (1964); the empirical multiplier on the
// argmin + libstdc++ + xoshiro256+ stack is reported in the Plan
// 05 A/B verdict doc.
//
// The pair-cache carries NO cross-call state: each single draw
// computes one polar pair and discards the spare, so the value a
// caller receives is a pure function of the RNG stream it passes in.
// An earlier version cached the spare half-pair in a `thread_local`,
// which leaked draws across independently-seeded RNG instances and
// across separate draw sequences on the same thread: two identically
// seeded sequences of ODD length would disagree on their first draw,
// because the second sequence silently consumed the spare left over
// by the first. That broke seeded reproducibility (the same seed no
// longer implied the same stream). To retain the pair efficiency
// without the leakage, sample_offspring below consumes both halves of
// each polar pair locally within a single call.
//
// References:
//   Marsaglia, G. & Bray, T. A. (1964) "A Convenient Method for
//     Generating Normal Variables", SIAM Review 6(3), 260-264.
//   Blackman, D. & Vigna, S. (2021) "Scrambled Linear Pseudorandom
//     Number Generators", ACM TOMS 47(4) -- xoshiro256+ engine.
//   K&W (2025) Algorithms for Optimization 2e, §8.7, Algorithm 8.10
//     step 1 (CMA-ES offspring sampling).

#include "argmin/detail/xoshiro256.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <utility>

namespace argmin::detail::alternative::marsaglia
{

// One Marsaglia polar pair: two independent N(0,1) draws from two
// uniforms on (-1, 1) accepted into the unit disc.
// Reference: Marsaglia & Bray (1964), SIAM Review 6(3), 260-264.
template <typename Scalar = double, typename RNG>
auto marsaglia_polar(RNG& rng) -> std::pair<Scalar, Scalar>
{
    std::uniform_real_distribution<Scalar> uni(Scalar(-1), Scalar(1));
    while(true)
    {
        const Scalar u = uni(rng);
        const Scalar v = uni(rng);
        const Scalar s = u * u + v * v;
        if(s > Scalar(0) && s < Scalar(1))
        {
            const Scalar factor = std::sqrt(Scalar(-2) * std::log(s) / s);
            return {u * factor, v * factor};
        }
    }
}

// Single N(0,1) draw. Stateless with respect to prior calls: it
// computes one polar pair and returns the first half, discarding the
// spare. The result depends only on the RNG stream, so identically
// seeded RNGs always produce identical draw sequences (no cross-call
// or cross-instance carry-over). Callers wanting to consume both
// halves of a pair should use sample_offspring, which fills its
// buffer two components at a time.
template <typename Scalar = double, typename RNG>
auto marsaglia_normal(RNG& rng) -> Scalar
{
    return marsaglia_polar<Scalar>(rng).first;
}

// Sample lambda offspring x_i = mean + sigma * B * D * z_i, z_i ~ N(0, I).
// Same template signature as detail::sample_offspring so this is a
// drop-in replacement when production aliases the variant.
// Reference: K&W (2025) Algorithm 8.10 step 1.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
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
        // Consume both halves of each polar pair locally: fill z two
        // components at a time. No persistent cache, so the RNG stream
        // fully determines the output. For odd n the final component
        // takes the first half of a fresh pair and drops the spare.
        int j = 0;
        for(; j + 1 < n; j += 2)
        {
            const auto [a, b] = marsaglia_polar<Scalar>(rng);
            z[j] = a;
            z[j + 1] = b;
        }
        if(j < n)
            z[j] = marsaglia_polar<Scalar>(rng).first;

        offspring.col(i).noalias() = mean + sigma * (B * D.asDiagonal() * z);
    }

    return offspring;
}

}

#endif
