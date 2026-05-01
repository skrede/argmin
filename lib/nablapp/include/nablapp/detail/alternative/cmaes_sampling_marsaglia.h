#ifndef HPP_GUARD_NABLAPP_DETAIL_ALTERNATIVE_CMAES_SAMPLING_MARSAGLIA_H
#define HPP_GUARD_NABLAPP_DETAIL_ALTERNATIVE_CMAES_SAMPLING_MARSAGLIA_H

// CMA-ES offspring sampling -- Marsaglia polar Gaussian variant.
//
// Drop-in replacement for detail::sample_offspring. Replaces the
// per-coord std::normal_distribution<double>::operator() draw with
// the Marsaglia polar method on top of xoshiro256+. Generates two
// Gaussians per pair of uniforms with one log + one sqrt; no trig
// calls. Roughly 2-3x faster than std::normal_distribution per
// Marsaglia & Bray (1964); the empirical multiplier on the
// nablapp + libstdc++ + xoshiro256+ stack is reported in the Plan
// 05 A/B verdict doc.
//
// The pair-cache is thread_local: even-rank Gaussian draws produce
// a pair whose second element is cached for the next call. The
// publish_bench / micro_cmaes / unit harnesses are single-threaded,
// so thread_local is trivially safe and avoids touching the policy
// state_type. NO new dynamic Eigen per CONTEXT D-12; the cache is
// two scalars + a one-byte flag.
//
// References:
//   Marsaglia, G. & Bray, T. A. (1964) "A Convenient Method for
//     Generating Normal Variables", SIAM Review 6(3), 260-264.
//   Blackman, D. & Vigna, S. (2021) "Scrambled Linear Pseudorandom
//     Number Generators", ACM TOMS 47(4) -- xoshiro256+ engine.
//   K&W (2025) Algorithms for Optimization 2e, §8.7, Algorithm 8.10
//     step 1 (CMA-ES offspring sampling).

#include "nablapp/detail/xoshiro256.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <utility>

namespace nablapp::detail::alternative::marsaglia
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

// Single N(0,1) draw with the spare half-pair cached for the next
// call. Single-threaded harness; thread_local keeps the cache
// per-thread so a future multithreaded caller remains correct.
template <typename Scalar = double, typename RNG>
auto marsaglia_normal(RNG& rng) -> Scalar
{
    thread_local Scalar cached_value{0};
    thread_local bool has_cached{false};

    if(has_cached)
    {
        has_cached = false;
        return cached_value;
    }

    auto [a, b] = marsaglia_polar<Scalar>(rng);
    cached_value = b;
    has_cached = true;
    return a;
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
            z[j] = marsaglia_normal<Scalar>(rng);

        offspring.col(i).noalias() = mean + sigma * (B * D.asDiagonal() * z);
    }

    return offspring;
}

}

#endif
