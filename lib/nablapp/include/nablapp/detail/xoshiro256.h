#ifndef HPP_GUARD_NABLAPP_DETAIL_XOSHIRO256_H
#define HPP_GUARD_NABLAPP_DETAIL_XOSHIRO256_H

// xoshiro256+ pseudorandom number generator.
//
// 256-bit state (32 bytes), period 2^256-1. Optimized for floating-point
// output (fastest variant). Lowest 3 bits have slight linear bias,
// irrelevant when consumed by std::normal_distribution.
//
// Conforms to the C++ UniformRandomBitGenerator named requirement:
// result_type, min(), max(), operator()().
//
// Reference: Blackman, D. and Vigna, S. (2021) "Scrambled Linear
//            Pseudorandom Number Generators", ACM TOMS 47(4).
//            https://prng.di.unimi.it/xoshiro256plus.c

#include <array>
#include <cstdint>

namespace nablapp::detail
{

struct xoshiro256
{
    using result_type = std::uint64_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    std::array<std::uint64_t, 4> state{};

    xoshiro256() = delete;

    // Expand seed into 4 state words via splitmix64.
    // Reference: Vigna, S. (2020) splitmix64.c
    explicit xoshiro256(std::uint64_t seed)
    {
        for(auto& s : state)
        {
            seed += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s = z ^ (z >> 31);
        }
    }

    result_type operator()()
    {
        const result_type result = state[0] + state[3];
        const std::uint64_t t = state[1] << 17;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = (state[3] << 45) | (state[3] >> 19);

        return result;
    }
};

}

#endif
