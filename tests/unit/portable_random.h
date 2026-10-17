#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_PORTABLE_RANDOM_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_PORTABLE_RANDOM_H

#include <random>
#include <cstdint>

// The standard distributions are implementation-defined, so libstdc++, libc++ and the MSVC STL draw
// different values from one engine and seed. These read only the engine's raw output and round
// exactly, so a seeded fixture builds the same data on every platform.
namespace argmin::portable_random
{

inline double canonical(std::mt19937& rng)
{
    return static_cast<double>(rng() >> 6) / static_cast<double>(std::uint32_t{1} << 26);
}

// Two statements so no compiler fuses the multiply and add into one differently rounded operation.
inline double uniform(std::mt19937& rng, double lo, double hi)
{
    const double offset = (hi - lo) * canonical(rng);
    return lo + offset;
}

// Irwin-Hall approximation of a standard normal: twelve unit uniforms, shifted to zero mean.
inline double normal(std::mt19937& rng)
{
    double sum = 0.0;
    for(std::int32_t i = 0; i < 12; ++i)
        sum += canonical(rng);
    return sum - 6.0;
}

}

#endif
