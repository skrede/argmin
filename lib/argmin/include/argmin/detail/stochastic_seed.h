#ifndef HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_SEED_H
#define HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_SEED_H

#include "argmin/detail/xoshiro256.h"

#include <random>
#include <cstdint>
#include <optional>

namespace argmin::detail
{

// An unset seed draws from std::random_device, so every construction and every reset starts a new stream.
inline std::uint64_t resolved_seed(const std::optional<std::uint64_t>& seed)
{
    if(seed.has_value())
        return *seed;
    return static_cast<std::uint64_t>(std::random_device{}());
}

inline xoshiro256 seeded_generator(const std::optional<std::uint64_t>& seed)
{
    return xoshiro256{resolved_seed(seed)};
}

// The index-th output of a splitmix64 stream started at the base seed, so a fixed base gives every restart its own
// reproducible seed (Vigna, "Further scramblings of Marsaglia's xorshift generators", 2017; splitmix64 seeds xoshiro).
inline std::uint64_t restart_seed(std::uint64_t base, std::uint32_t index)
{
    std::uint64_t state = base + std::uint64_t{index} * 0x9e3779b97f4a7c15ULL;
    return splitmix64(state);
}

}

#endif
