#ifndef HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_SEED_H
#define HPP_GUARD_ARGMIN_DETAIL_STOCHASTIC_SEED_H

#include "argmin/detail/xoshiro256.h"

#include <random>
#include <cstdint>
#include <optional>

namespace argmin::detail
{

// An unset seed draws from std::random_device, so every construction and every reset starts a new stream.
inline xoshiro256 seeded_generator(const std::optional<std::uint64_t>& seed)
{
    if(seed.has_value())
        return xoshiro256{*seed};
    return xoshiro256{static_cast<std::uint64_t>(std::random_device{}())};
}

}

#endif
