#ifndef HPP_GUARD_NABLAPP_OPTIONS_ASYMPTOTE_OPTIONS_H
#define HPP_GUARD_NABLAPP_OPTIONS_ASYMPTOTE_OPTIONS_H

#include <optional>

namespace nablapp
{

// MMA/GCMMA asymptote update parameters.
// Reference: Svanberg 1987.
struct asymptote_options
{
    std::optional<double> minimum_distance_fraction{};  // min distance as fraction of range (default: 0.01, Svanberg 1987)
};

}

#endif
