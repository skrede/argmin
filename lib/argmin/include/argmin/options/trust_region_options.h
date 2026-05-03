#ifndef HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H

#include <optional>

namespace argmin
{

// Trust region radius update parameters.
// Reference: Powell 2009, Sections 3-6.
struct trust_region_options
{
    std::optional<double> eta_good{};           // rho threshold for expansion (default: 0.7, Powell S5)
    std::optional<double> eta_poor{};           // rho threshold for contraction (default: 0.1, Powell S5)
    std::optional<double> expand_factor{};      // radius expansion multiplier (default: 2.0, Powell S5)
    std::optional<double> shrink_factor{};      // radius contraction multiplier (default: 0.5, Powell S5)
    std::optional<double> step_threshold{};     // step/delta ratio for expansion (default: 0.5, Powell S5)
    std::optional<double> geometry_factor{};    // geometry check distance factor (default: 2.0, Powell S6)
};

}

#endif
