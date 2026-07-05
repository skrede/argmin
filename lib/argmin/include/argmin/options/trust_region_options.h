#ifndef HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H

namespace argmin
{

// Trust region radius update parameters.
// Reference: Powell 2009, Sections 3-6.
struct trust_region_options
{
    double eta_good{0.7};           // rho threshold for expansion (Powell S5)
    double eta_poor{0.1};           // rho threshold for contraction (Powell S5)
    double expand_factor{2.0};      // radius expansion multiplier (Powell S5)
    double shrink_factor{0.5};      // radius contraction multiplier (Powell S5)
    double step_threshold{0.5};     // step/delta ratio for expansion (Powell S5)
    double geometry_factor{2.0};    // geometry check distance factor (Powell S6)
};

}

#endif
