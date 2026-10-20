#ifndef HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_OPTIONS_H

#include "argmin/types.h"

namespace argmin
{

// Trust region radius update parameters.
// Reference: Powell 2009, Sections 3-6.
template <scalar Scalar = double>
struct trust_region_options
{
    Scalar eta_good{Scalar(0.7)};           // rho threshold for expansion (Powell S5)
    Scalar eta_poor{Scalar(0.1)};           // rho threshold for contraction (Powell S5)
    Scalar expand_factor{Scalar(2.0)};      // radius expansion multiplier (Powell S5)
    Scalar shrink_factor{Scalar(0.5)};      // radius contraction multiplier (Powell S5)
    Scalar step_threshold{Scalar(0.5)};     // step/delta ratio for expansion (Powell S5)
    Scalar geometry_factor{Scalar(2.0)};    // geometry check distance factor (Powell S6)
};

}

#endif
