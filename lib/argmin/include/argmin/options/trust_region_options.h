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
    Scalar eta_good{Scalar(0.7)};
    Scalar eta_poor{Scalar(0.1)};
    Scalar expand_factor{Scalar(2.0)};
    Scalar shrink_factor{Scalar(0.5)};
    Scalar step_threshold{Scalar(0.5)};
    Scalar geometry_factor{Scalar(2.0)};
};

}

#endif
