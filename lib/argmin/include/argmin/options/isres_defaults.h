#ifndef HPP_GUARD_ARGMIN_OPTIONS_ISRES_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_ISRES_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct isres_defaults;

template <>
struct isres_defaults<double>
{
    static constexpr double feasibility_gate = 1e-4;
    static constexpr double sigma_collapse_ratio = 1e-9;
};

template <>
struct isres_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float feasibility_gate = 1e-3f;      // gate: rule value 2e-2 failed hs023
    static constexpr float sigma_collapse_ratio = 1e-4f;  // rule
};

}

#endif
