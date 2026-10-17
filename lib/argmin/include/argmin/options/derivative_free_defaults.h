#ifndef HPP_GUARD_ARGMIN_OPTIONS_DERIVATIVE_FREE_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_DERIVATIVE_FREE_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct derivative_free_defaults;

template <>
struct derivative_free_defaults<double>
{
    static constexpr double final_trust_radius = 1e-8;
    static constexpr double cobyla_feasibility_gate = 1e-4;
    static constexpr double scale_guard = 1e-15;
};

template <>
struct derivative_free_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float final_trust_radius = 3e-4f;       // rule
    static constexpr float cobyla_feasibility_gate = 2e-2f;  // rule
    static constexpr float scale_guard = 2e-7f;              // rule
};

}

#endif
