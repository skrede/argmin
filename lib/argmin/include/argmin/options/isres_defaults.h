#ifndef HPP_GUARD_ARGMIN_OPTIONS_ISRES_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_ISRES_DEFAULTS_H

#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

template <scalar S>
struct isres_defaults;

template <>
struct isres_defaults<double>
{
    static constexpr double sigma_collapse_ratio = 1e-9;
    // Twice the longest plateau measured on the pinned ISRES cases at the default stall threshold: 232 generations on
    // hs048 before the best value falls from 84 to 3e-9. A 570-generation plateau of the original_argmin variant on
    // hs071 stays uncovered; that run sits near 95.9, nowhere near the 17.01 optimum.
    static constexpr std::uint16_t stall_window = 500;
};

template <>
struct isres_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float sigma_collapse_ratio = 1e-4f;  // rule
    static constexpr std::uint16_t stall_window = 500;
};

}

#endif
