#ifndef HPP_GUARD_ARGMIN_OPTIONS_CONVERGENCE_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_CONVERGENCE_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct convergence_defaults;

template <>
struct convergence_defaults<double>
{
    static constexpr double gradient_threshold = 1e-5;
    static constexpr double objective_threshold = 1e-10;
    static constexpr double stationarity_threshold = 1e-8;
    static constexpr double step_threshold = 1e-8;
    static constexpr double feasibility_tolerance = 1e-6;
    static constexpr double stall_threshold = 1e-6;
};

template <>
struct convergence_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float gradient_threshold = 6e-3f;      // rule
    static constexpr float objective_threshold = 4e-5f;     // rule
    static constexpr float stationarity_threshold = 3e-4f;  // rule
    static constexpr float step_threshold = 3e-4f;          // rule
    static constexpr float feasibility_tolerance = 1e-3f;   // gate: rule value 2e-3 failed hs043
    static constexpr float stall_threshold = 2e-3f;         // rule
};

}

#endif
