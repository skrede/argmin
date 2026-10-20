#ifndef HPP_GUARD_ARGMIN_OPTIONS_AUGMENTED_LAGRANGIAN_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_AUGMENTED_LAGRANGIAN_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct augmented_lagrangian_defaults;

template <>
struct augmented_lagrangian_defaults<double>
{
    static constexpr double mu_init = 0.1;
    static constexpr double mu_decrease = 0.25;
    static constexpr double mu_min = 1e-6;
    static constexpr double constraint_tolerance = 1e-6;
    static constexpr double inner_gradient_tolerance = 1e-6;
    static constexpr double feasibility_progress = 0.25;
    static constexpr double inner_tolerance_eta = 0.1;
    static constexpr double inner_tolerance_alpha = 0.9;
    static constexpr double inner_feasibility_tolerance = 1e-6;
    static constexpr double inner_stall_threshold = 1e-15;
};

template <>
struct augmented_lagrangian_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float mu_init = 0.1f;                       // dimensionless
    static constexpr float mu_decrease = 0.25f;                  // dimensionless
    static constexpr float mu_min = 2e-3f;                       // rule
    static constexpr float constraint_tolerance = 2e-3f;         // rule
    static constexpr float inner_gradient_tolerance = 2e-3f;     // rule
    static constexpr float feasibility_progress = 0.25f;         // dimensionless
    static constexpr float inner_tolerance_eta = 0.1f;           // dimensionless
    static constexpr float inner_tolerance_alpha = 0.9f;         // dimensionless
    static constexpr float inner_feasibility_tolerance = 2e-3f;  // rule
    static constexpr float inner_stall_threshold = 2e-7f;        // rule
};

}

#endif
