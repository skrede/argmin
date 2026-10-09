#ifndef HPP_GUARD_ARGMIN_OPTIONS_LEAST_SQUARES_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_LEAST_SQUARES_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct least_squares_defaults;

template <>
struct least_squares_defaults<double>
{
    static constexpr double tau = 1e-3;
    static constexpr double diagonal_min_clamp = 1e-8;
    static constexpr double lambda_min = 1e-20;
    static constexpr double lambda_max = 1e20;
    static constexpr double nu_initial = 2.0;
    static constexpr double damping_factor = 1.0 / 3.0;
    static constexpr double fallback_lambda = 1e-4;
    static constexpr double armijo_c = 1e-4;
    static constexpr double backtrack_rho = 0.5;
};

template <>
struct least_squares_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float tau = 1e-3f;                   // dimensionless
    static constexpr float diagonal_min_clamp = 3e-4f;    // rule
    static constexpr float lambda_min = 1e-20f;           // magnitude cap
    static constexpr float lambda_max = 1e20f;            // magnitude cap
    static constexpr float nu_initial = 2.0f;             // dimensionless
    static constexpr float damping_factor = 1.0f / 3.0f;  // dimensionless
    static constexpr float fallback_lambda = 1e-4f;       // dimensionless
    static constexpr float armijo_c = 1e-4f;              // dimensionless
    static constexpr float backtrack_rho = 0.5f;          // dimensionless
};

}

#endif
