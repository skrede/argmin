#ifndef HPP_GUARD_ARGMIN_OPTIONS_ADMM_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_ADMM_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct admm_defaults;

template <>
struct admm_defaults<double>
{
    static constexpr double rho = 0.1;
    static constexpr double sigma = 1e-6;
    static constexpr double alpha = 1.6;
    static constexpr double eps_abs = 1e-6;
    static constexpr double eps_rel = 1e-6;
    static constexpr double eps_prim_inf = 1e-4;
    static constexpr double eps_dual_inf = 1e-4;
    static constexpr double adaptive_rho_tolerance = 5.0;
    static constexpr double delta = 1e-6;
    static constexpr double rho_min = 1e-6;
    static constexpr double rho_max = 1e6;
    static constexpr double eq_boost = 1e3;
    static constexpr double eq_tol = 1e-9;
    static constexpr double polish_active_tol = 1e-12;
};

template <>
struct admm_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float rho = 0.1f;                     // dimensionless
    static constexpr float sigma = 2e-3f;                  // rule
    static constexpr float alpha = 1.6f;                   // dimensionless
    static constexpr float eps_abs = 2e-3f;                // rule
    static constexpr float eps_rel = 2e-3f;                // rule
    static constexpr float eps_prim_inf = 2e-2f;           // rule
    static constexpr float eps_dual_inf = 2e-2f;           // rule
    static constexpr float adaptive_rho_tolerance = 5.0f;  // dimensionless
    static constexpr float delta = 2e-3f;                  // rule
    static constexpr float rho_min = 2e-3f;                // rule
    static constexpr float rho_max = 1e6f;                 // magnitude cap
    static constexpr float eq_boost = 1e3f;                // dimensionless
    static constexpr float eq_tol = 1e-4f;                 // rule
    static constexpr float polish_active_tol = 5e-6f;      // rule
};

}

#endif
