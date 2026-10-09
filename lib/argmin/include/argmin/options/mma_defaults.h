#ifndef HPP_GUARD_ARGMIN_OPTIONS_MMA_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_MMA_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct mma_defaults;

template <>
struct mma_defaults<double>
{
    static constexpr double dual_gradient_threshold = 1e-9;
    static constexpr double dual_step_threshold = 1e-15;
    static constexpr double min_asymptote_distance = 1e-12;
    static constexpr double violation_slack = 1e-12;
    static constexpr double subproblem_epsilon = 1e-10;
    static constexpr double dual_newton_tolerance = 1e-12;
    static constexpr double subproblem_dual_tolerance = 1e-9;
    static constexpr double subproblem_backtrack_factor = 0.95;
    static constexpr double stall_threshold = 1e-6;
    static constexpr double asymptote_contract = 0.7;
    static constexpr double asymptote_expand = 1.2;
    static constexpr double asymptote_init = 0.5;
    static constexpr double asymptote_init_unbounded = 1.0;
    static constexpr double asymptote_min_fraction = 1e-4;
    static constexpr double asymptote_max_fraction = 10.0;
    static constexpr double move_limit_fraction = 0.1;
    static constexpr double move_bound_fraction = 0.5;
    static constexpr double raai = 1e-5;
    static constexpr double dual_bound_scale = 1000.0;
    static constexpr double sigma_init = 0.5;
    static constexpr double sigma_decrease = 0.7;
    static constexpr double sigma_increase = 1.2;
    static constexpr double sigma_min_fraction = 1e-8;
    static constexpr double sigma_max_fraction = 10.0;
    static constexpr double rho_init = 1.0;
    static constexpr double rho_min = 1e-5;
    static constexpr double rho_decay = 0.1;
    static constexpr double raa_init = 1e-5;
    static constexpr double raa_min = 1e-7;
    static constexpr double raa_decay = 0.5;
    static constexpr double shrink_factor = 0.5;
};

template <>
struct mma_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure. The asymptote, move-limit,
    // sigma, rho and raa parameters are ratios of the variable range or the penalty scale.
    static constexpr float dual_gradient_threshold = 1e-4f;      // rule
    static constexpr float dual_step_threshold = 2e-7f;          // rule
    static constexpr float min_asymptote_distance = 5e-6f;       // rule
    static constexpr float violation_slack = 5e-6f;              // rule
    static constexpr float subproblem_epsilon = 4e-5f;           // rule
    static constexpr float dual_newton_tolerance = 5e-6f;        // rule
    static constexpr float subproblem_dual_tolerance = 1e-4f;    // rule
    static constexpr float subproblem_backtrack_factor = 0.95f;  // dimensionless
    static constexpr float stall_threshold = 2e-3f;              // rule
    static constexpr float asymptote_contract = 0.7f;            // dimensionless
    static constexpr float asymptote_expand = 1.2f;              // dimensionless
    static constexpr float asymptote_init = 0.5f;                // dimensionless
    static constexpr float asymptote_init_unbounded = 1.0f;      // dimensionless
    static constexpr float asymptote_min_fraction = 1e-4f;       // dimensionless
    static constexpr float asymptote_max_fraction = 10.0f;       // dimensionless
    static constexpr float move_limit_fraction = 0.1f;           // dimensionless
    static constexpr float move_bound_fraction = 0.5f;           // dimensionless
    static constexpr float raai = 1e-5f;                         // dimensionless
    static constexpr float dual_bound_scale = 1000.0f;           // dimensionless
    static constexpr float sigma_init = 0.5f;                    // dimensionless
    static constexpr float sigma_decrease = 0.7f;                // dimensionless
    static constexpr float sigma_increase = 1.2f;                // dimensionless
    static constexpr float sigma_min_fraction = 1e-8f;           // dimensionless
    static constexpr float sigma_max_fraction = 10.0f;           // dimensionless
    static constexpr float rho_init = 1.0f;                      // dimensionless
    static constexpr float rho_min = 1e-5f;                      // dimensionless
    static constexpr float rho_decay = 0.1f;                     // dimensionless
    static constexpr float raa_init = 1e-5f;                     // dimensionless
    static constexpr float raa_min = 1e-7f;                      // dimensionless
    static constexpr float raa_decay = 0.5f;                     // dimensionless
    static constexpr float shrink_factor = 0.5f;                 // dimensionless
};

}

#endif
