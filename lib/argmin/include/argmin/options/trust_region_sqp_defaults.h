#ifndef HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_SQP_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_TRUST_REGION_SQP_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct trust_region_sqp_defaults;

template <>
struct trust_region_sqp_defaults<double>
{
    static constexpr double gradient_tolerance_fast = 1e-4;
    static constexpr double gradient_tolerance_accurate = 1e-8;
    static constexpr double step_tolerance_rel_fast = 1e-6;
    static constexpr double step_tolerance_rel_accurate = 1e-12;
    static constexpr double feasibility_tolerance_fast = 1e-4;
    static constexpr double feasibility_tolerance_accurate = 1e-6;
    static constexpr double initial_trust_radius = 1.0;
    static constexpr double min_trust_radius = 1e-12;
    static constexpr double initial_penalty = 1.0;
    static constexpr double penalty_factor = 0.3;
    static constexpr double filter_gamma = 1e-2;
    static constexpr double switching_delta = 1.0;
    static constexpr double switching_s_theta = 1.1;
    static constexpr double switching_s_phi = 2.3;
    static constexpr double switching_eta_phi = 1e-4;
    static constexpr double restoration_lambda_init = 1e-3;
    static constexpr double zero_guard = 1e-15;
};

template <>
struct trust_region_sqp_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure. The feasibility rows
    // take the float driver tolerance where the rule value is looser.
    static constexpr float gradient_tolerance_fast = 2e-2f;         // rule
    static constexpr float gradient_tolerance_accurate = 3e-4f;     // rule
    static constexpr float step_tolerance_rel_fast = 2e-3f;         // rule
    static constexpr float step_tolerance_rel_accurate = 5e-6f;     // rule
    static constexpr float feasibility_tolerance_fast = 1e-3f;      // driver row; rule value 2e-2
    static constexpr float feasibility_tolerance_accurate = 1e-3f;  // driver row; rule value 2e-3
    static constexpr float initial_trust_radius = 1.0f;             // dimensionless
    static constexpr float min_trust_radius = 5e-6f;                // rule
    static constexpr float initial_penalty = 1.0f;                  // dimensionless
    static constexpr float penalty_factor = 0.3f;                   // dimensionless
    static constexpr float filter_gamma = 1e-2f;                    // dimensionless
    static constexpr float switching_delta = 1.0f;                  // dimensionless
    static constexpr float switching_s_theta = 1.1f;                // dimensionless
    static constexpr float switching_s_phi = 2.3f;                  // dimensionless
    static constexpr float switching_eta_phi = 1e-4f;               // dimensionless
    static constexpr float restoration_lambda_init = 1e-3f;         // dimensionless
    static constexpr float zero_guard = 2e-7f;                      // rule
};

}

#endif
