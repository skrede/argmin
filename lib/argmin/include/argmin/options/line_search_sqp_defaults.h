#ifndef HPP_GUARD_ARGMIN_OPTIONS_LINE_SEARCH_SQP_DEFAULTS_H
#define HPP_GUARD_ARGMIN_OPTIONS_LINE_SEARCH_SQP_DEFAULTS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct line_search_sqp_defaults;

template <>
struct line_search_sqp_defaults<double>
{
    static constexpr double gradient_tolerance = 1e-8;
    static constexpr double step_tolerance_rel = 1e-12;
    static constexpr double feasibility_tolerance = 1e-6;
    static constexpr double armijo_c1 = 1e-4;
    static constexpr double armijo_rho = 0.5;
    static constexpr double qp_tolerance = 1e-12;
    static constexpr double sigma_max = 1e10;
    static constexpr double initial_penalty = 1.0;
    static constexpr double zero_guard = 1e-15;
    static constexpr double filter_gamma_f = 1e-5;
    static constexpr double filter_gamma_h = 1e-5;
    static constexpr double infeasibility_floor = 1e-12;
    static constexpr double restoration_alpha_min = 1e-10;
    static constexpr double restoration_trigger = 1e-8;
};

template <>
struct line_search_sqp_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float gradient_tolerance = 3e-4f;     // rule
    static constexpr float step_tolerance_rel = 5e-6f;     // rule
    static constexpr float feasibility_tolerance = 1e-3f;  // gate: rule value 2e-3 failed hs043
    static constexpr float armijo_c1 = 1e-4f;              // dimensionless
    static constexpr float armijo_rho = 0.5f;              // dimensionless
    static constexpr float qp_tolerance = 5e-6f;           // rule
    static constexpr float sigma_max = 1e10f;              // magnitude cap
    static constexpr float initial_penalty = 1.0f;         // dimensionless
    static constexpr float zero_guard = 2e-7f;             // rule
    static constexpr float filter_gamma_f = 1e-5f;         // dimensionless
    static constexpr float filter_gamma_h = 1e-5f;         // dimensionless
    static constexpr float infeasibility_floor = 5e-6f;    // rule
    static constexpr float restoration_alpha_min = 4e-5f;  // rule
    static constexpr float restoration_trigger = 3e-4f;    // rule
};

}

#endif
