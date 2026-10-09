#ifndef HPP_GUARD_ARGMIN_OPTIONS_CMAES_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_CMAES_OPTIONS_H

#include "argmin/types.h"

namespace argmin
{

template <scalar S>
struct cmaes_defaults;

template <>
struct cmaes_defaults<double>
{
    static constexpr double sigma_collapse_threshold = 1e-12;
    static constexpr double condition_number_limit = 1e14;
    static constexpr double objective_value_tolerance = 1e-12;
    static constexpr double step_size_tolerance = 1e-12;
    static constexpr double eigenvalue_floor = 1e-30;
};

template <>
struct cmaes_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float sigma_collapse_threshold = 5e-6f;   // rule
    static constexpr float condition_number_limit = 2e6f;      // rule
    static constexpr float objective_value_tolerance = 5e-6f;  // rule
    static constexpr float step_size_tolerance = 5e-6f;        // rule
    static constexpr float eigenvalue_floor = 1e-30f;          // magnitude cap
};

// CMA-ES strategy and convergence detection parameters.
// Reference: Hansen 2023 tutorial, K&W 2e 8.7.
template <scalar Scalar = double>
struct cmaes_options
{
    // sigma below this = roundoff_limited (Hansen tutorial)
    Scalar sigma_collapse_threshold{cmaes_defaults<Scalar>::sigma_collapse_threshold};
    // covariance condition limit (Hansen tutorial)
    Scalar condition_number_limit{cmaes_defaults<Scalar>::condition_number_limit};

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 6 (paper symbol: TolFun):
    // if the range of the best objective function values over the last
    // 10 + ceil(30 n / lambda) generations and the current generation's
    // offspring drops below this value, the policy exits with
    // solver_status::ftol_reached. Default: 1e-12 in double (Hansen 2023 page 34,
    // "10^-12 is a conservative first guess").
    Scalar objective_value_tolerance{cmaes_defaults<Scalar>::objective_value_tolerance};

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 7 (paper symbol: TolX):
    // if the standard deviation of the sampling distribution drops below
    // `step_size_tolerance * initial_sigma` in all coordinates AND
    // sigma * p_c drops below the same threshold in all components, the
    // policy exits with solver_status::roundoff_limited. Default factor:
    // 1e-12 in double (multiplied by initial_sigma at the use site, per Hansen 2023
    // page 34, "By default we set TolX to 10^-12 times the initial sigma").
    Scalar step_size_tolerance{cmaes_defaults<Scalar>::step_size_tolerance};
};

}

#endif
