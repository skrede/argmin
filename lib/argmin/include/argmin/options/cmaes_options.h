#ifndef HPP_GUARD_ARGMIN_OPTIONS_CMAES_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_CMAES_OPTIONS_H

namespace argmin
{

// CMA-ES strategy and convergence detection parameters.
// Reference: Hansen 2023 tutorial, K&W 2e 8.7.
struct cmaes_options
{
    double sigma_collapse_threshold{1e-12};  // sigma below this = roundoff_limited (Hansen tutorial)
    double condition_number_limit{1e14};     // covariance condition limit (Hansen tutorial)

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 6 (paper symbol: TolFun):
    // if the range of the best objective function values over the last
    // 10 + ceil(30 n / lambda) generations and the current generation's
    // offspring drops below this value, the policy exits with
    // solver_status::ftol_reached. Default: 1e-12 (Hansen 2023 page 34,
    // "10^-12 is a conservative first guess").
    double objective_value_tolerance{1e-12};

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 7 (paper symbol: TolX):
    // if the standard deviation of the sampling distribution drops below
    // `step_size_tolerance * initial_sigma` in all coordinates AND
    // sigma * p_c drops below the same threshold in all components, the
    // policy exits with solver_status::roundoff_limited. Default factor:
    // 1e-12 (multiplied by initial_sigma at the use site, per Hansen 2023
    // page 34, "By default we set TolX to 10^-12 times the initial sigma").
    double step_size_tolerance{1e-12};
};

}

#endif
