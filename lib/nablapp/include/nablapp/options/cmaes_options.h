#ifndef HPP_GUARD_NABLAPP_OPTIONS_CMAES_OPTIONS_H
#define HPP_GUARD_NABLAPP_OPTIONS_CMAES_OPTIONS_H

#include <cstdint>
#include <optional>

namespace nablapp
{

// CMA-ES strategy and convergence detection parameters.
// Reference: Hansen 2023 tutorial, K&W 2e 8.7.
struct cmaes_options
{
    std::optional<double> sigma_collapse_threshold{};  // sigma below this = roundoff_limited (default: 1e-12, Hansen tutorial)
    std::optional<double> condition_number_limit{};    // covariance condition limit (default: 1e14, Hansen tutorial)
    std::optional<std::uint32_t> stagnation_limit{};   // override auto formula 10+ceil(30*n/lambda) (default: nullopt = auto)

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 6 (paper symbol: TolFun):
    // if the range of the best objective function values over the last
    // 10 + ceil(30 n / lambda) generations and the current generation's
    // offspring drops below this value, the policy exits with
    // solver_status::ftol_reached. Default: 1e-12 (Hansen 2023 page 34,
    // "10^-12 is a conservative first guess").
    std::optional<double> objective_value_tolerance{};

    // Hansen 2023 (arXiv:1604.00772) section B.3 item 7 (paper symbol: TolX):
    // if the standard deviation of the sampling distribution drops below
    // `step_size_tolerance * initial_sigma` in all coordinates AND
    // sigma * p_c drops below the same threshold in all components, the
    // policy exits with solver_status::roundoff_limited. Default factor:
    // 1e-12 (multiplied by initial_sigma at the use site, per Hansen 2023
    // page 34, "By default we set TolX to 10^-12 times the initial sigma").
    std::optional<double> step_size_tolerance{};
};

}

#endif
