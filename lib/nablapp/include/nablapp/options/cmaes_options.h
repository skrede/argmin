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
};

}

#endif
