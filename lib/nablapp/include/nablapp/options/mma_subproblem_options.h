#ifndef HPP_GUARD_NABLAPP_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H
#define HPP_GUARD_NABLAPP_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H

#include <cstdint>
#include <optional>

namespace nablapp
{

// MMA subproblem solver parameters.
// Reference: Svanberg 1987.
struct mma_subproblem_options
{
    std::optional<double> regularization_epsilon{};      // coefficient regularization (default: 1e-7, Svanberg 1987)
    std::optional<std::uint16_t> dual_max_iterations{};  // dual solve iteration limit (default: 50)
    std::optional<double> dual_tolerance{};              // dual solve convergence (default: 1e-9)
    std::optional<double> backtrack_factor{};            // y backtrack multiplier (default: 0.95)
};

}

#endif
