#ifndef HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H

#include <cstdint>

namespace argmin
{

// MMA subproblem solver parameters.
// Reference: Svanberg 1987.
struct mma_subproblem_options
{
    std::uint16_t dual_max_iterations{50};  // dual solve iteration limit
    double dual_tolerance{1e-9};            // dual solve convergence
    double backtrack_factor{0.95};          // y backtrack multiplier
};

}

#endif
