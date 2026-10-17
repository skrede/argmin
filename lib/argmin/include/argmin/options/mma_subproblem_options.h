#ifndef HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H

#include "argmin/options/mma_defaults.h"

#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

// MMA subproblem solver parameters.
// Reference: Svanberg 1987.
template <scalar Scalar = double>
struct mma_subproblem_options
{
    std::uint16_t dual_max_iterations{50};                                        // dual solve iteration limit
    Scalar dual_tolerance{mma_defaults<Scalar>::subproblem_dual_tolerance};       // dual solve convergence
    Scalar backtrack_factor{mma_defaults<Scalar>::subproblem_backtrack_factor};  // y backtrack multiplier
};

}

#endif
