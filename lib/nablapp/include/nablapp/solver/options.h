#ifndef HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H
#define HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H

#include "nablapp/solver/convergence.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace nablapp
{

// Solver configuration carrying convergence policy and iteration limits.
//
// The Convergence template parameter is a convergence_policy<Criteria...>
// that composes criterion types via fold expression. Individual tolerance
// thresholds live in the criterion types, not here.

template <typename Convergence = default_convergence>
struct solver_options
{
    using convergence_type = Convergence;

    std::uint32_t max_iterations{1000};
    std::uint8_t verbosity{0};
    std::optional<std::chrono::nanoseconds> max_time{};
    std::optional<double> constraint_tolerance{};
    Convergence convergence{};
};

}

#endif
