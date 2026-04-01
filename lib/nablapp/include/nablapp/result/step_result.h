#ifndef HPP_GUARD_NABLAPP_RESULT_STEP_RESULT_H
#define HPP_GUARD_NABLAPP_RESULT_STEP_RESULT_H

#include "nablapp/result/status.h"

#include <optional>

namespace nablapp
{

// Metrics from a single solver iteration.
//
// basic_solver inspects these after each Policy::step() to decide
// convergence, stall detection, and divergence.

template <typename Scalar = double>
struct step_result
{
    Scalar objective_value{};
    Scalar gradient_norm{};
    Scalar step_size{};
    Scalar objective_change{};
    bool improved{false};
    Scalar constraint_violation{};
    Scalar x_norm{};
    std::optional<solver_status> policy_status{};
};

}

#endif
