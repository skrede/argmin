#ifndef HPP_GUARD_NABLAPP_OPTIONS_QP_OPTIONS_H
#define HPP_GUARD_NABLAPP_OPTIONS_QP_OPTIONS_H

#include <cstdint>
#include <optional>

namespace nablapp
{

// Active-set QP solver parameters.
// Used by SLSQP and NW-SQP policies.
struct qp_options
{
    std::optional<std::uint16_t> max_iterations{};  // iteration limit (default: 200)
    std::optional<double> tolerance{};              // convergence tolerance (default: 1e-12)
};

}

#endif
