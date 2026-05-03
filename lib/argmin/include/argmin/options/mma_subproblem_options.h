#ifndef HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_MMA_SUBPROBLEM_OPTIONS_H

#include <cstdint>
#include <optional>

namespace argmin
{

// MMA subproblem solver parameters.
// Reference: Svanberg 1987.
struct mma_subproblem_options
{
    // Division-hygiene floor on p_0j / (U - x) and q_0j / (x - L) in the
    // separable approximation. NOT the primary regularizer; Svanberg 2002
    // Section 3 specifies the 0.001 * |grad_f| stabilizer (baseline, always
    // on) and the raa_0 / (U - L) conservativity term (GCMMA-only, grows
    // adaptively) as the load-bearing regularization, both inlined inside
    // compute_coefficients. This epsilon only prevents pathological
    // denominator collapse when an iterate approaches an asymptote.
    std::optional<double> regularization_epsilon{};      // default: 1e-10 (see note above)
    std::optional<std::uint16_t> dual_max_iterations{};  // dual solve iteration limit (default: 50)
    std::optional<double> dual_tolerance{};              // dual solve convergence (default: 1e-9)
    std::optional<double> backtrack_factor{};            // y backtrack multiplier (default: 0.95)
};

}

#endif
