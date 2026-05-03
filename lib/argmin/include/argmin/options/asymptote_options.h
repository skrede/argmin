#ifndef HPP_GUARD_ARGMIN_OPTIONS_ASYMPTOTE_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_ASYMPTOTE_OPTIONS_H

#include <optional>

namespace argmin
{

// MMA/GCMMA asymptote update parameters.
//
// Asymptote-interval clamping enforces the Svanberg 2002 CCSA convergence
// preconditions by keeping (U - L) inside a paper-literal bracket:
//
//   minimum_distance_fraction * range <= (U_j - L_j) <= maximum_distance_fraction * range
//
// where range = (x_max - x_min) for bounded dimensions and a scale-adaptive
// fallback for unbounded dimensions. The lower clamp prevents asymptote
// collapse onto the iterate (approximation denominators diverge). The upper
// clamp prevents the asym_inc expansion factor from running away on
// monotone-descent iterate sequences, which otherwise overshoots near a
// converged iterate and produces the "early-convergence-then-destabilization"
// trajectory documented in this repo's MMA/GCMMA diagnosis traces.
//
// Reference: Svanberg 1987, "The method of moving asymptotes",
//            IJNME 24:359-373 (lower clamp, Section 3);
//            Svanberg 2002, "A Class of Globally Convergent Optimization
//            Methods Based on Conservative Convex Separable Approximations",
//            SIAM J. Optim. 12(2):555-573 (upper clamp, Section 4.2);
//            arjendeetman/GCMMA-MMA-Python mmasub (symmetric paper port;
//            albefalow = 0.01, albefaup = 10 default).
struct asymptote_options
{
    std::optional<double> minimum_distance_fraction{};  // min (U - L) / range (default: 0.01, Svanberg 1987)
    std::optional<double> maximum_distance_fraction{};  // max (U - L) / range (default: 10.0,  Svanberg 2002)
};

}

#endif
