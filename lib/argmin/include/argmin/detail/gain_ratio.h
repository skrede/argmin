#ifndef HPP_GUARD_ARGMIN_DETAIL_GAIN_RATIO_H
#define HPP_GUARD_ARGMIN_DETAIL_GAIN_RATIO_H

// Guarded gain ratio for the least-squares trust-region / damping policies.
//
// The gain ratio rho = actual / predicted compares the achieved objective
// reduction to the reduction the local model predicted. lm_policy,
// projected_gn_policy and projected_gradient_gn_policy all feed it the direct
// model reduction predicted = -g^T d - 0.5||J d||^2 (a strictly positive
// quantity for a genuine descent step) and read the result as an accept /
// damping signal. This peer of the predicted-reduction helpers in detail/
// centralizes the shared guard so the three sites stay identical.
//
// Reference: Nielsen, H. B. (1999) "Damping Parameter in Marquardt's Method",
//            IMM-REP-1999-05 (the gain-ratio accept test); N&W 2e Section 4.1
//            (the trust-region ratio rho_k).

#include <cmath>

namespace argmin::detail
{

// Guarded gain ratio. Only a strictly positive predicted model reduction with a
// finite actual reduction yields a meaningful ratio; every other case returns 0,
// which the callers read as a reject / damping-increase signal.
//
// Gating on the finiteness of `actual` also gates a non-finite trial objective:
// callers compute actual = f(x) - f(x_trial) with f(x) finite, so a NaN or Inf
// trial makes `actual` non-finite and the ratio collapses to 0.
inline double gain_ratio(double actual, double predicted)
{
    const bool valid = std::isfinite(actual) && predicted > 0.0;
    return valid ? actual / predicted : 0.0;
}

}

#endif
