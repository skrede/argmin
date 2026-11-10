#ifndef HPP_GUARD_ARGMIN_DETAIL_GAIN_RATIO_H
#define HPP_GUARD_ARGMIN_DETAIL_GAIN_RATIO_H

// Guarded gain ratio for the least-squares trust-region / damping policies, and the roundoff floor of the
// trust-region ratio test.
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

#include "argmin/types.h"

#include <cmath>
#include <limits>
#include <algorithm>

namespace argmin::detail
{

// Guarded gain ratio. Only a strictly positive predicted model reduction with a
// finite actual reduction yields a meaningful ratio; every other case returns 0,
// which the callers read as a reject / damping-increase signal.
//
// Gating on the finiteness of `actual` also gates a non-finite trial objective:
// callers compute actual = f(x) - f(x_trial) with f(x) finite, so a NaN or Inf
// trial makes `actual` non-finite and the ratio collapses to 0.
template <scalar Scalar>
Scalar gain_ratio(Scalar actual, Scalar predicted)
{
    const bool valid = std::isfinite(actual) && predicted > Scalar(0);
    return valid ? actual / predicted : Scalar(0);
}

// Below eps * max(1, ||x||) a step moves x only within its relative machine precision, so that radius bounds the
// configured floor from below (Conn, Gould and Toint, Trust-Region Methods, SIAM 2000, Section 17.4: termination on
// the trust-region radius).
template <scalar Scalar>
Scalar trust_radius_floor(Scalar min_radius, Scalar x_norm)
{
    return std::max(min_radius, std::numeric_limits<Scalar>::epsilon() * std::max(Scalar(1), x_norm));
}

// A rejected trial whose actual and predicted merit reductions both lie within the merit's rounding error, bounded
// as IPOPT's Compare_le bounds it by 10 eps |merit| (Ipopt, src/Common/IpUtils.cpp), is noise to the ratio test
// (Conn, Gould and Toint, Section 17.4), and a smaller region only shrinks the prediction, so no radius left to try
// yields an acceptable step.
template <scalar Scalar>
bool reductions_within_rounding(Scalar actual, Scalar predicted, Scalar merit)
{
    const Scalar rounding = Scalar(10) * std::numeric_limits<Scalar>::epsilon() * std::abs(merit);
    return std::abs(actual) <= rounding && std::abs(predicted) <= rounding;
}

}

#endif
