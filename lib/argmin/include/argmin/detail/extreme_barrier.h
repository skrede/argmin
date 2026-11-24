#ifndef HPP_GUARD_ARGMIN_DETAIL_EXTREME_BARRIER_H
#define HPP_GUARD_ARGMIN_DETAIL_EXTREME_BARRIER_H

#include "argmin/types.h"

#include <cmath>
#include <limits>
#include <algorithm>

namespace argmin::detail
{

// The value a model-based derivative-free method stores for a non-finite evaluation, PRIMA's moderated extreme barrier
// (Zhang, HUGEFUN = 2^min(100, maxexponent / 2)): finite, so the model stays defined, and far above the values it
// usually fits, so the trial ranks worst and the trust region shrinks. A finite evaluation can exceed it, so a method
// records which values it moderated instead of reading that from their magnitude.
template <scalar Scalar>
Scalar extreme_barrier()
{
    return std::ldexp(Scalar(1), std::min(100, std::numeric_limits<Scalar>::max_exponent / 2));
}

template <scalar Scalar>
Scalar moderated(Scalar value)
{
    return std::isfinite(value) ? value : extreme_barrier<Scalar>();
}

template <scalar Scalar>
Scalar moderated(Scalar value, bool& non_finite_seen)
{
    non_finite_seen = non_finite_seen || !std::isfinite(value);
    return moderated(value);
}

// A row c >= 0 that evaluated non-finite is violated by the barrier.
template <scalar Scalar>
Scalar moderated_row(Scalar c)
{
    return std::isfinite(c) ? c : -extreme_barrier<Scalar>();
}

}

#endif
