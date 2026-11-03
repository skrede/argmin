#ifndef HPP_GUARD_ARGMIN_DETAIL_FINITE_ORDER_H
#define HPP_GUARD_ARGMIN_DETAIL_FINITE_ORDER_H

#include <cmath>

namespace argmin::detail
{

// Every non-finite value falls in one equivalence class ranked after all finite values, so this is a strict weak order
// even when NaN is present, where the raw < is not.
template <typename Scalar>
bool finite_first(Scalar a, Scalar b)
{
    return std::isfinite(a) && (!std::isfinite(b) || a < b);
}

}

#endif
