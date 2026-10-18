#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRIAL_STEP_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRIAL_STEP_H

#include "argmin/detail/cobyla_engine.h"
#include "argmin/detail/cobyla_trust_region.h"

#include <algorithm>

namespace argmin::detail
{

template <typename Scalar>
inline void cobyla_engine<Scalar>::run_trstlp()
{
    matrix_type Amat(n_, mp_);
    // The reference passes an oversized right-hand side (Powell's con array
    // holds the objective and violation slots after the m constraints), and
    // TRSTLP reads b at the stage-two objective index. Mirror that extent so
    // the objective pseudo-constraint's slot is addressable.
    vector_type bvec(mpp_);
    for(int i = 1; i <= n_; ++i)
        for(int k = 1; k <= mp_; ++k)
            Amat(i - 1, k - 1) = a_(i, k);
    for(int k = 1; k <= mpp_; ++k)
        bvec(k - 1) = con_(k);
    cobyla_trstlp<Scalar>(n_, m_, Amat, bvec, rho_, dx_, ifull_);
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::clamp_dx_to_bounds()
{
    for(int i = 1; i <= n_; ++i)
    {
        Scalar xi = sim_(i, np_);
        if(xi + dx_(i - 1) > upper_(i - 1))
            dx_(i - 1) = upper_(i - 1) - xi;
        if(xi + dx_(i - 1) < lower_(i - 1))
            dx_(i - 1) = xi - lower_(i - 1);
    }
}

// Predict the new maximum violation (resnew), the reduction in the maximum
// violation (prerec_), and the objective increase (sum_) along dx.
template <typename Scalar>
inline void cobyla_engine<Scalar>::compute_prediction()
{
    Scalar resnew = Scalar(0);
    con_(mp_) = Scalar(0);
    Scalar sumv = Scalar(0);
    for(int k = 1; k <= mp_; ++k)
    {
        sumv = con_(k);
        for(int i = 1; i <= n_; ++i)
            sumv -= a_(i, k) * dx_(i - 1);
        if(k < mp_)
            resnew = std::max(resnew, sumv);
    }
    sum_ = sumv; // objective term (k == mp_): predicted objective increase
    prerec_ = datmat_(mpp_, np_) - resnew;
}

// After a PARMU increase, report whether a different vertex now has the
// lower merit (in which case we must re-identify the optimal vertex).
template <typename Scalar>
inline bool cobyla_engine<Scalar>::parmu_raise_changes_optimal() const
{
    Scalar phi = datmat_(mp_, np_) + parmu_ * datmat_(mpp_, np_);
    for(int j = 1; j <= n_; ++j)
    {
        Scalar temp = datmat_(mp_, j) + parmu_ * datmat_(mpp_, j);
        if(temp < phi)
            return true;
        if(temp == phi && parmu_ == Scalar(0))
            if(datmat_(mpp_, j) < datmat_(mpp_, np_))
                return true;
    }
    return false;
}

}

#endif
