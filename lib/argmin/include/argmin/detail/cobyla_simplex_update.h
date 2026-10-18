#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_UPDATE_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_UPDATE_H

#include "argmin/detail/cobyla_engine.h"

#include "argmin/result/status.h"

#include <cmath>
#include <tuple>
#include <cstdint>
#include <utility>
#include <algorithm>

namespace argmin::detail
{

// L440: decide whether the trial replaces a simplex vertex, revise SIM /
// SIMI / DATMAT, and choose whether to keep iterating or reduce RHO.
template <typename Scalar>
inline typename cobyla_engine<Scalar>::l440_action cobyla_engine<Scalar>::process_trial()
{
    Scalar trured = trial_reduction();
    drop_by_ratio(trured);
    const std::int32_t l = drop_by_edge(trured);
    if(l > 0)
        jdrop_ = l;
    if(jdrop_ == 0)
        return l440_action::reduce_rho;
    replace_dropped_vertex();

    if(trured > Scalar(0) && trured >= prerem_ * Scalar(0.1))
    {
        if(trured >= prerem_ * Scalar(0.9) && trured <= prerem_ * Scalar(1.1) && iflag_ != 0)
            rho_ *= Scalar(2);
        return l440_action::back_to_main;
    }
    return l440_action::reduce_rho;
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::trial_reduction()
{
    const Scalar f = con_(mp_);
    Scalar vmold = datmat_(mp_, np_) + parmu_ * datmat_(mpp_, np_);
    Scalar vmnew = f + parmu_ * resmax_;
    Scalar trured = vmold - vmnew;
    if(parmu_ == Scalar(0) && f == datmat_(mp_, np_))
    {
        prerem_ = prerec_;
        trured = datmat_(mpp_, np_) - resmax_;
    }
    return trured;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::drop_by_ratio(Scalar trured)
{
    Scalar ratio = (trured <= Scalar(0)) ? Scalar(1) : Scalar(0);
    jdrop_ = 0;
    for(int j = 1; j <= n_; ++j)
    {
        Scalar temp = Scalar(0);
        for(int i = 1; i <= n_; ++i)
            temp += simi_(j, i) * dx_(i - 1);
        temp = std::abs(temp);
        if(temp > ratio)
        {
            jdrop_ = j;
            ratio = temp;
        }
        sigbar_(j) = temp * vsig_(j);
    }
}

template <typename Scalar>
inline std::int32_t cobyla_engine<Scalar>::drop_by_edge(Scalar trured) const
{
    Scalar edgmax = Scalar(1.1) * rho_;
    int l = 0;
    for(int j = 1; j <= n_; ++j)
    {
        if(sigbar_(j) >= parsig_ || sigbar_(j) >= vsig_(j))
        {
            Scalar temp = veta_(j);
            if(trured > Scalar(0))
                temp = trial_edge_length(j);
            if(temp > edgmax)
            {
                l = j;
                edgmax = temp;
            }
        }
    }
    return l;
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::trial_edge_length(std::int32_t j) const
{
    Scalar temp = Scalar(0);
    for(int i = 1; i <= n_; ++i)
    {
        Scalar d = dx_(i - 1) - sim_(i, j);
        temp += d * d;
    }
    return std::sqrt(temp);
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::replace_dropped_vertex()
{
    Scalar temp = Scalar(0);
    for(int i = 1; i <= n_; ++i)
    {
        sim_(i, jdrop_) = dx_(i - 1);
        temp += simi_(jdrop_, i) * dx_(i - 1);
    }
    for(int i = 1; i <= n_; ++i)
        simi_(jdrop_, i) /= temp;
    for(int j = 1; j <= n_; ++j)
    {
        if(j != jdrop_)
        {
            Scalar t = Scalar(0);
            for(int i = 1; i <= n_; ++i)
                t += simi_(j, i) * dx_(i - 1);
            for(int i = 1; i <= n_; ++i)
                simi_(j, i) -= t * simi_(jdrop_, i);
        }
    }
    for(int k = 1; k <= mpp_; ++k)
        datmat_(k, jdrop_) = con_(k);
}

// L550: rebuild an unacceptable simplex, or reduce RHO and re-derive PARMU;
// terminate once RHO reaches RHOEND.
template <typename Scalar>
inline typename cobyla_engine<Scalar>::l550_result cobyla_engine<Scalar>::l550()
{
    if(iflag_ == 0)
    {
        ibrnch_ = 0;
        return l550_result::continue_iter;
    }
    if(rho_ > rhoend_)
    {
        rho_ *= Scalar(0.5);
        if(rho_ <= rhoend_ * Scalar(1.5))
            rho_ = rhoend_;
        if(parmu_ > Scalar(0))
            rederive_parmu();
        return l550_result::continue_iter;
    }
    converged_ = true;
    status_ = solver_status::xtol_reached;
    return l550_result::converged;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::rederive_parmu()
{
    Scalar denom = Scalar(0);
    Scalar cmin = Scalar(0);
    Scalar cmax = Scalar(0);
    for(int k = 1; k <= mp_; ++k)
    {
        std::tie(cmin, cmax) = row_range(k);
        if(k <= m_ && cmin < cmax * Scalar(0.5))
        {
            Scalar t = std::max(cmax, Scalar(0)) - cmin;
            if(denom <= Scalar(0))
                denom = t;
            else
                denom = std::min(denom, t);
        }
    }
    if(denom == Scalar(0))
        parmu_ = Scalar(0);
    else if(cmax - cmin < parmu_ * denom)
        parmu_ = (cmax - cmin) / denom;
}

template <typename Scalar>
inline std::pair<Scalar, Scalar> cobyla_engine<Scalar>::row_range(std::int32_t k) const
{
    Scalar cmin = datmat_(k, np_);
    Scalar cmax = cmin;
    for(int i = 1; i <= n_; ++i)
    {
        Scalar v = datmat_(k, i);
        cmin = std::min(cmin, v);
        cmax = std::max(cmax, v);
    }
    return {cmin, cmax};
}

}

#endif
