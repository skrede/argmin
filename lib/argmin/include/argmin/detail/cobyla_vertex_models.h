#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_VERTEX_MODELS_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_VERTEX_MODELS_H

#include "argmin/detail/cobyla_engine.h"

#include "argmin/result/status.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <algorithm>

namespace argmin::detail
{

// L140: identify the optimal vertex and switch it into pole position.
template <typename Scalar>
inline void cobyla_engine<Scalar>::identify_optimal()
{
    const std::int32_t nbest = select_optimal_vertex();
    if(nbest <= n_)
        move_to_pole(nbest);
}

template <typename Scalar>
inline std::int32_t cobyla_engine<Scalar>::select_optimal_vertex() const
{
    Scalar phimin = datmat_(mp_, np_) + parmu_ * datmat_(mpp_, np_);
    int nbest = np_;
    for(int j = 1; j <= n_; ++j)
    {
        Scalar temp = datmat_(mp_, j) + parmu_ * datmat_(mpp_, j);
        if(temp < phimin)
        {
            nbest = j;
            phimin = temp;
        }
        else if(temp == phimin && parmu_ == Scalar(0))
        {
            if(datmat_(mpp_, j) < datmat_(mpp_, nbest))
                nbest = j;
        }
    }
    return nbest;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::move_to_pole(std::int32_t nbest)
{
    for(int i = 1; i <= mpp_; ++i)
        std::swap(datmat_(i, np_), datmat_(i, nbest));
    for(int i = 1; i <= n_; ++i)
    {
        Scalar temp = sim_(i, nbest);
        sim_(i, nbest) = Scalar(0);
        sim_(i, np_) += temp;
        Scalar tempa = Scalar(0);
        for(int k = 1; k <= n_; ++k)
        {
            sim_(i, k) -= temp;
            tempa -= simi_(k, i);
        }
        simi_(nbest, i) = tempa;
    }
}

// Guard against SIMI drifting away from the true inverse of SIM.
template <typename Scalar>
inline bool cobyla_engine<Scalar>::check_simi_error()
{
    Scalar error = Scalar(0);
    for(int i = 1; i <= n_; ++i)
    {
        for(int j = 1; j <= n_; ++j)
        {
            Scalar temp = (i == j) ? Scalar(-1) : Scalar(0);
            for(int k = 1; k <= n_; ++k)
                if(sim_(k, j) != Scalar(0))
                    temp += simi_(i, k) * sim_(k, j);
            error = std::max(error, std::abs(temp));
        }
    }
    if(error > Scalar(0.1))
    {
        converged_ = true;
        status_ = solver_status::roundoff_limited;
        mirror_best();
        return true;
    }
    return false;
}

// Linear approximations: constraint gradients in a_(.,1..m), minus the
// objective gradient in a_(.,mp_). con_(k) is set to -value at the origin.
template <typename Scalar>
inline void cobyla_engine<Scalar>::build_models()
{
    for(int k = 1; k <= mp_; ++k)
    {
        con_(k) = -datmat_(k, np_);
        for(int j = 1; j <= n_; ++j)
            w_(j) = datmat_(k, j) + con_(k);
        for(int i = 1; i <= n_; ++i)
        {
            Scalar temp = Scalar(0);
            for(int j = 1; j <= n_; ++j)
                temp += w_(j) * simi_(j, i);
            if(k == mp_)
                temp = -temp;
            a_(i, k) = temp;
        }
    }
}

// Acceptability measures vsig / veta; iflag_ == 0 marks an unacceptable
// simplex (a vertex too close, or an edge too long, relative to rho).
template <typename Scalar>
inline void cobyla_engine<Scalar>::compute_vsig_veta()
{
    iflag_ = 1;
    parsig_ = Scalar(0.25) * rho_;
    pareta_ = Scalar(2.1) * rho_;
    for(int j = 1; j <= n_; ++j)
    {
        Scalar wsig = Scalar(0);
        Scalar weta = Scalar(0);
        for(int i = 1; i <= n_; ++i)
        {
            wsig += simi_(j, i) * simi_(j, i);
            weta += sim_(i, j) * sim_(i, j);
        }
        vsig_(j) = Scalar(1) / std::sqrt(wsig);
        veta_(j) = std::sqrt(weta);
        if(vsig_(j) < parsig_ || veta_(j) > pareta_)
            iflag_ = 0;
    }
}

// Mirror the current best (pole) vertex into (x_best_, f_best_,
// resmax_best_). The pole is Powell's incumbent x*: it is promoted to the
// lowest-merit vertex by identify_optimal at the top of each iteration, so
// reporting it (rather than a fresh merit re-scan) keeps the reported
// objective and the merit-penalty adaptation in step with the reference.
template <typename Scalar>
inline void cobyla_engine<Scalar>::mirror_best()
{
    for(int i = 1; i <= n_; ++i)
        x_best_(i - 1) = sim_(i, np_);
    f_best_ = datmat_(mp_, np_);
    resmax_best_ = datmat_(mpp_, np_);
}

}

#endif
