#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_GEOMETRY_STEP_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_GEOMETRY_STEP_H

#include "argmin/detail/cobyla_engine.h"

#include <cstdint>
#include <algorithm>

namespace argmin::detail
{

// ---- Deterministic LCG used to jitter geometry (simplex) steps ----
template <typename Scalar>
inline std::uint32_t cobyla_engine<Scalar>::lcg_rand()
{
    return (seed_ = seed_ * std::uint32_t(1103515245) + std::uint32_t(12345));
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::lcg_urand(Scalar a, Scalar b)
{
    return a + static_cast<Scalar>(lcg_rand())
        * (b - a) / static_cast<Scalar>(static_cast<std::uint32_t>(-1));
}

// Geometry-improving vertex: replace the worst-poised vertex with a point
// that restores acceptability, evaluate it, and store it in the simplex.
template <typename Scalar>
inline void cobyla_engine<Scalar>::geometry_step()
{
    select_geometry_vertex();

    Scalar temp = Scalar(0.5) * rho_ * vsig_(jdrop_);
    for(int i = 1; i <= n_; ++i)
        dx_(i - 1) = temp * simi_(jdrop_, i);

    Scalar dxsign = geometry_step_sign();
    temp = fit_geometry_step(dxsign);
    apply_geometry_step(temp);

    evaluate(xcur_);
    ++nevals_;
    for(int k = 1; k <= mpp_; ++k)
        datmat_(k, jdrop_) = con_(k);
    ibrnch_ = 1;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::select_geometry_vertex()
{
    jdrop_ = 0;
    Scalar temp = pareta_;
    for(int j = 1; j <= n_; ++j)
        if(veta_(j) > temp)
        {
            jdrop_ = j;
            temp = veta_(j);
        }
    if(jdrop_ == 0)
    {
        for(int j = 1; j <= n_; ++j)
            if(vsig_(j) < temp)
            {
                jdrop_ = j;
                temp = vsig_(j);
            }
    }
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::geometry_step_sign() const
{
    Scalar cvmaxp = Scalar(0);
    Scalar cvmaxm = Scalar(0);
    Scalar sumg = Scalar(0);
    for(int k = 1; k <= mp_; ++k)
    {
        sumg = Scalar(0);
        for(int i = 1; i <= n_; ++i)
            sumg += a_(i, k) * dx_(i - 1);
        if(k < mp_)
        {
            Scalar t = datmat_(k, np_);
            cvmaxp = std::max(cvmaxp, -sumg - t);
            cvmaxm = std::max(cvmaxm, sumg - t);
        }
    }
    Scalar dxsign = (parmu_ * (cvmaxp - cvmaxm) > sumg + sumg) ? Scalar(-1) : Scalar(1);
    return dxsign;
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::fit_geometry_step(Scalar dxsign)
{
    Scalar temp = Scalar(0);
    for(int i = 1; i <= n_; ++i)
    {
        dx_(i - 1) = dxsign * dx_(i - 1) * lcg_urand(Scalar(0.01), Scalar(1));
        fit_coordinate_to_bounds(i);
        sim_(i, jdrop_) = dx_(i - 1);
        temp += simi_(jdrop_, i) * dx_(i - 1);
    }
    return temp;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::fit_coordinate_to_bounds(std::int32_t i)
{
    Scalar xi = sim_(i, np_);
    for(;;)
    {
        if(xi + dx_(i - 1) > upper_(i - 1))
            dx_(i - 1) = -dx_(i - 1);
        if(xi + dx_(i - 1) < lower_(i - 1))
        {
            if(xi - dx_(i - 1) <= upper_(i - 1))
                dx_(i - 1) = -dx_(i - 1);
            else
            {
                dx_(i - 1) *= Scalar(0.5);
                continue;
            }
        }
        break;
    }
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::apply_geometry_step(Scalar temp)
{
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
        xcur_(j - 1) = sim_(j, np_) + dx_(j - 1);
    }
}

}

#endif
