#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_BOOTSTRAP_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_BOOTSTRAP_H

#include "argmin/detail/cobyla_engine.h"

#include <cstdint>
#include <algorithm>

namespace argmin::detail
{

template <typename Scalar>
inline typename cobyla_engine<Scalar>::vector_type cobyla_engine<Scalar>::origin_coords() const
{
    vector_type x(n_);
    for(int i = 1; i <= n_; ++i)
        x(i - 1) = sim_(i, np_);
    return x;
}

// A NaN coordinate compares unequal, so it counts as moved and the counter never under-reports.
template <typename Scalar>
inline void cobyla_engine<Scalar>::clamp_into_box(const vector_type& x)
{
    x_eval_ = x.cwiseMax(lower_).cwiseMin(upper_);
    if(!(x_eval_.array() == x.array()).all())
        ++clamped_evaluations_;
}

// The user is evaluated at the point clamped into the box while the bound rows read the raw
// point, so the model keeps an excursion's violation (NLopt cobyla.c, func_wrap).
template <typename Scalar>
inline void cobyla_engine<Scalar>::evaluate(const vector_type& x)
{
    clamp_into_box(x);
    vector_type conv(m_);
    Scalar f = eval_(x_eval_, x, conv);
    resmax_ = Scalar(0);
    for(int k = 1; k <= m_; ++k)
    {
        Scalar d = -conv(k - 1);
        resmax_ = std::max(resmax_, d);
        con_(k) = conv(k - 1);
    }
    con_(mp_) = f;
    con_(mpp_) = resmax_;
    f_last_ = f;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::bootstrap(const vector_type& x0)
{
    build_initial_simplex(x0);

    jdrop_ = np_;
    ibrnch_ = 0;
    nevals_ = 0;

    xcur_ = x0;
    evaluate_initial_vertices();

    ibrnch_ = 1;
    mirror_best();
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::build_initial_simplex(const vector_type& x0)
{
    for(int i = 1; i <= n_; ++i)
    {
        sim_(i, np_) = x0(i - 1);
        for(int j = 1; j <= n_; ++j)
        {
            sim_(i, j) = Scalar(0);
            simi_(i, j) = Scalar(0);
        }
        Scalar rhocur = initial_step_length(i, x0(i - 1));
        sim_(i, i) = rhocur;
        simi_(i, i) = Scalar(1) / rhocur;
    }
}

template <typename Scalar>
inline Scalar cobyla_engine<Scalar>::initial_step_length(std::int32_t i, Scalar xi) const
{
    Scalar rhocur = rho_;
    if(xi + rhocur > upper_(i - 1))
    {
        if(xi - rhocur >= lower_(i - 1))
            rhocur = -rhocur;
        else if(upper_(i - 1) - xi > xi - lower_(i - 1))
            rhocur = Scalar(0.5) * (upper_(i - 1) - xi);
        else
            rhocur = -Scalar(0.5) * (xi - lower_(i - 1));
    }
    return rhocur;
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::evaluate_initial_vertices()
{
    while(true)
    {
        evaluate(xcur_);
        ++nevals_;
        for(int k = 1; k <= mpp_; ++k)
            datmat_(k, jdrop_) = con_(k);

        if(nevals_ <= n_)
        {
            if(jdrop_ <= n_)
                exchange_initial_vertex();
            jdrop_ = nevals_;
            xcur_ = origin_coords();
            xcur_(jdrop_ - 1) += sim_(jdrop_, jdrop_);
            continue;
        }
        // nevals_ == np_: the final initial vertex.
        if(jdrop_ <= n_)
            exchange_initial_vertex();
        break;
    }
}

// Exchange the newly evaluated initial vertex with the optimal vertex if it
// improves the objective (Powell's initial-simplex completion).
template <typename Scalar>
inline void cobyla_engine<Scalar>::exchange_initial_vertex()
{
    const Scalar f = con_(mp_);
    if(datmat_(mp_, np_) <= f)
        return; // no improvement; leave origin in place

    Scalar rhocur = xcur_(jdrop_ - 1) - sim_(jdrop_, np_);
    sim_(jdrop_, np_) = xcur_(jdrop_ - 1);
    for(int k = 1; k <= mpp_; ++k)
    {
        datmat_(k, jdrop_) = datmat_(k, np_);
        datmat_(k, np_) = con_(k);
    }
    for(int k = 1; k <= jdrop_; ++k)
    {
        sim_(jdrop_, k) = -rhocur;
        Scalar temp = Scalar(0);
        for(int i = k; i <= jdrop_; ++i)
            temp -= simi_(i, k);
        simi_(jdrop_, k) = temp;
    }
}

}

#endif
