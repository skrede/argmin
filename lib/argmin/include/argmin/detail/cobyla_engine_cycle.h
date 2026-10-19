#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_ENGINE_CYCLE_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_ENGINE_CYCLE_H

#include "argmin/detail/cobyla_engine.h"

#include "argmin/result/status.h"

#include <cstdint>
#include <utility>

namespace argmin::detail
{

template <typename Scalar>
inline cobyla_engine<Scalar>::cobyla_engine()
    : n_{0}
    , m_{0}
    , mp_{0}
    , mpp_{0}
    , np_{0}
    , rho_{0}
    , rho_initial_{0}
    , rhoend_{0}
    , parmu_{0}
    , parsig_{0}
    , pareta_{0}
    , resmax_{0}
    , f_last_{0}
    , prerec_{0}
    , prerem_{0}
    , sum_{0}
    , ibrnch_{0}
    , iflag_{0}
    , jdrop_{0}
    , ifull_{1}
    , nevals_{0}
    , seed_{0}
    , clamped_evaluations_{0}
    , converged_{false}
    , status_{solver_status::running}
    , f_best_{0}
    , resmax_best_{0}
    , eval_{}
{
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::init(int n, int m, const vector_type& x0, const vector_type& lower, const vector_type& upper, Scalar rhobeg, Scalar rhoend, evaluator eval)
{
    n_ = n;
    m_ = m;
    mp_ = m + 1;
    mpp_ = m + 2;
    np_ = n + 1;
    lower_ = lower;
    upper_ = upper;
    rhoend_ = rhoend;
    rho_ = rhobeg;
    rho_initial_ = rhobeg;
    parmu_ = Scalar(0);
    clamped_evaluations_ = 0;
    eval_ = std::move(eval);
    converged_ = false;
    status_ = solver_status::running;
    seed_ = static_cast<std::uint32_t>(n_ + m_);

    size_workspace();

    bootstrap(x0);
}

template <typename Scalar>
inline void cobyla_engine<Scalar>::size_workspace()
{
    sim_.setZero(n_ + 1, n_ + 2);
    simi_.setZero(n_ + 1, n_ + 1);
    datmat_.setZero(mpp_ + 1, n_ + 2);
    a_.setZero(n_ + 1, mp_ + 1);
    con_.setZero(mpp_ + 1);
    vsig_.setZero(n_ + 1);
    veta_.setZero(n_ + 1);
    sigbar_.setZero(n_ + 1);
    w_.setZero(n_ + 1);
    dx_.setZero(n_);
    xcur_.setZero(n_);
    x_eval_.setZero(n_);
    x_best_.setZero(n_);
}

template <typename Scalar>
inline bool cobyla_engine<Scalar>::advance()
{
    if(converged_)
        return false;
    while(true)
    {
        identify_optimal();
        if(check_simi_error())
            return false;
        build_models();
        compute_vsig_veta();

        const bool geometry_branch = (ibrnch_ == 0 && iflag_ == 0);
        if(geometry_branch)
        {
            geometry_step();
            mirror_best();
            return true;
        }
        const cycle_outcome outcome = trust_region_cycle();
        if(outcome != cycle_outcome::reiterate)
            return outcome == cycle_outcome::stepped;
    }
}

// Two-stage linear trust-region subproblem.
template <typename Scalar>
inline typename cobyla_engine<Scalar>::cycle_outcome cobyla_engine<Scalar>::trust_region_cycle()
{
    run_trstlp();
    clamp_dx_to_bounds();
    if(ifull_ == 0)
    {
        Scalar dsq = dx_.squaredNorm();
        if(dsq < rho_ * Scalar(0.25) * rho_)
        {
            ibrnch_ = 1;
            return l550() == l550_result::converged ? cycle_outcome::terminated : cycle_outcome::reiterate;
        }
    }
    compute_prediction();
    Scalar barmu = (prerec_ > Scalar(0)) ? sum_ / prerec_ : Scalar(0);
    if(parmu_ < barmu * Scalar(1.5))
    {
        parmu_ = barmu * Scalar(2);
        if(parmu_raise_changes_optimal())
            return cycle_outcome::reiterate;
    }
    prerem_ = parmu_ * prerec_ - sum_;
    return evaluate_trial();
}

template <typename Scalar>
inline typename cobyla_engine<Scalar>::cycle_outcome cobyla_engine<Scalar>::evaluate_trial()
{
    for(int i = 1; i <= n_; ++i)
        xcur_(i - 1) = sim_(i, np_) + dx_(i - 1);
    ibrnch_ = 1;
    evaluate(xcur_);

    l440_action act = process_trial();
    if(act == l440_action::reduce_rho)
    {
        if(l550() == l550_result::converged)
        {
            mirror_best();
            return cycle_outcome::terminated;
        }
    }
    mirror_best();
    return cycle_outcome::stepped;
}

}

#endif
