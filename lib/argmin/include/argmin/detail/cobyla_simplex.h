#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H

// Simplex driver engine for COBYLA (Powell 1994).
//
// COBYLA maintains a simplex of n+1 vertices in R^n. The last column of SIM
// holds the current best (optimal) vertex; the preceding n columns hold the
// displacements from the optimal vertex to the other vertices, and SIMI holds
// the inverse of that n-by-n displacement matrix. DATMAT stores, per vertex,
// the constraint values, the objective, and the greatest constraint violation.
//
// Each outer iteration builds linear approximations of the objective and every
// constraint by interpolation on the simplex, solves the two-stage linear
// trust-region subproblem (TRSTLP), evaluates the trial point, adapts the
// merit penalty PARMU toward feasibility only when reaching feasibility costs
// objective, and either revises the simplex or maintains its geometry. The
// trust radius RHO is reduced from RHOBEG to RHOEND, and termination occurs
// once RHO reaches RHOEND after a short, geometrically acceptable step.
//
// The merit function is  phi(x) = f(x) + parmu * RESMAX(x)  where RESMAX is the
// MAXIMUM constraint violation (all constraints are written in the >= 0 sense).
//
// This engine is a faithful port of M. J. D. Powell's COBYLA2 (the cobylb
// routine), restructured as a resumable state machine: init() bootstraps the
// simplex with n+1 evaluations, and each advance() performs one further
// function evaluation (a geometry-improving vertex or a trust-region trial)
// together with the surrounding bookkeeping, until RHO reaches RHOEND.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization method that
//            models the objective and constraint functions by linear
//            interpolation." Sections 2-5.

#include "argmin/detail/cobyla_trust_region.h"
#include "argmin/result/status.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace argmin::detail
{

// A single COBYLA solve, driven one evaluation at a time.
//
// The evaluator fills the length-m constraint vector in Powell's convention
// (every entry must become non-negative) and returns the objective value.
template <typename Scalar = double>
class cobyla_engine
{
public:
    using vector_type = Eigen::Vector<Scalar, argmin::dynamic_dimension>;
    using matrix_type = Eigen::Matrix<Scalar, argmin::dynamic_dimension, argmin::dynamic_dimension>;
    using evaluator = std::function<Scalar(const vector_type&, vector_type&)>;

    cobyla_engine() = default;

    // n      : number of variables.
    // m      : number of Powell constraints (all >= 0 when feasible).
    // x0     : starting point (already projected into the box).
    // lower  : lower bounds (may be -inf).
    // upper  : upper bounds (may be +inf).
    // rhobeg : initial trust radius.
    // rhoend : final trust radius.
    // eval   : (x, con_out[m]) -> f. Fills con_out (size m) and returns f.
    void init(int n, int m,
              const vector_type& x0,
              const vector_type& lower,
              const vector_type& upper,
              Scalar rhobeg, Scalar rhoend,
              evaluator eval)
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
        eval_ = std::move(eval);
        converged_ = false;
        status_ = solver_status::running;
        seed_ = static_cast<std::uint32_t>(n_ + m_);

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
        x_best_.setZero(n_);

        bootstrap(x0);
    }

    // Perform one further evaluation cycle. Returns true while iterating and
    // false once the solve has terminated (RHO reached RHOEND, or a
    // rounding-limited breakdown).
    bool advance()
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

            // Two-stage linear trust-region subproblem.
            run_trstlp();
            clamp_dx_to_bounds();

            if(ifull_ == 0)
            {
                Scalar dsq = dx_.squaredNorm();
                if(dsq < rho_ * Scalar(0.25) * rho_)
                {
                    ibrnch_ = 1;
                    if(l550() == l550_result::converged)
                        return false;
                    continue;
                }
            }

            compute_prediction();

            Scalar barmu = (prerec_ > Scalar(0)) ? sum_ / prerec_ : Scalar(0);
            if(parmu_ < barmu * Scalar(1.5))
            {
                parmu_ = barmu * Scalar(2);
                if(parmu_raise_changes_optimal())
                    continue;
            }
            prerem_ = parmu_ * prerec_ - sum_;

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
                    return false;
                }
            }
            mirror_best();
            return true;
        }
    }

    [[nodiscard]] bool converged() const { return converged_; }
    [[nodiscard]] solver_status status() const { return status_; }
    [[nodiscard]] Scalar rho() const { return rho_; }
    [[nodiscard]] Scalar parmu() const { return parmu_; }
    [[nodiscard]] Scalar objective_value() const { return f_best_; }
    [[nodiscard]] Scalar resmax() const { return resmax_best_; }
    [[nodiscard]] const vector_type& x() const { return x_best_; }
    [[nodiscard]] Scalar gradient_proxy() const
    {
        Scalar g = Scalar(0);
        for(int i = 1; i <= n_; ++i)
            g += a_(i, mp_) * a_(i, mp_);
        return std::sqrt(g);
    }

private:
    enum class l550_result { continue_iter, converged };
    enum class l440_action { back_to_main, reduce_rho };

    // ---- Deterministic LCG used to jitter geometry (simplex) steps ----
    std::uint32_t lcg_rand()
    {
        return (seed_ = seed_ * std::uint32_t(1103515245) + std::uint32_t(12345));
    }
    Scalar lcg_urand(Scalar a, Scalar b)
    {
        return a + static_cast<Scalar>(lcg_rand())
            * (b - a) / static_cast<Scalar>(static_cast<std::uint32_t>(-1));
    }

    vector_type origin_coords() const
    {
        vector_type x(n_);
        for(int i = 1; i <= n_; ++i)
            x(i - 1) = sim_(i, np_);
        return x;
    }

    // Evaluate f and constraints at x, filling con_ and resmax_/f_last_.
    void evaluate(const vector_type& x)
    {
        vector_type conv(m_);
        Scalar f = eval_(x, conv);
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

    void bootstrap(const vector_type& x0)
    {
        for(int i = 1; i <= n_; ++i)
        {
            sim_(i, np_) = x0(i - 1);
            for(int j = 1; j <= n_; ++j)
            {
                sim_(i, j) = Scalar(0);
                simi_(i, j) = Scalar(0);
            }
            Scalar rhocur = rho_;
            Scalar xi = x0(i - 1);
            if(xi + rhocur > upper_(i - 1))
            {
                if(xi - rhocur >= lower_(i - 1))
                    rhocur = -rhocur;
                else if(upper_(i - 1) - xi > xi - lower_(i - 1))
                    rhocur = Scalar(0.5) * (upper_(i - 1) - xi);
                else
                    rhocur = Scalar(0.5) * (xi - lower_(i - 1));
            }
            sim_(i, i) = rhocur;
            simi_(i, i) = Scalar(1) / rhocur;
        }

        jdrop_ = np_;
        ibrnch_ = 0;
        nevals_ = 0;

        xcur_ = x0;
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

        ibrnch_ = 1;
        mirror_best();
    }

    // Exchange the newly evaluated initial vertex with the optimal vertex if it
    // improves the objective (Powell's initial-simplex completion).
    void exchange_initial_vertex()
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

    // L140: identify the optimal vertex and switch it into pole position.
    void identify_optimal()
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

        if(nbest <= n_)
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
    }

    // Guard against SIMI drifting away from the true inverse of SIM.
    bool check_simi_error()
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
    void build_models()
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
    void compute_vsig_veta()
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

    // Geometry-improving vertex: replace the worst-poised vertex with a point
    // that restores acceptability, evaluate it, and store it in the simplex.
    void geometry_step()
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

        temp = Scalar(0.5) * rho_ * vsig_(jdrop_);
        for(int i = 1; i <= n_; ++i)
            dx_(i - 1) = temp * simi_(jdrop_, i);

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

        temp = Scalar(0);
        for(int i = 1; i <= n_; ++i)
        {
            dx_(i - 1) = dxsign * dx_(i - 1) * lcg_urand(Scalar(0.01), Scalar(1));
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
            xcur_(j - 1) = sim_(j, np_) + dx_(j - 1);
        }

        evaluate(xcur_);
        ++nevals_;
        for(int k = 1; k <= mpp_; ++k)
            datmat_(k, jdrop_) = con_(k);
        ibrnch_ = 1;
    }

    void run_trstlp()
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

    void clamp_dx_to_bounds()
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
    void compute_prediction()
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
    bool parmu_raise_changes_optimal() const
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

    // L440: decide whether the trial replaces a simplex vertex, revise SIM /
    // SIMI / DATMAT, and choose whether to keep iterating or reduce RHO.
    l440_action process_trial()
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

        Scalar edgmax = Scalar(1.1) * rho_;
        int l = 0;
        for(int j = 1; j <= n_; ++j)
        {
            if(sigbar_(j) >= parsig_ || sigbar_(j) >= vsig_(j))
            {
                Scalar temp = veta_(j);
                if(trured > Scalar(0))
                {
                    temp = Scalar(0);
                    for(int i = 1; i <= n_; ++i)
                    {
                        Scalar d = dx_(i - 1) - sim_(i, j);
                        temp += d * d;
                    }
                    temp = std::sqrt(temp);
                }
                if(temp > edgmax)
                {
                    l = j;
                    edgmax = temp;
                }
            }
        }
        if(l > 0)
            jdrop_ = l;
        if(jdrop_ == 0)
            return l440_action::reduce_rho;

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

        if(trured > Scalar(0) && trured >= prerem_ * Scalar(0.1))
        {
            if(trured >= prerem_ * Scalar(0.9) && trured <= prerem_ * Scalar(1.1) && iflag_ != 0)
                rho_ *= Scalar(2);
            return l440_action::back_to_main;
        }
        return l440_action::reduce_rho;
    }

    // L550: rebuild an unacceptable simplex, or reduce RHO and re-derive PARMU;
    // terminate once RHO reaches RHOEND.
    l550_result l550()
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
            {
                Scalar denom = Scalar(0);
                Scalar cmin = Scalar(0);
                Scalar cmax = Scalar(0);
                for(int k = 1; k <= mp_; ++k)
                {
                    cmin = datmat_(k, np_);
                    cmax = cmin;
                    for(int i = 1; i <= n_; ++i)
                    {
                        Scalar v = datmat_(k, i);
                        cmin = std::min(cmin, v);
                        cmax = std::max(cmax, v);
                    }
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
            return l550_result::continue_iter;
        }

        converged_ = true;
        status_ = solver_status::xtol_reached;
        return l550_result::converged;
    }

    // Mirror the current best (pole) vertex into (x_best_, f_best_,
    // resmax_best_). The pole is Powell's incumbent x*: it is promoted to the
    // lowest-merit vertex by identify_optimal at the top of each iteration, so
    // reporting it (rather than a fresh merit re-scan) keeps the reported
    // objective and the merit-penalty adaptation in step with the reference.
    void mirror_best()
    {
        for(int i = 1; i <= n_; ++i)
            x_best_(i - 1) = sim_(i, np_);
        f_best_ = datmat_(mp_, np_);
        resmax_best_ = datmat_(mpp_, np_);
    }

    // Problem dimensions.
    int n_{0};
    int m_{0};
    int mp_{0};
    int mpp_{0};
    int np_{0};

    // Reference working storage (1-indexed access, row/col 0 unused).
    matrix_type sim_;
    matrix_type simi_;
    matrix_type datmat_;
    matrix_type a_;
    vector_type con_;
    vector_type vsig_;
    vector_type veta_;
    vector_type sigbar_;
    vector_type w_;

    // 0-indexed working vectors.
    vector_type dx_;
    vector_type xcur_;
    vector_type lower_;
    vector_type upper_;

    // Scalars.
    Scalar rho_{0};
    Scalar rho_initial_{0};
    Scalar rhoend_{0};
    Scalar parmu_{0};
    Scalar parsig_{0};
    Scalar pareta_{0};
    Scalar resmax_{0};
    Scalar f_last_{0};
    Scalar prerec_{0};
    Scalar prerem_{0};
    Scalar sum_{0};

    // Control flags.
    int ibrnch_{0};
    int iflag_{0};
    int jdrop_{0};
    int ifull_{1};
    int nevals_{0};
    std::uint32_t seed_{0};
    bool converged_{false};
    solver_status status_{solver_status::running};

    // Incumbent mirror.
    vector_type x_best_;
    Scalar f_best_{0};
    Scalar resmax_best_{0};

    evaluator eval_{};
};

}

#endif
