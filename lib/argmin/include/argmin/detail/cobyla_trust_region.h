#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRUST_REGION_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_TRUST_REGION_H

// Two-stage linear trust-region subproblem solver (TRSTLP) for COBYLA.
//
// Given the linear approximations of the objective and constraints at the
// current best simplex vertex, TRSTLP computes a displacement d that
//
//   Stage 1: minimizes the greatest linearized constraint violation
//            max_k ( b_k - a_k . d )   subject to  ||d|| <= rho,
//
//   Stage 2: if the stage-1 optimum leaves ||d|| strictly below rho, uses the
//            remaining freedom to minimize the linearized objective
//            -a_{m+1} . d  without increasing any greatest violation.
//
// The two stages are distinguished by whether the objective gradient is
// treated as an extra active constraint (Powell's device: the gradient of the
// objective is appended to the constraint gradients in column m+1 of A). The
// method maintains an active set with an orthogonal factorization Z of the
// active-constraint gradients (Gram-Schmidt via Givens rotations), so the
// linear algebra is Householder/QR-equivalent and self-contained -- no shared
// LP or QP kernel is required.
//
// This is a faithful port of M. J. D. Powell's TRSTLP (COBYLA2), matching the
// f2c reference behavior. Constraint convention: constraint k is satisfied
// when a_k . d >= b_k; the objective direction column m+1 holds minus the
// objective gradient (so maximizing a_{m+1} . d minimizes the objective).
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization method
//            that models the objective and constraint functions by linear
//            interpolation." Section 3 (the trust-region subproblem).

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <vector>

namespace argmin::detail
{

// Solve the COBYLA linear trust-region subproblem.
//
// a  : n x (m+1) matrix. Column k in [0,m) is the gradient of constraint k;
//      column m holds minus the objective gradient (the direction along which
//      the objective decreases). Access a(i,k), i in [0,n), k in [0,m].
// b  : length-m constraint right-hand sides; constraint k feasible at d when
//      a.col(k).dot(d) >= b(k). The residual at d=0 is b(k), so a positive
//      b(k) marks a violated constraint.
// rho: trust-region radius (Euclidean).
// dx : output displacement (length n).
// ifull: set to 1 if the returned dx reaches the full stage-1/stage-2 optimum
//        at the trust-region boundary, 0 if a degeneracy shortened it.
//
// Returns true on success, false if a rounding-limited breakdown occurred.
template <typename Scalar = double>
bool cobyla_trstlp(int n, int m,
                   const Eigen::Matrix<Scalar, argmin::dynamic_dimension, argmin::dynamic_dimension>& a,
                   const Eigen::Vector<Scalar, argmin::dynamic_dimension>& b,
                   Scalar rho,
                   Eigen::Vector<Scalar, argmin::dynamic_dimension>& dx,
                   int& ifull)
{
    // 1-indexed working storage (index 0 unused) so the transcription follows
    // Powell's reference line for line.
    const int mp = m + 1;

    // z is an n x n orthogonal matrix (columns 1..n).
    Eigen::Matrix<Scalar, argmin::dynamic_dimension, argmin::dynamic_dimension> z(n + 1, n + 1);
    z.setZero();
    std::vector<Scalar> zdota(static_cast<std::size_t>(n) + 1, Scalar(0));
    std::vector<Scalar> vmultc(static_cast<std::size_t>(m) + 2, Scalar(0));
    std::vector<Scalar> vmultd(static_cast<std::size_t>(m) + 2, Scalar(0));
    std::vector<Scalar> sdirn(static_cast<std::size_t>(n) + 1, Scalar(0));
    std::vector<Scalar> dxnew(static_cast<std::size_t>(n) + 1, Scalar(0));
    std::vector<int> iact(static_cast<std::size_t>(m) + 2, 0);

    // 1-indexed accessors into the caller's 0-indexed a / b.
    auto A = [&](int i, int k) -> Scalar { return a(i - 1, k - 1); };
    auto B = [&](int k) -> Scalar { return b(k - 1); };

    dx.setZero(n);

    ifull = 1;
    int mcon = m;
    int nact = 0;
    Scalar resmax = Scalar(0);
    int icon = 0;

    for(int i = 1; i <= n; ++i)
    {
        for(int j = 1; j <= n; ++j)
            z(i, j) = Scalar(0);
        z(i, i) = Scalar(1);
        dx(i - 1) = Scalar(0);
    }

    if(m >= 1)
    {
        for(int k = 1; k <= m; ++k)
        {
            if(B(k) > resmax)
            {
                resmax = B(k);
                icon = k;
            }
        }
        for(int k = 1; k <= m; ++k)
        {
            iact[static_cast<std::size_t>(k)] = k;
            vmultc[static_cast<std::size_t>(k)] = resmax - B(k);
        }
    }

    Scalar optold = Scalar(0);
    int nactx = 0;
    int icount = 0;
    Scalar step = Scalar(0);
    Scalar stpful = Scalar(0);
    Scalar resold = Scalar(0);

    if(resmax == Scalar(0))
        goto L480;

    for(int i = 1; i <= n; ++i)
        sdirn[static_cast<std::size_t>(i)] = Scalar(0);

L60:
    optold = Scalar(0);
    icount = 0;

L70:
    {
        Scalar optnew;
        if(mcon == m)
            optnew = resmax;
        else
        {
            optnew = Scalar(0);
            for(int i = 1; i <= n; ++i)
                optnew -= dx(i - 1) * A(i, mcon);
        }

        if(icount == 0 || optnew < optold)
        {
            optold = optnew;
            nactx = nact;
            icount = 3;
        }
        else if(nact > nactx)
        {
            nactx = nact;
            icount = 3;
        }
        else
        {
            --icount;
            if(icount == 0)
                goto L490;
        }
    }

    // If ICON exceeds NACT, add the constraint IACT(ICON) to the active set.
    if(icon <= nact)
        goto L260;
    {
        int kk = iact[static_cast<std::size_t>(icon)];
        for(int i = 1; i <= n; ++i)
            dxnew[static_cast<std::size_t>(i)] = A(i, kk);
        Scalar tot = Scalar(0);
        int k = n;
        while(k > nact)
        {
            Scalar sp = Scalar(0);
            Scalar spabs = Scalar(0);
            for(int i = 1; i <= n; ++i)
            {
                Scalar temp = z(i, k) * dxnew[static_cast<std::size_t>(i)];
                sp += temp;
                spabs += std::abs(temp);
            }
            Scalar acca = spabs + std::abs(sp) * Scalar(0.1);
            Scalar accb = spabs + std::abs(sp) * Scalar(0.2);
            if(spabs >= acca || acca >= accb)
                sp = Scalar(0);
            if(tot == Scalar(0))
                tot = sp;
            else
            {
                int kp = k + 1;
                Scalar temp = std::sqrt(sp * sp + tot * tot);
                Scalar alpha = sp / temp;
                Scalar beta = tot / temp;
                tot = temp;
                for(int i = 1; i <= n; ++i)
                {
                    Scalar t = alpha * z(i, k) + beta * z(i, kp);
                    z(i, kp) = alpha * z(i, kp) - beta * z(i, k);
                    z(i, k) = t;
                }
            }
            --k;
        }

        // Add the new constraint if this can be done without a deletion.
        if(tot != Scalar(0))
        {
            ++nact;
            zdota[static_cast<std::size_t>(nact)] = tot;
            vmultc[static_cast<std::size_t>(icon)] = vmultc[static_cast<std::size_t>(nact)];
            vmultc[static_cast<std::size_t>(nact)] = Scalar(0);
            goto L210_add;
        }

        // A deletion is needed to make room. Set VMULTD to the multipliers of
        // the linear combination expressing the new gradient.
        {
            Scalar ratio = Scalar(-1);
            k = nact;
            while(k > 0)
            {
                Scalar zdotv = Scalar(0);
                Scalar zdvabs = Scalar(0);
                for(int i = 1; i <= n; ++i)
                {
                    Scalar temp = z(i, k) * dxnew[static_cast<std::size_t>(i)];
                    zdotv += temp;
                    zdvabs += std::abs(temp);
                }
                Scalar acca = zdvabs + std::abs(zdotv) * Scalar(0.1);
                Scalar accb = zdvabs + std::abs(zdotv) * Scalar(0.2);
                if(zdvabs < acca && acca < accb)
                {
                    Scalar temp = zdotv / zdota[static_cast<std::size_t>(k)];
                    if(temp > Scalar(0) && iact[static_cast<std::size_t>(k)] <= m)
                    {
                        Scalar tempa = vmultc[static_cast<std::size_t>(k)] / temp;
                        if(ratio < Scalar(0) || tempa < ratio)
                            ratio = tempa;
                    }
                    if(k >= 2)
                    {
                        int kw = iact[static_cast<std::size_t>(k)];
                        for(int i = 1; i <= n; ++i)
                            dxnew[static_cast<std::size_t>(i)] -= temp * A(i, kw);
                    }
                    vmultd[static_cast<std::size_t>(k)] = temp;
                }
                else
                    vmultd[static_cast<std::size_t>(k)] = Scalar(0);
                --k;
            }
            if(ratio < Scalar(0))
                goto L490;

            // Revise the multipliers and reorder the active constraints so the
            // one to be replaced is at the end of the list.
            for(int kk2 = 1; kk2 <= nact; ++kk2)
            {
                Scalar v = vmultc[static_cast<std::size_t>(kk2)]
                    - ratio * vmultd[static_cast<std::size_t>(kk2)];
                vmultc[static_cast<std::size_t>(kk2)] = std::max(Scalar(0), v);
            }
            if(icon < nact)
            {
                int isave = iact[static_cast<std::size_t>(icon)];
                Scalar vsave = vmultc[static_cast<std::size_t>(icon)];
                k = icon;
                do
                {
                    int kp = k + 1;
                    int kw = iact[static_cast<std::size_t>(kp)];
                    Scalar sp = Scalar(0);
                    for(int i = 1; i <= n; ++i)
                        sp += z(i, k) * A(i, kw);
                    Scalar temp = std::sqrt(sp * sp
                        + zdota[static_cast<std::size_t>(kp)] * zdota[static_cast<std::size_t>(kp)]);
                    Scalar alpha = zdota[static_cast<std::size_t>(kp)] / temp;
                    Scalar beta = sp / temp;
                    zdota[static_cast<std::size_t>(kp)] = alpha * zdota[static_cast<std::size_t>(k)];
                    zdota[static_cast<std::size_t>(k)] = temp;
                    for(int i = 1; i <= n; ++i)
                    {
                        Scalar t = alpha * z(i, kp) + beta * z(i, k);
                        z(i, kp) = alpha * z(i, k) - beta * z(i, kp);
                        z(i, k) = t;
                    }
                    iact[static_cast<std::size_t>(k)] = kw;
                    vmultc[static_cast<std::size_t>(k)] = vmultc[static_cast<std::size_t>(kp)];
                    k = kp;
                } while(k < nact);
                iact[static_cast<std::size_t>(k)] = isave;
                vmultc[static_cast<std::size_t>(k)] = vsave;
            }
            {
                Scalar temp = Scalar(0);
                for(int i = 1; i <= n; ++i)
                    temp += z(i, nact) * A(i, kk);
                if(temp == Scalar(0))
                    goto L490;
                zdota[static_cast<std::size_t>(nact)] = temp;
            }
            vmultc[static_cast<std::size_t>(icon)] = Scalar(0);
            vmultc[static_cast<std::size_t>(nact)] = ratio;
        }

    L210_add:
        // Update IACT; keep the objective as the last active constraint when
        // MCON>M.
        iact[static_cast<std::size_t>(icon)] = iact[static_cast<std::size_t>(nact)];
        iact[static_cast<std::size_t>(nact)] = kk;
        if(mcon > m && kk != mcon)
        {
            int k2 = nact - 1;
            Scalar sp = Scalar(0);
            for(int i = 1; i <= n; ++i)
                sp += z(i, k2) * A(i, kk);
            Scalar temp = std::sqrt(sp * sp
                + zdota[static_cast<std::size_t>(nact)] * zdota[static_cast<std::size_t>(nact)]);
            Scalar alpha = zdota[static_cast<std::size_t>(nact)] / temp;
            Scalar beta = sp / temp;
            zdota[static_cast<std::size_t>(nact)] = alpha * zdota[static_cast<std::size_t>(k2)];
            zdota[static_cast<std::size_t>(k2)] = temp;
            for(int i = 1; i <= n; ++i)
            {
                Scalar t = alpha * z(i, nact) + beta * z(i, k2);
                z(i, nact) = alpha * z(i, k2) - beta * z(i, nact);
                z(i, k2) = t;
            }
            iact[static_cast<std::size_t>(nact)] = iact[static_cast<std::size_t>(k2)];
            iact[static_cast<std::size_t>(k2)] = kk;
            std::swap(vmultc[static_cast<std::size_t>(k2)], vmultc[static_cast<std::size_t>(nact)]);
        }

        // Set SDIRN for stage one.
        if(mcon > m)
            goto L320;
        {
            int kk3 = iact[static_cast<std::size_t>(nact)];
            Scalar temp = Scalar(0);
            for(int i = 1; i <= n; ++i)
                temp += sdirn[static_cast<std::size_t>(i)] * A(i, kk3);
            temp += Scalar(-1);
            temp /= zdota[static_cast<std::size_t>(nact)];
            for(int i = 1; i <= n; ++i)
                sdirn[static_cast<std::size_t>(i)] -= temp * z(i, nact);
        }
        goto L340;
    }

L260:
    // Delete the constraint IACT(ICON) from the active set.
    if(icon < nact)
    {
        int isave = iact[static_cast<std::size_t>(icon)];
        Scalar vsave = vmultc[static_cast<std::size_t>(icon)];
        int k = icon;
        do
        {
            int kp = k + 1;
            int kk = iact[static_cast<std::size_t>(kp)];
            Scalar sp = Scalar(0);
            for(int i = 1; i <= n; ++i)
                sp += z(i, k) * A(i, kk);
            Scalar temp = std::sqrt(sp * sp
                + zdota[static_cast<std::size_t>(kp)] * zdota[static_cast<std::size_t>(kp)]);
            Scalar alpha = zdota[static_cast<std::size_t>(kp)] / temp;
            Scalar beta = sp / temp;
            zdota[static_cast<std::size_t>(kp)] = alpha * zdota[static_cast<std::size_t>(k)];
            zdota[static_cast<std::size_t>(k)] = temp;
            for(int i = 1; i <= n; ++i)
            {
                Scalar t = alpha * z(i, kp) + beta * z(i, k);
                z(i, kp) = alpha * z(i, k) - beta * z(i, kp);
                z(i, k) = t;
            }
            iact[static_cast<std::size_t>(k)] = kk;
            vmultc[static_cast<std::size_t>(k)] = vmultc[static_cast<std::size_t>(kp)];
            k = kp;
        } while(k < nact);
        iact[static_cast<std::size_t>(k)] = isave;
        vmultc[static_cast<std::size_t>(k)] = vsave;
    }
    --nact;

    if(mcon > m)
        goto L320;
    {
        Scalar temp = Scalar(0);
        for(int i = 1; i <= n; ++i)
            temp += sdirn[static_cast<std::size_t>(i)] * z(i, nact + 1);
        for(int i = 1; i <= n; ++i)
            sdirn[static_cast<std::size_t>(i)] -= temp * z(i, nact + 1);
    }
    goto L340;

L320:
    {
        Scalar temp = Scalar(1) / zdota[static_cast<std::size_t>(nact)];
        for(int i = 1; i <= n; ++i)
            sdirn[static_cast<std::size_t>(i)] = temp * z(i, nact);
    }

L340:
    {
        Scalar dd = rho * rho;
        Scalar sd = Scalar(0);
        Scalar ss = Scalar(0);
        for(int i = 1; i <= n; ++i)
        {
            if(std::abs(dx(i - 1)) >= rho * Scalar(1e-6))
                dd -= dx(i - 1) * dx(i - 1);
            sd += dx(i - 1) * sdirn[static_cast<std::size_t>(i)];
            ss += sdirn[static_cast<std::size_t>(i)] * sdirn[static_cast<std::size_t>(i)];
        }
        if(dd <= Scalar(0))
            goto L490;
        Scalar temp = std::sqrt(ss * dd);
        if(std::abs(sd) >= temp * Scalar(1e-6))
            temp = std::sqrt(ss * dd + sd * sd);
        stpful = dd / (temp + sd);
        step = stpful;
        if(mcon == m)
        {
            Scalar acca = step + resmax * Scalar(0.1);
            Scalar accb = step + resmax * Scalar(0.2);
            if(step >= acca || acca >= accb)
                goto L480;
            step = std::min(step, resmax);
        }
        if(!std::isfinite(step))
            return false;

        // DXNEW = new variables at this steplength; reduce RESMAX in stage one.
        for(int i = 1; i <= n; ++i)
            dxnew[static_cast<std::size_t>(i)] = dx(i - 1) + step * sdirn[static_cast<std::size_t>(i)];
        if(mcon == m)
        {
            resold = resmax;
            resmax = Scalar(0);
            for(int k = 1; k <= nact; ++k)
            {
                int kk = iact[static_cast<std::size_t>(k)];
                Scalar t = B(kk);
                for(int i = 1; i <= n; ++i)
                    t -= A(i, kk) * dxnew[static_cast<std::size_t>(i)];
                resmax = std::max(resmax, t);
            }
        }

        // New Lagrange multipliers VMULTD.
        int k = nact;
        while(k >= 1)
        {
            Scalar zdotw = Scalar(0);
            Scalar zdwabs = Scalar(0);
            for(int i = 1; i <= n; ++i)
            {
                Scalar t = z(i, k) * dxnew[static_cast<std::size_t>(i)];
                zdotw += t;
                zdwabs += std::abs(t);
            }
            Scalar acca = zdwabs + std::abs(zdotw) * Scalar(0.1);
            Scalar accb = zdwabs + std::abs(zdotw) * Scalar(0.2);
            if(zdwabs >= acca || acca >= accb)
                zdotw = Scalar(0);
            vmultd[static_cast<std::size_t>(k)] = zdotw / zdota[static_cast<std::size_t>(k)];
            if(k >= 2)
            {
                int kk = iact[static_cast<std::size_t>(k)];
                for(int i = 1; i <= n; ++i)
                    dxnew[static_cast<std::size_t>(i)] -= vmultd[static_cast<std::size_t>(k)] * A(i, kk);
                --k;
                continue;
            }
            break;
        }
        if(mcon > m)
            vmultd[static_cast<std::size_t>(nact)] = std::max(Scalar(0), vmultd[static_cast<std::size_t>(nact)]);

        // Complete VMULTD with the residuals of the remaining constraints.
        for(int i = 1; i <= n; ++i)
            dxnew[static_cast<std::size_t>(i)] = dx(i - 1) + step * sdirn[static_cast<std::size_t>(i)];
        if(mcon > nact)
        {
            for(int kk2 = nact + 1; kk2 <= mcon; ++kk2)
            {
                int kk = iact[static_cast<std::size_t>(kk2)];
                Scalar sum = resmax - B(kk);
                Scalar sumabs = resmax + std::abs(B(kk));
                for(int i = 1; i <= n; ++i)
                {
                    Scalar t = A(i, kk) * dxnew[static_cast<std::size_t>(i)];
                    sum += t;
                    sumabs += std::abs(t);
                }
                Scalar acca = sumabs + std::abs(sum) * Scalar(0.1);
                Scalar accb = sumabs + std::abs(sum) * Scalar(0.2);
                if(sumabs >= acca || acca >= accb)
                    sum = Scalar(0);
                vmultd[static_cast<std::size_t>(kk2)] = sum;
            }
        }

        // Fraction of the step from DX to DXNEW that will be taken.
        Scalar ratio = Scalar(1);
        icon = 0;
        for(int kk2 = 1; kk2 <= mcon; ++kk2)
        {
            if(vmultd[static_cast<std::size_t>(kk2)] < Scalar(0))
            {
                Scalar t = vmultc[static_cast<std::size_t>(kk2)]
                    / (vmultc[static_cast<std::size_t>(kk2)] - vmultd[static_cast<std::size_t>(kk2)]);
                if(t < ratio)
                {
                    ratio = t;
                    icon = kk2;
                }
            }
        }

        // Update DX, VMULTC and RESMAX.
        Scalar temp2 = Scalar(1) - ratio;
        for(int i = 1; i <= n; ++i)
            dx(i - 1) = temp2 * dx(i - 1) + ratio * dxnew[static_cast<std::size_t>(i)];
        for(int kk2 = 1; kk2 <= mcon; ++kk2)
        {
            Scalar v = temp2 * vmultc[static_cast<std::size_t>(kk2)]
                + ratio * vmultd[static_cast<std::size_t>(kk2)];
            vmultc[static_cast<std::size_t>(kk2)] = std::max(Scalar(0), v);
        }
        if(mcon == m)
            resmax = resold + ratio * (resmax - resold);

        if(icon > 0)
            goto L70;
        if(step == stpful)
            goto L500;
    }

L480:
    mcon = m + 1;
    icon = mcon;
    iact[static_cast<std::size_t>(mcon)] = mcon;
    vmultc[static_cast<std::size_t>(mcon)] = Scalar(0);
    goto L60;

L490:
    if(mcon == m)
        goto L480;
    ifull = 0;

L500:
    return true;
}

}

#endif
