#ifndef HPP_GUARD_ARGMIN_DETAIL_MMA_SUBPROBLEM_H
#define HPP_GUARD_ARGMIN_DETAIL_MMA_SUBPROBLEM_H

// CCSA quadratic-penalty subproblem solver for MMA/GCMMA.
//
// Implements the dual solve for Svanberg 2002 CCSA with the quadratic
// penalty approximation used by NLopt LD_MMA (Steven G. Johnson
// 2008-2012):
//
//   g_0(x) = f + grad_f . dx + 0.5 rho  sum_j dx_j^2 / sigma_j^2
//   g_i(x) = fc_i + dfc_i . dx + 0.5 rhoc_i  sum_j dx_j^2 / sigma_j^2
//
// The primal x(y) is solved analytically per dimension:
//   dx_j = clamp(-sigma_j^2 v_j / u, -sigma_j, sigma_j),
//          then clamp to [lb_j, ub_j]
// where u = rho + sum rhoc_i y_i, v_j = grad_f_j + sum y_i dfc_{ij}.
//
// The dual y >= 0 is solved via projected Newton ascent with warm-start
// across inner and outer iterations.
//
// References:
//   Svanberg 2002, "A class of globally convergent optimization methods
//     based on conservative convex separable approximations", SIAM J.
//     Optim. 12(2):555-573.
//   NLopt ccsa_quadratic.c (Steven G. Johnson 2008-2012), dual_func()
//     and ccsa_quadratic_minimize().

#include "argmin/types.h"
#include "argmin/options/mma_subproblem_options.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>

namespace argmin::detail
{

template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int M = argmin::dynamic_dimension>
class mma_subproblem_solver
{
    using vec_n = Eigen::Vector<Scalar, N>;
    using vec_m = Eigen::Vector<Scalar, M>;
    using mat_mn = Eigen::Matrix<Scalar, M, N>;
    using mat_mm = Eigen::Matrix<Scalar, M, M>;

public:
    explicit mma_subproblem_solver(int n, int m)
        : n_{n}, m_{m}
    {
        y_.resize(m);
        y_.setZero();
        xcur_.resize(n);
        gcval_.resize(m);
        dual_grad_.resize(m);
        dual_hess_.resize(m, m);
        dy_.resize(m);
    }

    mma_subproblem_solver() = default;

    // Solve the CCSA subproblem. Returns the trial point.
    //
    // After return, gval(), wval(), gcval() hold approximation values at
    // the trial point for conservativity testing. multipliers() returns
    // the dual y, warm-started from the previous call.
    //
    // fc, dfc: constraint values and Jacobian in g_i <= 0 form. Caller
    //          negates argmin's c_ineq >= 0 convention before passing.
    //
    // Reference: NLopt ccsa_quadratic.c dual_func().
    vec_n solve(
        const vec_n& x, Scalar f, const vec_n& grad_f,
        const vec_m& fc, const mat_mn& dfc,
        const vec_n& sigma,
        Scalar rho, const vec_m& rhoc,
        const vec_n& lb, const vec_n& ub,
        const mma_subproblem_options& opts = {})
    {
        constexpr Scalar eps = Scalar(1e-10);
        const int max_iter = static_cast<int>(
            opts.dual_max_iterations.value_or(50));
        const auto tol = static_cast<Scalar>(
            opts.dual_tolerance.value_or(1e-9));
        const auto bt = static_cast<Scalar>(
            opts.backtrack_factor.value_or(0.95));

        if(m_ == 0)
        {
            eval_primal(x, f, grad_f, fc, dfc, sigma, rho, rhoc, lb, ub);
            return xcur_;
        }

        for(int iter = 0; iter < max_iter; ++iter)
        {
            eval_primal(x, f, grad_f, fc, dfc, sigma, rho, rhoc, lb, ub);

            // Projected-gradient convergence check. For y_i > 0, any
            // nonzero gradient is a violation. For y_i near zero, only
            // gcval > 0 (= infeasible constraint approximation) is a
            // violation; gcval <= 0 with y = 0 satisfies complementarity.
            //
            // Reference: NLopt ccsa_quadratic.c dual_func() grad
            //            convention; Svanberg 2002 Section 5 KKT.
            Scalar max_proj = Scalar(0);
            for(int i = 0; i < m_; ++i)
            {
                if(y_[i] > eps)
                    max_proj = std::max(max_proj, std::abs(gcval_[i]));
                else
                    max_proj = std::max(max_proj,
                                        std::max(gcval_[i], Scalar(0)));
            }
            if(max_proj < tol)
                break;

            // Analytic Hessian of the Lagrange dual (negative
            // semidefinite for concave W; we build -H which is positive
            // semidefinite for the Newton system (-H) dy = gcval).
            //
            //   -H_{ik} = D_{ik}/u - (rhoc_k A_i + rhoc_i A_k)/u^2
            //             + rhoc_i rhoc_k B / u^3
            //
            // where D_{ik} = sum_j s_j^2 dfc_{ij} dfc_{kj}   (Gram)
            //       A_i    = sum_j s_j^2 v_j dfc_{ij}
            //       B      = sum_j s_j^2 v_j^2
            // summed only over unclamped j (clamped dx_j / dy_k = 0).
            Scalar u = rho;
            for(int i = 0; i < m_; ++i)
                u += rhoc[i] * y_[i];
            u = std::max(u, eps);

            dual_hess_.setZero();

            vec_m A = vec_m::Zero(m_);
            Scalar B = Scalar(0);

            for(int j = 0; j < n_; ++j)
            {
                Scalar v = grad_f[j];
                for(int i = 0; i < m_; ++i)
                    v += y_[i] * dfc(i, j);

                Scalar s2 = sigma[j] * sigma[j];
                Scalar dx_raw = -s2 * v / u;

                // Skip clamped variables (dx/dy = 0 at bounds).
                bool clamped = (std::abs(dx_raw) >= sigma[j])
                    || (x[j] + dx_raw >= ub[j])
                    || (x[j] + dx_raw <= lb[j]);
                if(clamped)
                    continue;

                B += s2 * v * v;
                for(int i = 0; i < m_; ++i)
                    A[i] += s2 * v * dfc(i, j);

                for(int i1 = 0; i1 < m_; ++i1)
                    for(int i2 = i1; i2 < m_; ++i2)
                    {
                        Scalar d = s2 * dfc(i1, j) * dfc(i2, j);
                        dual_hess_(i1, i2) += d;
                        if(i1 != i2)
                            dual_hess_(i2, i1) += d;
                    }
            }

            // Assemble -H = D/u - (rhoc A^T + A rhoc^T)/u^2 + rhoc rhoc^T B/u^3
            Scalar u2 = u * u;
            Scalar u3 = u2 * u;
            for(int i1 = 0; i1 < m_; ++i1)
                for(int i2 = 0; i2 < m_; ++i2)
                {
                    dual_hess_(i1, i2) /= u;
                    dual_hess_(i1, i2) -= (rhoc[i2] * A[i1]
                                            + rhoc[i1] * A[i2]) / u2;
                    dual_hess_(i1, i2) += rhoc[i1] * rhoc[i2] * B / u3;
                }

            dual_hess_.diagonal().array() += eps;

            ldlt_.compute(dual_hess_);
            dy_ = ldlt_.solve(gcval_);

            // Backtrack to keep y >= 0.
            Scalar alpha = Scalar(1);
            for(int i = 0; i < m_; ++i)
            {
                if(dy_[i] < Scalar(0) && y_[i] > Scalar(0))
                {
                    Scalar max_step = y_[i] / (-dy_[i]) * bt;
                    alpha = std::min(alpha, max_step);
                }
            }
            alpha = std::max(alpha, eps);

            y_ += alpha * dy_;
            y_ = y_.cwiseMax(Scalar(0));
        }

        return xcur_;
    }

    Scalar gval() const noexcept { return gval_; }
    Scalar wval() const noexcept { return wval_; }
    const vec_m& gcval() const noexcept { return gcval_; }

    // Subproblem dual multipliers from the last solve(). Valid until the
    // next solve() call. Sign: y_i >= 0 matches argmin's mu_ineq with
    // no flip (g_i = -c_ineq_i, stationarity grad_L = grad_f - sum y_i
    // grad(c_ineq_i) per detail/lagrangian.h).
    const vec_m& multipliers() const noexcept { return y_; }

    void reset_dual() { y_.setZero(); }

private:
    // Evaluate primal x(y) and populate gval_, wval_, gcval_.
    //
    // Reference: NLopt ccsa_quadratic.c dual_func() lines 100-141.
    void eval_primal(
        const vec_n& x, Scalar f, const vec_n& grad_f,
        const vec_m& fc, const mat_mn& dfc,
        const vec_n& sigma,
        Scalar rho, const vec_m& rhoc,
        const vec_n& lb, const vec_n& ub)
    {
        constexpr Scalar eps = Scalar(1e-10);

        Scalar u = rho;
        for(int i = 0; i < m_; ++i)
            u += rhoc[i] * y_[i];
        u = std::max(u, eps);

        wval_ = Scalar(0);
        gval_ = f;
        for(int i = 0; i < m_; ++i)
            gcval_[i] = fc[i];

        for(int j = 0; j < n_; ++j)
        {
            Scalar v = grad_f[j];
            for(int i = 0; i < m_; ++i)
                v += y_[i] * dfc(i, j);

            Scalar s2 = sigma[j] * sigma[j];
            if(s2 < eps) { xcur_[j] = x[j]; continue; }

            Scalar dx = -s2 * v / u;
            if(std::abs(dx) > sigma[j])
                dx = std::copysign(sigma[j], dx);
            xcur_[j] = x[j] + dx;
            if(xcur_[j] > ub[j]) xcur_[j] = ub[j];
            else if(xcur_[j] < lb[j]) xcur_[j] = lb[j];
            dx = xcur_[j] - x[j];

            Scalar dx2sig = Scalar(0.5) * dx * dx / s2;
            wval_ += dx2sig;
            gval_ += grad_f[j] * dx + rho * dx2sig;
            for(int i = 0; i < m_; ++i)
                gcval_[i] += dfc(i, j) * dx + rhoc[i] * dx2sig;
        }
    }

    int n_{0};
    int m_{0};
    vec_m y_;
    vec_n xcur_;
    Scalar gval_{0};
    Scalar wval_{0};
    vec_m gcval_;

    vec_m dual_grad_;
    mat_mm dual_hess_;
    Eigen::LDLT<mat_mm> ldlt_;
    vec_m dy_;
};

}

#endif
