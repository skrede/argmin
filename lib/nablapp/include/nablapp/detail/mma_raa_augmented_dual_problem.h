#ifndef HPP_GUARD_NABLAPP_DETAIL_MMA_RAA_AUGMENTED_DUAL_PROBLEM_H
#define HPP_GUARD_NABLAPP_DETAIL_MMA_RAA_AUGMENTED_DUAL_PROBLEM_H

// Strict-Svanberg-2002 raa-augmented MMA dual problem.
//
// Augments the MMA reciprocal approximation with the asymptote-divergent
// penalty term from Svanberg 2002 §3 (raa-augmentation):
//
//   g_tilde_i(x) = sum_j [ p_ij / (U_j - x_j) + q_ij / (x_j - L_j) ] + r_i
//                  + raa_i * sum_j d_j(x, x_k)
//
//   d_j(x, x_k) = (U_j - L_j) * (x_j - x_kj)^2
//                 / ( (U_j - x_j) * (x_j - L_j) )
//
// Unlike the rho_wval variant (mma_augmented_dual_problem.h) whose
// penalty has an x_k-only denominator and is just a separable
// quadratic, Svanberg 2002's d_j keeps the same asymptotic behavior
// as the underlying reciprocal approximation: d_j -> infinity as x_j
// approaches L_j or U_j. d_j and its gradient both vanish at x = x_k
// so the augmented approximation still matches g_i at x_k in value
// and gradient.
//
// The augmented FOC in component j is:
//   P_j / (U_j - x_j)^2 - Q_j / (x_j - L_j)^2
//     + R(y) * d_j_prime(x_j, x_kj) = 0
//
//   d_j_prime = (U_j - L_j)
//               * [ 2 (x_j - x_kj) / ((U_j - x_j)(x_j - L_j))
//                   - (x_j - x_kj)^2 * (U_j + L_j - 2 x_j)
//                                    / ((U_j - x_j)^2 (x_j - L_j)^2) ]
//
// Solved per-component via Newton iteration with bracket damping.
//
// References:
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573, eq. 3.4-3.6.

#include "nablapp/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

template <typename Scalar, int N, int M>
struct mma_raa_augmented_dual_problem
{
    static constexpr int problem_dimension = M;

    const Eigen::Vector<Scalar, N>* L_out{};
    const Eigen::Vector<Scalar, N>* U_out{};
    const Eigen::Vector<Scalar, N>* x_k_out{};
    const Eigen::Vector<Scalar, N>* alpha_out{};
    const Eigen::Vector<Scalar, N>* beta_out{};
    const Eigen::Vector<Scalar, N>* p_obj_out{};
    const Eigen::Vector<Scalar, N>* q_obj_out{};
    const Eigen::Matrix<Scalar, M, N>* p_con_out{};
    const Eigen::Matrix<Scalar, M, N>* q_con_out{};

    Scalar r_obj{0};
    const Eigen::Vector<Scalar, M>* r_con_out{};

    // Penalty parameters (raa-augmentation; mutate inside conservativity loop).
    Scalar raa_obj{0};
    const Eigen::Vector<Scalar, M>* raa_con_out{};

    int n_primal{0};
    int m_dual{0};

    int newton_max_iter{30};
    Scalar newton_tol{1e-12};

    mutable Eigen::Vector<Scalar, N> x_primal;
    mutable Scalar gval{0};
    mutable Eigen::Vector<Scalar, M> gcval;
    // dval_sum = sum_j d_j(x_primal, x_k); used by raa-growth formulas.
    mutable Scalar dval_sum{0};

    [[nodiscard]] int dimension() const { return m_dual; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, M>& y) const
    {
        eval_primal(y);
        Scalar W = gval;
        for(int i = 0; i < m_dual; ++i)
            W += y[i] * gcval[i];
        return -W;
    }

    void gradient(const Eigen::Vector<Scalar, M>& y,
                  Eigen::Vector<Scalar, M>& g) const
    {
        eval_primal(y);
        for(int i = 0; i < m_dual; ++i)
            g[i] = -gcval[i];
    }

    [[nodiscard]] Eigen::Vector<Scalar, M> lower_bounds() const
    {
        return Eigen::Vector<Scalar, M>::Zero(m_dual);
    }

    [[nodiscard]] Eigen::Vector<Scalar, M> upper_bounds() const
    {
        return Eigen::Vector<Scalar, M>::Constant(
            m_dual, std::numeric_limits<Scalar>::infinity());
    }

private:
    // d_j(x, x_k) and its derivative w.r.t. x (Svanberg 2002 eq. 3.6).
    static Scalar d_value(Scalar x, Scalar x_k, Scalar L, Scalar U)
    {
        const Scalar a = x - x_k;
        const Scalar u = U - x;
        const Scalar l = x - L;
        return (U - L) * a * a / (u * l);
    }

    static Scalar d_deriv(Scalar x, Scalar x_k, Scalar L, Scalar U)
    {
        const Scalar a = x - x_k;
        const Scalar u = U - x;
        const Scalar l = x - L;
        const Scalar denom1 = u * l;
        const Scalar term1 = Scalar(2) * a / denom1;
        const Scalar denom2 = denom1 * denom1;
        const Scalar term2 = a * a * (U + L - Scalar(2) * x) / denom2;
        return (U - L) * (term1 - term2);
    }

    // Per-component Newton on the augmented FOC. The function is monotone
    // increasing in x_j (each piece -- the reciprocal terms and the d_j
    // derivative term in (alpha, beta) -- contributes a non-negative
    // slope), so the root is unique. Initial guess from the un-augmented
    // analytic primal.
    Scalar solve_per_component(Scalar P, Scalar Q, Scalar R,
                               Scalar L, Scalar U, Scalar x_k,
                               Scalar alpha, Scalar beta) const
    {
        const Scalar sP = std::sqrt(std::max(P, Scalar(0)));
        const Scalar sQ = std::sqrt(std::max(Q, Scalar(0)));
        Scalar x = (sP + sQ > Scalar(0))
            ? (sP * L + sQ * U) / (sP + sQ)
            : x_k;
        x = std::clamp(x, alpha, beta);

        if(R == Scalar(0))
            return x;  // un-augmented analytic case

        for(int it = 0; it < newton_max_iter; ++it)
        {
            const Scalar dxU = U - x;
            const Scalar dxL = x - L;
            const Scalar f = P / (dxU * dxU) - Q / (dxL * dxL)
                             + R * d_deriv(x, x_k, L, U);

            // Numerical derivative of f for Newton step. f' is positive
            // throughout (L, U) for a strictly-convex augmented Lagrangian,
            // so a finite-difference approximation is acceptable; use
            // analytical where convenient.
            const Scalar h = std::max(Scalar(1e-8) * std::abs(x),
                                      Scalar(1e-12));
            const Scalar dxU_p = U - (x + h);
            const Scalar dxL_p = (x + h) - L;
            const Scalar f_p = P / (dxU_p * dxU_p) - Q / (dxL_p * dxL_p)
                               + R * d_deriv(x + h, x_k, L, U);
            const Scalar fp = (f_p - f) / h;

            if(std::abs(f) < newton_tol)
                break;
            if(fp <= Scalar(0))
                break;

            Scalar step = f / fp;
            Scalar x_new = x - step;
            for(int damp = 0; damp < 20 && (x_new <= L || x_new >= U); ++damp)
            {
                step *= Scalar(0.5);
                x_new = x - step;
            }
            x = x_new;
            if(std::abs(step) < newton_tol)
                break;
        }

        return std::clamp(x, alpha, beta);
    }

    void eval_primal(const Eigen::Vector<Scalar, M>& y) const
    {
        const auto& L = *L_out;
        const auto& U = *U_out;
        const auto& x_k = *x_k_out;
        const auto& alpha = *alpha_out;
        const auto& beta = *beta_out;
        const auto& p0 = *p_obj_out;
        const auto& q0 = *q_obj_out;
        const auto& pc = *p_con_out;
        const auto& qc = *q_con_out;
        const auto& rc = *r_con_out;
        const auto& raac = *raa_con_out;

        if(x_primal.size() != n_primal)
            x_primal.resize(n_primal);
        if(gcval.size() != m_dual)
            gcval.resize(m_dual);

        Scalar R = raa_obj;
        for(int i = 0; i < m_dual; ++i)
            R += y[i] * raac[i];

        for(int j = 0; j < n_primal; ++j)
        {
            Scalar P = p0[j];
            Scalar Q = q0[j];
            for(int i = 0; i < m_dual; ++i)
            {
                P += y[i] * pc(i, j);
                Q += y[i] * qc(i, j);
            }
            x_primal[j] = solve_per_component(
                P, Q, R, L[j], U[j], x_k[j], alpha[j], beta[j]);
        }

        gval = r_obj;
        for(int i = 0; i < m_dual; ++i)
            gcval[i] = rc[i];
        dval_sum = Scalar(0);

        for(int j = 0; j < n_primal; ++j)
        {
            const Scalar dxU = U[j] - x_primal[j];
            const Scalar dxL = x_primal[j] - L[j];
            const Scalar dj = d_value(x_primal[j], x_k[j], L[j], U[j]);

            gval += p0[j] / dxU + q0[j] / dxL + raa_obj * dj;
            for(int i = 0; i < m_dual; ++i)
                gcval[i] += pc(i, j) / dxU + qc(i, j) / dxL
                          + raac[i] * dj;
            dval_sum += dj;
        }
    }
};

}

#endif
