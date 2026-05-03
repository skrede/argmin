#ifndef HPP_GUARD_ARGMIN_DETAIL_MMA_AUGMENTED_DUAL_PROBLEM_H
#define HPP_GUARD_ARGMIN_DETAIL_MMA_AUGMENTED_DUAL_PROBLEM_H

// Augmented MMA reciprocal-approximation dual problem.
//
// Extends the plain Svanberg 1987 reciprocal approximation with an
// additional separable quadratic penalty term for conservativity-driven
// globalization (Svanberg 2002 / NLopt mma.c style):
//
//   g_tilde_i(x) = sum_j [ p_ij / (U_j - x_j) + q_ij / (x_j - L_j) ] + r_i
//                  + rho_i * 0.5 * sum_j w_j * (x_j - x_kj)^2
//
// where w_j = (U_j - L_j) / ((U_j - x_kj) * (x_kj - L_j)) is the
// asymptote-symmetric weight (NLopt mma.c chooses sigma_j^2 with
// sigma_j = U_j - x_kj = x_kj - L_j; the asymmetric generalization
// here uses the geometric-mean style weight that recovers NLopt's
// expression in the symmetric case).
//
// The augmented per-j Lagrangian is no longer minimized in closed form;
// the per-component primal is found by Newton iteration on the FOC
//   P_j(y) / (U_j - x_j)^2 - Q_j(y) / (x_j - L_j)^2
//     + R(y) * w_j * (x_j - x_kj) = 0
// where
//   P_j(y) = p_0j + sum_i y_i p_ij
//   Q_j(y) = q_0j + sum_i y_i q_ij
//   R(y)   = rho_0 + sum_i y_i rho_i
//
// Used by alternative GCMMA variants that rely on a quadratic-penalty
// conservativity mechanism (rho_wval, raa_augmented). These variants
// differ in how rho_i / raa_i grow on a non-conservative trial; the
// approximation form and primal solver are shared.
//
// References:
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573.
//   NLopt mma.c (Steven G. Johnson 2008-2012).

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

template <typename Scalar, int N, int M>
struct mma_augmented_dual_problem
{
    static constexpr int problem_dimension = M;

    // Outer state pointers.
    const Eigen::Vector<Scalar, N>* L_out{};
    const Eigen::Vector<Scalar, N>* U_out{};
    const Eigen::Vector<Scalar, N>* x_k_out{};
    const Eigen::Vector<Scalar, N>* alpha_out{};
    const Eigen::Vector<Scalar, N>* beta_out{};
    const Eigen::Vector<Scalar, N>* w_out{};        // separable quadratic weights
    const Eigen::Vector<Scalar, N>* p_obj_out{};
    const Eigen::Vector<Scalar, N>* q_obj_out{};
    const Eigen::Matrix<Scalar, M, N>* p_con_out{};
    const Eigen::Matrix<Scalar, M, N>* q_con_out{};

    // Approximation constants.
    Scalar r_obj{0};
    const Eigen::Vector<Scalar, M>* r_con_out{};

    // Penalty parameters: rho_obj for objective augmentation, rho_con
    // per constraint. These mutate inside the conservativity loop.
    Scalar rho_obj{0};
    const Eigen::Vector<Scalar, M>* rho_con_out{};

    int n_primal{0};
    int m_dual{0};

    // Newton inner-solve controls.
    int newton_max_iter{30};
    Scalar newton_tol{1e-12};

    // Cache populated by eval_primal().
    mutable Eigen::Vector<Scalar, N> x_primal;
    mutable Scalar gval{0};
    mutable Eigen::Vector<Scalar, M> gcval;
    // wval = sum_j w_j * (x_primal_j - x_kj)^2; used by rho-growth
    // formulas in the conservativity loop.
    mutable Scalar wval{0};

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
    // Per-component Newton on FOC:
    //   P / (U - x)^2 - Q / (x - L)^2 + R * w * (x - x_k) = 0
    // The function is monotone-increasing in x (P/(U-x)^2 increasing,
    // -Q/(x-L)^2 also increasing in x, R*w*(x-x_k) increasing), so the
    // root is unique and Newton converges quadratically when started
    // bracketed. Initial guess is the analytic primal of the
    // un-augmented problem (R = 0 limit).
    Scalar solve_per_component(Scalar P, Scalar Q, Scalar R, Scalar w,
                               Scalar L, Scalar U, Scalar x_k,
                               Scalar alpha, Scalar beta) const
    {
        const Scalar sP = std::sqrt(std::max(P, Scalar(0)));
        const Scalar sQ = std::sqrt(std::max(Q, Scalar(0)));
        Scalar x = (sP + sQ > Scalar(0))
            ? (sP * L + sQ * U) / (sP + sQ)
            : x_k;
        x = std::clamp(x, alpha, beta);

        if(R * w == Scalar(0))
            return x;  // un-augmented analytic case

        for(int it = 0; it < newton_max_iter; ++it)
        {
            const Scalar dxU = U - x;
            const Scalar dxL = x - L;
            const Scalar f = P / (dxU * dxU) - Q / (dxL * dxL)
                             + R * w * (x - x_k);
            const Scalar fp = Scalar(2) * P / (dxU * dxU * dxU)
                            + Scalar(2) * Q / (dxL * dxL * dxL)
                            + R * w;

            if(std::abs(f) < newton_tol)
                break;
            if(fp <= Scalar(0))
                break;  // degenerate; bail out

            Scalar step = f / fp;
            // Damping: keep x strictly in (L, U) to avoid asymptote
            // singularities. If a full Newton step would cross L or U,
            // halve until inside.
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
        const auto& w = *w_out;
        const auto& p0 = *p_obj_out;
        const auto& q0 = *q_obj_out;
        const auto& pc = *p_con_out;
        const auto& qc = *q_con_out;
        const auto& rc = *r_con_out;
        const auto& rhoc = *rho_con_out;

        if(x_primal.size() != n_primal)
            x_primal.resize(n_primal);
        if(gcval.size() != m_dual)
            gcval.resize(m_dual);

        Scalar R = rho_obj;
        for(int i = 0; i < m_dual; ++i)
            R += y[i] * rhoc[i];

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
                P, Q, R, w[j], L[j], U[j], x_k[j], alpha[j], beta[j]);
        }

        gval = r_obj;
        for(int i = 0; i < m_dual; ++i)
            gcval[i] = rc[i];
        wval = Scalar(0);

        for(int j = 0; j < n_primal; ++j)
        {
            const Scalar dxU = U[j] - x_primal[j];
            const Scalar dxL = x_primal[j] - L[j];
            const Scalar dx = x_primal[j] - x_k[j];
            const Scalar quad_j = Scalar(0.5) * w[j] * dx * dx;

            gval += p0[j] / dxU + q0[j] / dxL + rho_obj * quad_j;
            for(int i = 0; i < m_dual; ++i)
                gcval[i] += pc(i, j) / dxU + qc(i, j) / dxL
                          + rhoc[i] * quad_j;
            wval += Scalar(2) * quad_j;  // sum_j w_j * dx_j^2
        }
    }
};

}

#endif
