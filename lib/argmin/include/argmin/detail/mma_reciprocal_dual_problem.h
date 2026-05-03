#ifndef HPP_GUARD_ARGMIN_DETAIL_MMA_RECIPROCAL_DUAL_PROBLEM_H
#define HPP_GUARD_ARGMIN_DETAIL_MMA_RECIPROCAL_DUAL_PROBLEM_H

// MMA reciprocal-approximation dual problem (Svanberg 1987).
//
// Wraps the m-dimensional Lagrange dual of the MMA reciprocal subproblem
// as a argmin problem type satisfying differentiable + bound_constrained.
// Any argmin box-constrained policy (lbfgsb, byrd_lbfgsb, bobyqa) can
// solve it.
//
// Subproblem at outer iterate x_k with asymptotes L_k, U_k:
//   min  f_tilde(x) s.t. g_tilde_i(x) <= 0,  alpha <= x <= beta
// where
//   f_tilde(x) = sum_j [ p_0j / (U_j - x_j) + q_0j / (x_j - L_j) ] + const
//   g_tilde_i(x) = sum_j [ p_ij / (U_j - x_j) + q_ij / (x_j - L_j) ] + const
//   p_ij = max(d g_i / d x_j, 0) * (U_j - x_kj)^2 + epsilon
//   q_ij = max(-d g_i / d x_j, 0) * (x_kj - L_j)^2 + epsilon
//
// Lagrangian L(x, y) = f_tilde(x) + sum_i y_i g_tilde_i(x). Each j-th
// component is independent, and the per-j minimizer (over alpha_j <= x_j
// <= beta_j) has the closed form
//   x_j(y) = (sqrt(P_j) L_j + sqrt(Q_j) U_j) / (sqrt(P_j) + sqrt(Q_j))
// clamped to [alpha_j, beta_j], where
//   P_j(y) = p_0j + sum_{i=1..m} y_i p_ij
//   Q_j(y) = q_0j + sum_{i=1..m} y_i q_ij.
//
// The dual is W(y) = min_x L(x, y), concave in y. We expose -W(y) as
// the value() (since argmin policies minimize). By the envelope theorem,
// at interior x or at a bound (where d x / d y = 0), partial of W w.r.t.
// y_i equals g_tilde_i(x(y)).
//
// The reciprocal approximation has a constant term r_i defined by
//   r_i = g_i(x_k) - sum_j [ p_ij / (U_j - x_kj) + q_ij / (x_kj - L_j) ]
// chosen so g_tilde_i(x_k) = g_i(x_k). Constants matter here: r_i
// multiplies y_i in W(y), so its gradient contribution is constant but
// nonzero. Omitting r_i shifts the dual gradient by a y-independent
// offset and pushes y* to the wrong place. r_obj (= r_0) is a true
// constant in y and could be dropped without affecting y*, but we
// keep it so g_tilde values are exact when reported back to the
// outer loop.
//
// References:
//   Svanberg 1987, "The method of moving asymptotes",
//     Int. J. Numer. Methods Engng 24:359-373, eq. 6-7 and Section 4.

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

template <typename Scalar, int N, int M>
struct mma_reciprocal_dual_problem
{
    static constexpr int problem_dimension = M;

    // Outer state pointers (valid for the lifetime of one step()).
    const Eigen::Vector<Scalar, N>* L_out{};
    const Eigen::Vector<Scalar, N>* U_out{};
    const Eigen::Vector<Scalar, N>* alpha_out{};
    const Eigen::Vector<Scalar, N>* beta_out{};
    const Eigen::Vector<Scalar, N>* p_obj_out{};
    const Eigen::Vector<Scalar, N>* q_obj_out{};
    const Eigen::Matrix<Scalar, M, N>* p_con_out{};
    const Eigen::Matrix<Scalar, M, N>* q_con_out{};

    // Approximation constants (see header). r_obj for the objective,
    // r_con per constraint. Computed once per outer iter by the policy.
    Scalar r_obj{0};
    const Eigen::Vector<Scalar, M>* r_con_out{};

    int n_primal{0};
    int m_dual{0};

    // Cache populated by eval_primal(): trial x, full f_tilde value
    // (variable + r_obj), full g_tilde values per constraint
    // (variable + r_con).
    mutable Eigen::Vector<Scalar, N> x_primal;
    mutable Scalar gval{0};
    mutable Eigen::Vector<Scalar, M> gcval;

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
    // Compute primal x(y) per component and populate v_obj, v_con.
    void eval_primal(const Eigen::Vector<Scalar, M>& y) const
    {
        const auto& L = *L_out;
        const auto& U = *U_out;
        const auto& alpha = *alpha_out;
        const auto& beta = *beta_out;
        const auto& p0 = *p_obj_out;
        const auto& q0 = *q_obj_out;
        const auto& pc = *p_con_out;
        const auto& qc = *q_con_out;

        const auto& rc = *r_con_out;

        if(x_primal.size() != n_primal)
            x_primal.resize(n_primal);
        if(gcval.size() != m_dual)
            gcval.resize(m_dual);

        for(int j = 0; j < n_primal; ++j)
        {
            Scalar P = p0[j];
            Scalar Q = q0[j];
            for(int i = 0; i < m_dual; ++i)
            {
                P += y[i] * pc(i, j);
                Q += y[i] * qc(i, j);
            }

            const Scalar sP = std::sqrt(std::max(P, Scalar(0)));
            const Scalar sQ = std::sqrt(std::max(Q, Scalar(0)));
            const Scalar denom = sP + sQ;
            Scalar xj = (denom > Scalar(0))
                ? (sP * L[j] + sQ * U[j]) / denom
                : Scalar(0.5) * (alpha[j] + beta[j]);

            x_primal[j] = std::clamp(xj, alpha[j], beta[j]);
        }

        gval = r_obj;
        for(int i = 0; i < m_dual; ++i)
            gcval[i] = rc[i];

        for(int j = 0; j < n_primal; ++j)
        {
            const Scalar dxU = U[j] - x_primal[j];
            const Scalar dxL = x_primal[j] - L[j];
            gval += p0[j] / dxU + q0[j] / dxL;
            for(int i = 0; i < m_dual; ++i)
                gcval[i] += pc(i, j) / dxU + qc(i, j) / dxL;
        }
    }
};

}

#endif
