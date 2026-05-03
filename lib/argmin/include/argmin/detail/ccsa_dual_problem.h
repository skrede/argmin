#ifndef HPP_GUARD_ARGMIN_DETAIL_CCSA_DUAL_PROBLEM_H
#define HPP_GUARD_ARGMIN_DETAIL_CCSA_DUAL_PROBLEM_H

// CCSA dual problem wrapper for Svanberg 2002 / NLopt LD_MMA.
//
// Wraps the m-dimensional Lagrange dual of the CCSA quadratic-penalty
// subproblem as a argmin problem type satisfying differentiable and
// bound_constrained. Any argmin box-constrained policy (lbfgsb,
// byrd_lbfgsb, bobyqa, ...) can solve it.
//
// The primal x(y) is computed analytically per NLopt ccsa_quadratic.c
// dual_func(): for each j, dx_j = clamp(-sigma_j^2 v_j / u, -sigma_j,
// sigma_j), clamp to [lb_j, ub_j], where u = rho + sum rhoc_i y_i and
// v_j = grad_f_j + sum y_i dfc_{ij}. Side products gval, wval, gcval
// are cached for the outer conservativity test.
//
// Template parameters:
//   N — outer problem dimension (compile-time or dynamic_dimension)
//   M — outer constraint count = dual problem dimension
//
// References:
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573, Section 4.2.
//   NLopt ccsa_quadratic.c dual_func() (Steven G. Johnson 2008-2012).

#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace argmin::detail
{

template <typename Scalar, int N, int M>
struct ccsa_dual_problem
{
    static constexpr int problem_dimension = M;

    // Outer problem data (pointers into the ccsa_quadratic_policy state; valid
    // for the lifetime of one step() call).
    const Eigen::Vector<Scalar, N>* x_out{};
    Scalar f_out{};
    const Eigen::Vector<Scalar, N>* grad_f_out{};
    const Eigen::Vector<Scalar, M>* fc_out{};
    const Eigen::Matrix<Scalar, M, N>* dfc_out{};
    const Eigen::Vector<Scalar, N>* sigma_out{};
    Scalar rho_out{};
    const Eigen::Vector<Scalar, M>* rhoc_out{};
    const Eigen::Vector<Scalar, N>* lb_out{};
    const Eigen::Vector<Scalar, N>* ub_out{};
    int n_primal{0};
    int m_dual{0};

    // Cache: populated by eval_primal(), read by the outer loop after
    // the dual solve completes.
    mutable Eigen::Vector<Scalar, N> x_primal;
    mutable Scalar gval{0};
    mutable Scalar wval{0};
    mutable Eigen::Vector<Scalar, M> gcval;

    [[nodiscard]] int dimension() const { return m_dual; }

    // Minimize -W(y) where W(y) = gval + sum y_i gcval_i.
    // -W is convex (W is concave for the Lagrange dual).
    [[nodiscard]] Scalar value(
        const Eigen::Vector<Scalar, M>& y) const
    {
        eval_primal(y);
        Scalar W = gval;
        for(int i = 0; i < m_dual; ++i)
            W += y[i] * gcval[i];
        return -W;
    }

    // Gradient of -W w.r.t. y.  grad_i = -gcval_i by the envelope
    // theorem (x(y) is the primal minimizer for fixed y).
    void gradient(
        const Eigen::Vector<Scalar, M>& y,
        Eigen::Vector<Scalar, M>& g) const
    {
        eval_primal(y);
        g = -gcval;
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
    // Compute primal x(y) and populate gval, wval, gcval.
    // Mirror of NLopt ccsa_quadratic.c dual_func() lines 100-141.
    void eval_primal(const Eigen::Vector<Scalar, M>& y) const
    {
        constexpr Scalar eps = Scalar(1e-20);

        Scalar u = rho_out;
        for(int i = 0; i < m_dual; ++i)
            u += (*rhoc_out)[i] * y[i];
        u = std::max(u, eps);

        const auto& x = *x_out;
        const auto& gf = *grad_f_out;
        const auto& fc = *fc_out;
        const auto& dfc = *dfc_out;
        const auto& sig = *sigma_out;
        const auto& lb = *lb_out;
        const auto& ub = *ub_out;

        if(x_primal.size() != n_primal)
            x_primal.resize(n_primal);
        if(gcval.size() != m_dual)
            gcval.resize(m_dual);

        wval = Scalar(0);
        gval = f_out;
        for(int i = 0; i < m_dual; ++i)
            gcval[i] = fc[i];

        for(int j = 0; j < n_primal; ++j)
        {
            Scalar v = gf[j];
            for(int i = 0; i < m_dual; ++i)
                v += y[i] * dfc(i, j);

            Scalar s2 = sig[j] * sig[j];
            if(s2 < eps) { x_primal[j] = x[j]; continue; }

            Scalar dx = -s2 * v / u;
            if(std::abs(dx) > sig[j])
                dx = std::copysign(sig[j], dx);
            x_primal[j] = x[j] + dx;
            if(x_primal[j] > ub[j]) x_primal[j] = ub[j];
            else if(x_primal[j] < lb[j]) x_primal[j] = lb[j];
            dx = x_primal[j] - x[j];

            Scalar dx2sig = Scalar(0.5) * dx * dx / s2;
            wval += dx2sig;
            gval += gf[j] * dx + rho_out * dx2sig;
            for(int i = 0; i < m_dual; ++i)
                gcval[i] += dfc(i, j) * dx + (*rhoc_out)[i] * dx2sig;
        }
    }
};

}

#endif
