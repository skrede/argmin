#ifndef HPP_GUARD_NABLAPP_DETAIL_MMA_SUBPROBLEM_H
#define HPP_GUARD_NABLAPP_DETAIL_MMA_SUBPROBLEM_H

// MMA reciprocal separable approximation and dual solve.
//
// Provides the core computational kernels for MMA and GCMMA:
//   1. mma_coefficients: compute reciprocal approximation coefficients
//      from function/constraint values and gradients
//   2. mma_dual_solve: solve the dual of the separable convex subproblem
//      via Newton iteration with LDLT factorization
//   3. mma_subproblem_value: evaluate the MMA approximation at a point
//      (used by GCMMA for the conservatism check)
//
// Reference: Svanberg 1987; Svanberg 2002;
//            jdumas/mma reference implementation.

#include "nablapp/types.h"
#include "nablapp/options/mma_subproblem_options.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>

namespace nablapp::detail
{

// Reciprocal approximation coefficients for the MMA subproblem.
//
// p0, q0: objective coefficients (n)
// pi, qi: constraint coefficients (m x n)
// r0:     objective constant (scalar, stored as 1-element vector)
// ri:     constraint constants (m)
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
struct mma_coeffs
{
    Eigen::Vector<Scalar, N> p0, q0;
    Eigen::Matrix<Scalar, M, N> pi, qi;
    Eigen::Vector<Scalar, 1> r0;
    Eigen::Vector<Scalar, M> ri;
};

// Compute reciprocal approximation coefficients from gradients.
//
// x:      current iterate (n)
// f:      objective value at x
// grad_f: objective gradient at x (n)
// g:      constraint values at x (m), with g[i] >= 0 feasible
// dg:     constraint Jacobian at x (m x n)
// L, U:   lower/upper asymptotes (n)
// raa_0:  objective conservativity regularizer (0 for Svanberg 1987 MMA;
//         adaptive state for Svanberg 2002 GCMMA)
// raa:    per-constraint conservativity regularizer (zeros for MMA;
//         adaptive state vector for GCMMA)
// opts:   subproblem options; regularization_epsilon is a division-hygiene
//         floor only, NOT the primary regularizer.
//
// Svanberg 2002 Section 3 structured regularization:
//     p_0j = (U_j - x_j)^2 * ( max(df/dx_j, 0) + 0.001 * |df/dx_j|
//                              + raa_0 / (U_j - L_j) ) + eps_floor
//     q_0j = (x_j - L_j)^2 * ( max(-df/dx_j, 0) + 0.001 * |df/dx_j|
//                              + raa_0 / (U_j - L_j) ) + eps_floor
// with |g| = max(g, 0) + max(-g, 0) (split-gradient magnitude).
//
// The 0.001 * |g| stabilizer is active even at raa_0 = 0: this recovers
// Svanberg's own 1987 baseline mmasub.m coefficient form. The
// raa_0 / (U - L) term is Svanberg 2002's GCMMA-specific addition which
// grows adaptively on non-conservative trials.
//
// Reference: Svanberg 2002, SIAM J. Optim. 12(2), Section 3 (paper
//            notation rho_0j == raa_0); Svanberg 1987, IJNME 24,
//            Section 3 (recovered by raa_0 = 0).
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
mma_coeffs<Scalar, N, M> mma_coefficients(
    const Eigen::Vector<Scalar, N>& x,
    Scalar f,
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::Vector<Scalar, M>& g,
    const Eigen::Matrix<Scalar, M, N>& dg,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U,
    Scalar raa_0,
    const Eigen::Vector<Scalar, M>& raa,
    const mma_subproblem_options& opts = {})
{
    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(g.size());
    const auto eps_floor = static_cast<Scalar>(
        opts.regularization_epsilon.value_or(1e-10));

    mma_coeffs<Scalar, N, M> c;
    c.p0.resize(n);
    c.q0.resize(n);
    c.pi.resize(m, n);
    c.qi.resize(m, n);
    c.r0.resize(1);
    c.ri.resize(m);

    // Objective coefficients (Svanberg 2002 Section 3).
    Scalar r0_val = f;
    for(int j = 0; j < n; ++j)
    {
        const Scalar ux = U[j] - x[j];
        const Scalar xl = x[j] - L[j];
        const Scalar inv_UL = Scalar(1) / (U[j] - L[j]);

        const Scalar gp = std::max(grad_f[j], Scalar(0));
        const Scalar gn = std::max(-grad_f[j], Scalar(0));
        const Scalar abs_g = gp + gn;

        c.p0[j] = ux * ux
            * (gp + Scalar(0.001) * abs_g + raa_0 * inv_UL)
            + eps_floor;
        c.q0[j] = xl * xl
            * (gn + Scalar(0.001) * abs_g + raa_0 * inv_UL)
            + eps_floor;

        r0_val -= c.p0[j] / ux + c.q0[j] / xl;
    }
    c.r0[0] = r0_val;

    // Constraint coefficients: same structure with per-constraint raa[i].
    //
    // Reference: Svanberg 2002, Section 3 (constraint-level regularization).
    for(int i = 0; i < m; ++i)
    {
        Scalar ri_val = g[i];
        for(int j = 0; j < n; ++j)
        {
            const Scalar ux = U[j] - x[j];
            const Scalar xl = x[j] - L[j];
            const Scalar inv_UL = Scalar(1) / (U[j] - L[j]);

            const Scalar dgp = std::max(dg(i, j), Scalar(0));
            const Scalar dgn = std::max(-dg(i, j), Scalar(0));
            const Scalar abs_dg = dgp + dgn;

            c.pi(i, j) = ux * ux
                * (dgp + Scalar(0.001) * abs_dg + raa[i] * inv_UL)
                + eps_floor;
            c.qi(i, j) = xl * xl
                * (dgn + Scalar(0.001) * abs_dg + raa[i] * inv_UL)
                + eps_floor;

            ri_val -= c.pi(i, j) / ux + c.qi(i, j) / xl;
        }
        c.ri[i] = ri_val;
    }

    return c;
}

// Evaluate the MMA approximation of the objective at point x.
//
// Returns: sum_j(p0[j]/(U[j]-x[j]) + q0[j]/(x[j]-L[j])) + r0
//
// Used by GCMMA to check conservatism.
//
// Reference: Svanberg 2002, conservatism condition.
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
Scalar mma_subproblem_value(
    const mma_coeffs<Scalar, N, M>& coeffs,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U)
{
    const int n = static_cast<int>(x.size());
    Scalar val = coeffs.r0[0];
    for(int j = 0; j < n; ++j)
        val += coeffs.p0[j] / (U[j] - x[j]) + coeffs.q0[j] / (x[j] - L[j]);
    return val;
}

// Evaluate the MMA approximation of constraint i at point x.
//
// Returns: sum_j(pi[i][j]/(U[j]-x[j]) + qi[i][j]/(x[j]-L[j])) + ri[i]
//
// Reference: Svanberg 2002.
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
Scalar mma_subproblem_constraint(
    const mma_coeffs<Scalar, N, M>& coeffs,
    int i,
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U)
{
    const int n = static_cast<int>(x.size());
    Scalar val = coeffs.ri[i];
    for(int j = 0; j < n; ++j)
        val += coeffs.pi(i, j) / (U[j] - x[j]) + coeffs.qi(i, j) / (x[j] - L[j]);
    return val;
}

// Stateful MMA subproblem solver with pre-allocated workspace.
//
// Owns LDLT factorization object, coefficient storage, and dual solve
// buffers. Eliminates per-call allocation in the Newton loop.
//
// Template parameters:
//   N — problem dimension (compile-time or dynamic_dimension)
//   M — constraint count (compile-time or dynamic_dimension)
//
// When M is compile-time, the dual Newton system (LDLT, gradient,
// Hessian, step) uses fixed-size M×M types — zero heap allocation.
//
// Reference: Svanberg 1987; Svanberg 2002, Section 5.
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
class mma_subproblem_solver
{
    using constraint_vector = Eigen::Vector<Scalar, M>;
    using constraint_matrix = Eigen::Matrix<Scalar, M, N>;
    using dual_matrix = Eigen::Matrix<Scalar, M, M>;

public:
    explicit mma_subproblem_solver(int n, int m)
        : n_{n}, m_{m}
    {
        coeffs_.p0.resize(n);
        coeffs_.q0.resize(n);
        coeffs_.pi.resize(m, n);
        coeffs_.qi.resize(m, n);
        coeffs_.r0.resize(1);
        coeffs_.ri.resize(m);

        y_.resize(m);
        x_opt_.resize(n);
        dual_grad_.resize(m);
        dual_hess_.resize(m, m);
        dy_.resize(m);
    }

    mma_subproblem_solver() = default;

    // Compute MMA approximation coefficients into pre-allocated workspace.
    //
    // raa_0, raa: conservativity regularizers. Zero for Svanberg 1987 MMA
    //             (the 0.001 * |grad| stabilizer is always active);
    //             adaptive state members for Svanberg 2002 GCMMA.
    //
    // Reference: Svanberg 2002, SIAM J. Optim. 12(2), Section 3
    //            (structured regularization); Svanberg 1987 Section 3
    //            (baseline recovered by raa_0 = 0).
    void compute_coefficients(
        const Eigen::Vector<Scalar, N>& x,
        Scalar f,
        const Eigen::Vector<Scalar, N>& grad_f,
        const constraint_vector& g,
        const constraint_matrix& dg,
        const Eigen::Vector<Scalar, N>& L,
        const Eigen::Vector<Scalar, N>& U,
        Scalar raa_0,
        const constraint_vector& raa,
        const mma_subproblem_options& opts = {})
    {
        const auto eps_floor = static_cast<Scalar>(
            opts.regularization_epsilon.value_or(1e-10));

        // Objective coefficients (Svanberg 2002 Section 3):
        //     p_0j = (U-x)^2 * ( max(df/dx, 0) + 0.001 * |df/dx|
        //                        + raa_0 / (U-L) ) + eps_floor
        //     q_0j = (x-L)^2 * ( max(-df/dx, 0) + 0.001 * |df/dx|
        //                        + raa_0 / (U-L) ) + eps_floor
        Scalar r0_val = f;
        for(int j = 0; j < n_; ++j)
        {
            const Scalar ux = U[j] - x[j];
            const Scalar xl = x[j] - L[j];
            const Scalar inv_UL = Scalar(1) / (U[j] - L[j]);

            const Scalar gp = std::max(grad_f[j], Scalar(0));
            const Scalar gn = std::max(-grad_f[j], Scalar(0));
            const Scalar abs_g = gp + gn;

            coeffs_.p0[j] = ux * ux
                * (gp + Scalar(0.001) * abs_g + raa_0 * inv_UL)
                + eps_floor;
            coeffs_.q0[j] = xl * xl
                * (gn + Scalar(0.001) * abs_g + raa_0 * inv_UL)
                + eps_floor;

            r0_val -= coeffs_.p0[j] / ux + coeffs_.q0[j] / xl;
        }
        coeffs_.r0[0] = r0_val;

        // Constraint coefficients: same Svanberg 2002 Section 3 structure
        // with per-constraint raa[i] in place of raa_0.
        for(int i = 0; i < m_; ++i)
        {
            Scalar ri_val = g[i];
            for(int j = 0; j < n_; ++j)
            {
                const Scalar ux = U[j] - x[j];
                const Scalar xl = x[j] - L[j];
                const Scalar inv_UL = Scalar(1) / (U[j] - L[j]);

                const Scalar dgp = std::max(dg(i, j), Scalar(0));
                const Scalar dgn = std::max(-dg(i, j), Scalar(0));
                const Scalar abs_dg = dgp + dgn;

                coeffs_.pi(i, j) = ux * ux
                    * (dgp + Scalar(0.001) * abs_dg + raa[i] * inv_UL)
                    + eps_floor;
                coeffs_.qi(i, j) = xl * xl
                    * (dgn + Scalar(0.001) * abs_dg + raa[i] * inv_UL)
                    + eps_floor;

                ri_val -= coeffs_.pi(i, j) / ux + coeffs_.qi(i, j) / xl;
            }
            coeffs_.ri[i] = ri_val;
        }
    }

    // Solve dual problem via Newton iteration using pre-allocated LDLT.
    //
    // Reference: Svanberg 1987, dual formulation; Svanberg 2002, Section 5.
    Eigen::Vector<Scalar, N> dual_solve(
        const Eigen::Vector<Scalar, N>& L,
        const Eigen::Vector<Scalar, N>& U,
        const Eigen::Vector<Scalar, N>& x_min,
        const Eigen::Vector<Scalar, N>& x_max,
        const mma_subproblem_options& opts = {})
    {
        constexpr Scalar eps = Scalar(1e-10);

        const int max_iter = static_cast<int>(opts.dual_max_iterations.value_or(50));
        const auto tol = static_cast<Scalar>(opts.dual_tolerance.value_or(1e-9));
        const auto backtrack = static_cast<Scalar>(opts.backtrack_factor.value_or(0.95));

        if(m_ == 0)
        {
            Eigen::Vector<Scalar, N> x_opt(n_);
            for(int j = 0; j < n_; ++j)
            {
                Scalar sp = std::sqrt(coeffs_.p0[j]);
                Scalar sq = std::sqrt(coeffs_.q0[j]);
                x_opt[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
                x_opt[j] = std::clamp(x_opt[j],
                    std::max(L[j] + eps, x_min[j]),
                    std::min(U[j] - eps, x_max[j]));
            }
            return x_opt;
        }

        y_.setOnes();

        for(int iter = 0; iter < max_iter; ++iter)
        {
            for(int j = 0; j < n_; ++j)
            {
                Scalar pj = coeffs_.p0[j];
                Scalar qj = coeffs_.q0[j];
                for(int i = 0; i < m_; ++i)
                {
                    pj += y_[i] * coeffs_.pi(i, j);
                    qj += y_[i] * coeffs_.qi(i, j);
                }
                pj = std::max(pj, eps);
                qj = std::max(qj, eps);

                Scalar sp = std::sqrt(pj);
                Scalar sq = std::sqrt(qj);
                x_opt_[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
                x_opt_[j] = std::clamp(x_opt_[j],
                    std::max(L[j] + eps, x_min[j]),
                    std::min(U[j] - eps, x_max[j]));
            }

            for(int i = 0; i < m_; ++i)
            {
                dual_grad_[i] = coeffs_.ri[i];
                for(int j = 0; j < n_; ++j)
                {
                    Scalar ux = U[j] - x_opt_[j];
                    Scalar xl = x_opt_[j] - L[j];
                    dual_grad_[i] += coeffs_.pi(i, j) / ux + coeffs_.qi(i, j) / xl;
                }
            }

            // Projected gradient convergence: for y_i > 0, |grad_i| < tol;
            // for y_i ≈ 0, only the positive part matters (negative gradient
            // means the constraint is satisfied and the dual variable is
            // correctly at zero). Reference: Svanberg 1987, KKT conditions.
            Scalar max_proj = Scalar(0);
            for(int i = 0; i < m_; ++i)
            {
                if(y_[i] > eps)
                    max_proj = std::max(max_proj, std::abs(dual_grad_[i]));
                else
                    max_proj = std::max(max_proj, std::max(dual_grad_[i], Scalar(0)));
            }
            if(max_proj < tol)
                break;

            dual_hess_.setZero();
            for(int j = 0; j < n_; ++j)
            {
                Scalar pj = coeffs_.p0[j];
                Scalar qj = coeffs_.q0[j];
                for(int i = 0; i < m_; ++i)
                {
                    pj += y_[i] * coeffs_.pi(i, j);
                    qj += y_[i] * coeffs_.qi(i, j);
                }
                pj = std::max(pj, eps);
                qj = std::max(qj, eps);

                Scalar aj = U[j] - x_opt_[j];
                Scalar bj = x_opt_[j] - L[j];
                aj = std::max(aj, eps);
                bj = std::max(bj, eps);

                Scalar wj = Scalar(2) * pj / (aj * aj * aj)
                          + Scalar(2) * qj / (bj * bj * bj);
                wj = std::max(wj, eps);

                for(int i1 = 0; i1 < m_; ++i1)
                {
                    Scalar v1 = coeffs_.pi(i1, j) / (aj * aj)
                              - coeffs_.qi(i1, j) / (bj * bj);
                    for(int i2 = i1; i2 < m_; ++i2)
                    {
                        Scalar v2 = coeffs_.pi(i2, j) / (aj * aj)
                                  - coeffs_.qi(i2, j) / (bj * bj);
                        Scalar val = v1 * v2 / wj;
                        dual_hess_(i1, i2) += val;
                        if(i1 != i2)
                            dual_hess_(i2, i1) += val;
                    }
                }
            }

            dual_hess_.diagonal().array() += eps;

            ldlt_.compute(dual_hess_);
            dy_ = ldlt_.solve(dual_grad_);

            Scalar alpha = Scalar(1);
            for(int i = 0; i < m_; ++i)
            {
                if(dy_[i] < Scalar(0) && y_[i] > Scalar(0))
                {
                    Scalar max_step = y_[i] / (-dy_[i]) * backtrack;
                    alpha = std::min(alpha, max_step);
                }
            }
            alpha = std::max(alpha, eps);

            y_ += alpha * dy_;
            y_ = y_.cwiseMax(Scalar(0));
        }

        return x_opt_;
    }

    // Evaluate the MMA approximation of the objective at point x.
    //
    // Reference: Svanberg 2002, conservatism condition.
    Scalar subproblem_value(
        const Eigen::Vector<Scalar, N>& x,
        const Eigen::Vector<Scalar, N>& L,
        const Eigen::Vector<Scalar, N>& U) const
    {
        Scalar val = coeffs_.r0[0];
        for(int j = 0; j < n_; ++j)
            val += coeffs_.p0[j] / (U[j] - x[j]) + coeffs_.q0[j] / (x[j] - L[j]);
        return val;
    }

    // Evaluate the MMA approximation of constraint i at point x.
    //
    // Reference: Svanberg 2002.
    Scalar subproblem_constraint(
        int i,
        const Eigen::Vector<Scalar, N>& x,
        const Eigen::Vector<Scalar, N>& L,
        const Eigen::Vector<Scalar, N>& U) const
    {
        Scalar val = coeffs_.ri[i];
        for(int j = 0; j < n_; ++j)
            val += coeffs_.pi(i, j) / (U[j] - x[j]) + coeffs_.qi(i, j) / (x[j] - L[j]);
        return val;
    }

    const mma_coeffs<Scalar, N, M>& coefficients() const { return coeffs_; }

    // Returns the subproblem dual multipliers y_ as populated by the last
    // dual_solve call. Valid immediately after dual_solve returns; callers
    // must read before the next compute_coefficients / dual_solve round.
    //
    // Zero-overhead const reference: no copy, no allocation, noexcept.
    //
    // Sign convention: y_i >= 0 is the dual feasibility condition for the
    // MMA constraint g_i(x) <= 0. Under nablapp's c_ineq >= 0 convention
    // with g_i = -c_ineq_i, the stationarity expression reads
    //     grad_L = grad_f - sum_i y_i * grad(c_ineq_i),
    // matching nablapp's Lagrangian sign convention in
    // detail/lagrangian.h (lines 42-60) -- y_i maps directly to nablapp's
    // mu_ineq_i with no flip.
    //
    // Reference: Svanberg 1987, "The method of moving asymptotes",
    //            Section 5 (dual KKT); Svanberg 2002, SIAM J. Optim.
    //            12(2), Section 5.
    const constraint_vector& multipliers() const noexcept { return y_; }

private:
    int n_{0};
    int m_{0};

    mma_coeffs<Scalar, N, M> coeffs_;

    constraint_vector y_;
    Eigen::Vector<Scalar, N> x_opt_;
    constraint_vector dual_grad_;
    dual_matrix dual_hess_;
    Eigen::LDLT<dual_matrix> ldlt_;
    constraint_vector dy_;
};

// Solve the MMA dual problem via Newton iteration.
//
// The MMA subproblem is separable and convex. For given dual variables
// y (one per constraint), the optimal primal x_j is found analytically
// from the KKT condition of the separable Lagrangian. The dual function
// is then maximized via Newton iteration.
//
// Returns the optimal primal x (n-vector).
//
// coeffs:  MMA approximation coefficients
// L, U:    lower/upper asymptotes (n)
// x_min, x_max: variable bounds (n)
// max_iter: maximum Newton iterations for dual solve
// tol:     convergence tolerance on dual gradient infinity norm
//
// Reference: Svanberg 1987, dual formulation; Svanberg 2002, Section 5.
template <typename Scalar, int N = nablapp::dynamic_dimension, int M = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> mma_dual_solve(
    const mma_coeffs<Scalar, N, M>& coeffs,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U,
    const Eigen::Vector<Scalar, N>& x_min,
    const Eigen::Vector<Scalar, N>& x_max,
    const mma_subproblem_options& opts = {})
{
    const int n = static_cast<int>(L.size());
    const int m = static_cast<int>(coeffs.ri.size());

    constexpr Scalar eps = Scalar(1e-10);

    // No constraints: solve unconstrained primal directly
    if(m == 0)
    {
        Eigen::Vector<Scalar, N> x_opt(n);
        for(int j = 0; j < n; ++j)
        {
            Scalar sp = std::sqrt(coeffs.p0[j]);
            Scalar sq = std::sqrt(coeffs.q0[j]);
            x_opt[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
            x_opt[j] = std::clamp(x_opt[j],
                std::max(L[j] + eps, x_min[j]),
                std::min(U[j] - eps, x_max[j]));
        }
        return x_opt;
    }

    // Dual variables initialized to 1
    Eigen::Vector<Scalar, M> y = Eigen::Vector<Scalar, M>::Ones(m);

    Eigen::Vector<Scalar, N> x_opt(n);

    const int max_iter = static_cast<int>(opts.dual_max_iterations.value_or(50));
    const auto tol = static_cast<Scalar>(opts.dual_tolerance.value_or(1e-9));
    const auto backtrack = static_cast<Scalar>(opts.backtrack_factor.value_or(0.95));

    Eigen::Vector<Scalar, M> grad(m);
    Eigen::Matrix<Scalar, M, M> negH(m, m);

    for(int iter = 0; iter < max_iter; ++iter)
    {
        // For current y, compute optimal primal x_j analytically.
        // p_j = p0[j] + sum_i(y[i]*pi[i][j])
        // q_j = q0[j] + sum_i(y[i]*qi[i][j])
        // x_j = (sqrt(p_j)*L[j] + sqrt(q_j)*U[j]) / (sqrt(p_j) + sqrt(q_j))

        for(int j = 0; j < n; ++j)
        {
            Scalar pj = coeffs.p0[j];
            Scalar qj = coeffs.q0[j];
            for(int i = 0; i < m; ++i)
            {
                pj += y[i] * coeffs.pi(i, j);
                qj += y[i] * coeffs.qi(i, j);
            }
            pj = std::max(pj, eps);
            qj = std::max(qj, eps);

            Scalar sp = std::sqrt(pj);
            Scalar sq = std::sqrt(qj);
            x_opt[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
            x_opt[j] = std::clamp(x_opt[j],
                std::max(L[j] + eps, x_min[j]),
                std::min(U[j] - eps, x_max[j]));
        }

        // Compute dual gradient: dW/dy_i = constraint approximation value at x_opt
        for(int i = 0; i < m; ++i)
        {
            grad[i] = coeffs.ri[i];
            for(int j = 0; j < n; ++j)
            {
                Scalar ux = U[j] - x_opt[j];
                Scalar xl = x_opt[j] - L[j];
                grad[i] += coeffs.pi(i, j) / ux + coeffs.qi(i, j) / xl;
            }
        }

        // Projected gradient convergence check (see stateful solver above).
        Scalar max_proj = Scalar(0);
        for(int i = 0; i < m; ++i)
        {
            if(y[i] > eps)
                max_proj = std::max(max_proj, std::abs(grad[i]));
            else
                max_proj = std::max(max_proj, std::max(grad[i], Scalar(0)));
        }
        if(max_proj < tol)
            break;

        // Compute the negative dual Hessian (-d^2W/dy^2), which is positive
        // semidefinite. Uses implicit differentiation of KKT condition
        // dL/dx_j = 0 to get dx_j/dy_k.
        //
        // -H_{i1,i2} = sum_j [ dg_{i1}/dx_j * (-dx_j/dy_{i2}) ]
        //            = sum_j [ v_{i1,j} * v_{i2,j} / w_j ]
        //
        // where v_{i,j} = pi_{i,j}/a_j^2 - qi_{i,j}/b_j^2
        //       w_j     = 2*P_j/a_j^3 + 2*Q_j/b_j^3
        //       a_j     = U_j - x_j, b_j = x_j - L_j
        //
        // Reference: Svanberg 1987, dual Newton system.
        negH.setZero();
        for(int j = 0; j < n; ++j)
        {
            Scalar pj = coeffs.p0[j];
            Scalar qj = coeffs.q0[j];
            for(int i = 0; i < m; ++i)
            {
                pj += y[i] * coeffs.pi(i, j);
                qj += y[i] * coeffs.qi(i, j);
            }
            pj = std::max(pj, eps);
            qj = std::max(qj, eps);

            Scalar aj = U[j] - x_opt[j];
            Scalar bj = x_opt[j] - L[j];
            aj = std::max(aj, eps);
            bj = std::max(bj, eps);

            Scalar wj = Scalar(2) * pj / (aj * aj * aj)
                      + Scalar(2) * qj / (bj * bj * bj);
            wj = std::max(wj, eps);

            for(int i1 = 0; i1 < m; ++i1)
            {
                Scalar v1 = coeffs.pi(i1, j) / (aj * aj)
                          - coeffs.qi(i1, j) / (bj * bj);
                for(int i2 = i1; i2 < m; ++i2)
                {
                    Scalar v2 = coeffs.pi(i2, j) / (aj * aj)
                              - coeffs.qi(i2, j) / (bj * bj);
                    Scalar val = v1 * v2 / wj;
                    negH(i1, i2) += val;
                    if(i1 != i2)
                        negH(i2, i1) += val;
                }
            }
        }

        // Regularize
        negH.diagonal().array() += eps;

        // Newton ascent: dy = (-H)^{-1} * grad, then y += alpha * dy.
        // This maximizes the concave dual W(y).
        Eigen::LDLT<Eigen::Matrix<Scalar, M, M>> ldlt(std::move(negH));
        Eigen::Vector<Scalar, M> dy = ldlt.solve(grad);

        // Line search: backtrack to keep y >= 0
        Scalar alpha = Scalar(1);
        for(int i = 0; i < m; ++i)
        {
            if(dy[i] < Scalar(0) && y[i] > Scalar(0))
            {
                Scalar max_step = y[i] / (-dy[i]) * backtrack;
                alpha = std::min(alpha, max_step);
            }
        }
        alpha = std::max(alpha, eps);

        y += alpha * dy;

        // Project to non-negative orthant
        y = y.cwiseMax(Scalar(0));
    }

    return x_opt;
}

}

#endif
