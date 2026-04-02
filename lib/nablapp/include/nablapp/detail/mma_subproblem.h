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
template <typename Scalar, int N = nablapp::dynamic_dimension>
struct mma_coeffs
{
    Eigen::Vector<Scalar, N> p0, q0;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> pi, qi;
    Eigen::VectorX<Scalar> r0;
    Eigen::VectorX<Scalar> ri;
};

// Compute reciprocal approximation coefficients from gradients.
//
// x:      current iterate (n)
// f:      objective value at x
// grad_f: objective gradient at x (n)
// g:      constraint values at x (m), with g[i] >= 0 feasible
// dg:     constraint Jacobian at x (m x n)
// L, U:   lower/upper asymptotes (n)
// epsilon: regularization for strict convexity
//
// The approximation matches function and gradient values at x.
//
// Reference: Svanberg 1987, eq. (2.1)-(2.5).
template <typename Scalar, int N = nablapp::dynamic_dimension>
mma_coeffs<Scalar, N> mma_coefficients(
    const Eigen::Vector<Scalar, N>& x,
    Scalar f,
    const Eigen::Vector<Scalar, N>& grad_f,
    const Eigen::VectorX<Scalar>& g,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, N>& dg,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U,
    const mma_subproblem_options& opts = {})
{
    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(g.size());
    const auto epsilon = static_cast<Scalar>(opts.regularization_epsilon.value_or(1e-7));

    mma_coeffs<Scalar, N> c;
    c.p0.resize(n);
    c.q0.resize(n);
    c.pi.resize(m, n);
    c.qi.resize(m, n);
    c.r0.resize(1);
    c.ri.resize(m);

    // Objective coefficients
    Scalar r0_val = f;
    for(int j = 0; j < n; ++j)
    {
        Scalar ux = U[j] - x[j];
        Scalar xl = x[j] - L[j];

        c.p0[j] = ux * ux * std::max(grad_f[j], Scalar(0)) + epsilon;
        c.q0[j] = xl * xl * std::max(-grad_f[j], Scalar(0)) + epsilon;

        r0_val -= c.p0[j] / ux + c.q0[j] / xl;
    }
    c.r0[0] = r0_val;

    // Constraint coefficients
    for(int i = 0; i < m; ++i)
    {
        Scalar ri_val = g[i];
        for(int j = 0; j < n; ++j)
        {
            Scalar ux = U[j] - x[j];
            Scalar xl = x[j] - L[j];

            c.pi(i, j) = ux * ux * std::max(dg(i, j), Scalar(0)) + epsilon;
            c.qi(i, j) = xl * xl * std::max(-dg(i, j), Scalar(0)) + epsilon;

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
template <typename Scalar, int N = nablapp::dynamic_dimension>
Scalar mma_subproblem_value(
    const mma_coeffs<Scalar, N>& coeffs,
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
template <typename Scalar, int N = nablapp::dynamic_dimension>
Scalar mma_subproblem_constraint(
    const mma_coeffs<Scalar, N>& coeffs,
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
template <typename Scalar, int N = nablapp::dynamic_dimension>
Eigen::Vector<Scalar, N> mma_dual_solve(
    const mma_coeffs<Scalar, N>& coeffs,
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
    Eigen::VectorX<Scalar> y = Eigen::VectorX<Scalar>::Ones(m);

    Eigen::Vector<Scalar, N> x_opt(n);

    const int max_iter = static_cast<int>(opts.dual_max_iterations.value_or(50));
    const auto tol = static_cast<Scalar>(opts.dual_tolerance.value_or(1e-9));
    const auto backtrack = static_cast<Scalar>(opts.backtrack_factor.value_or(0.95));

    Eigen::VectorX<Scalar> grad(m);
    Eigen::MatrixX<Scalar> negH(m, m);

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

        // Check convergence
        if(grad.cwiseAbs().maxCoeff() < tol)
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
        Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(std::move(negH));
        Eigen::VectorX<Scalar> dy = ldlt.solve(grad);

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
