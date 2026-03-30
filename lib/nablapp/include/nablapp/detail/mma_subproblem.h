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
template <typename Scalar>
struct mma_coeffs
{
    Eigen::VectorX<Scalar> p0, q0;
    Eigen::MatrixX<Scalar> pi, qi;
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
template <typename Scalar>
mma_coeffs<Scalar> mma_coefficients(
    const Eigen::VectorX<Scalar>& x,
    Scalar f,
    const Eigen::VectorX<Scalar>& grad_f,
    const Eigen::VectorX<Scalar>& g,
    const Eigen::MatrixX<Scalar>& dg,
    const Eigen::VectorX<Scalar>& L,
    const Eigen::VectorX<Scalar>& U,
    Scalar epsilon = Scalar(1e-7))
{
    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(g.size());

    mma_coeffs<Scalar> c;
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
template <typename Scalar>
Scalar mma_subproblem_value(
    const mma_coeffs<Scalar>& coeffs,
    const Eigen::VectorX<Scalar>& x,
    const Eigen::VectorX<Scalar>& L,
    const Eigen::VectorX<Scalar>& U)
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
template <typename Scalar>
Scalar mma_subproblem_constraint(
    const mma_coeffs<Scalar>& coeffs,
    int i,
    const Eigen::VectorX<Scalar>& x,
    const Eigen::VectorX<Scalar>& L,
    const Eigen::VectorX<Scalar>& U)
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
template <typename Scalar>
Eigen::VectorX<Scalar> mma_dual_solve(
    const mma_coeffs<Scalar>& coeffs,
    const Eigen::VectorX<Scalar>& L,
    const Eigen::VectorX<Scalar>& U,
    const Eigen::VectorX<Scalar>& x_min,
    const Eigen::VectorX<Scalar>& x_max,
    int max_iter = 50,
    Scalar tol = Scalar(1e-9))
{
    const int n = static_cast<int>(L.size());
    const int m = static_cast<int>(coeffs.ri.size());

    constexpr Scalar eps = Scalar(1e-10);

    // No constraints: solve unconstrained primal directly
    if(m == 0)
    {
        Eigen::VectorX<Scalar> x_opt(n);
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

    Eigen::VectorX<Scalar> x_opt(n);

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
        Eigen::VectorX<Scalar> grad(m);
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

        // Compute dual Hessian (m x m, symmetric)
        // d^2W/dy_i dy_k = -sum_j [ (dp_j/dy_i * dp_j/dy_k) / (4*pj^{3/2})
        //                          + (dq_j/dy_i * dq_j/dy_k) / (4*qj^{3/2}) ]
        //                  * dx_j/d(stuff)
        // In practice, we compute it from the second derivative of the dual function.
        // For the separable structure, the Hessian has a simple form.

        Eigen::MatrixX<Scalar> H = Eigen::MatrixX<Scalar>::Zero(m, m);
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

            Scalar ux = U[j] - x_opt[j];
            Scalar xl = x_opt[j] - L[j];

            // dx_j/dy_i contribution to Hessian
            // The dual Hessian is negative semidefinite (we maximize the dual).
            // For Newton: H * dy = -grad, where H is the negative dual Hessian
            // (so H is positive semidefinite).
            Scalar sp = std::sqrt(pj);
            Scalar sq = std::sqrt(qj);
            Scalar denom = sp + sq;
            Scalar factor = (U[j] - L[j]) / (Scalar(2) * denom * denom * denom);

            for(int i1 = 0; i1 < m; ++i1)
            {
                Scalar a1 = coeffs.pi(i1, j) / sp - coeffs.qi(i1, j) / sq;
                for(int i2 = i1; i2 < m; ++i2)
                {
                    Scalar a2 = coeffs.pi(i2, j) / sp - coeffs.qi(i2, j) / sq;
                    Scalar val = factor * a1 * a2;
                    H(i1, i2) += val;
                    if(i1 != i2)
                        H(i2, i1) += val;
                }
            }
        }

        // Regularize
        H.diagonal().array() += eps;

        // Newton step: H * dy = -grad (note: we use the negative Hessian
        // since we maximize the dual, so -H is psd and we solve H dy = grad)
        Eigen::LDLT<Eigen::MatrixX<Scalar>> ldlt(H);
        Eigen::VectorX<Scalar> dy = ldlt.solve(grad);

        // Line search: backtrack to keep y >= 0
        Scalar alpha = Scalar(1);
        for(int i = 0; i < m; ++i)
        {
            if(dy[i] < Scalar(0) && y[i] > Scalar(0))
            {
                Scalar max_step = -y[i] / dy[i] * Scalar(0.95);
                alpha = std::min(alpha, max_step);
            }
        }
        alpha = std::max(alpha, eps);

        y -= alpha * dy;

        // Project to non-negative orthant
        y = y.cwiseMax(Scalar(0));
    }

    return x_opt;
}

}

#endif
