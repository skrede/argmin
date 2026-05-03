#ifndef HPP_GUARD_ARGMIN_DERIVATIVE_FINITE_DIFFERENCE_H
#define HPP_GUARD_ARGMIN_DERIVATIVE_FINITE_DIFFERENCE_H

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace argmin
{

// Central-difference gradient approximation.
//
// df/dx_i = (f(x + h*e_i/2) - f(x - h*e_i/2)) / h
//
// Adaptive step: h_i = cbrt(eps) * max(1, |x_i|).
// cbrt(eps) is optimal for central differences -- it balances O(h^2)
// truncation error against O(eps/h) roundoff error.
//
// N is extracted from Problem::problem_dimension. When the problem declares
// a compile-time dimension, all vectors are fixed-size. When the problem uses
// dynamic_dimension, vectors remain dynamic (VectorXd).
//
// Reference: K&W Section 2.3.1, Eq. 2.18, Algorithm 2.1.

template <typename Problem, typename Scalar = double,
          int N = problem_dimension_v<Problem>>
    requires objective<Problem, Scalar>
void fd_gradient(const Problem& p,
                 const Eigen::Vector<Scalar, N>& x,
                 Eigen::Vector<Scalar, N>& g)
{
    const int n = x.size();
    g.resize(n);
    const Scalar cbrt_eps = std::cbrt(std::numeric_limits<Scalar>::epsilon());
    Eigen::Vector<Scalar, N> x_perturbed = x;

    for(int i = 0; i < n; ++i)
    {
        const Scalar h = cbrt_eps * std::max(Scalar(1), std::abs(x(i)));
        const Scalar half_h = h / Scalar(2);
        x_perturbed(i) = x(i) + half_h;
        const Scalar f_plus = p.value(x_perturbed);
        x_perturbed(i) = x(i) - half_h;
        const Scalar f_minus = p.value(x_perturbed);
        g(i) = (f_plus - f_minus) / h;
        x_perturbed(i) = x(i);
    }
}

// Central-difference Jacobian for vector-valued functions.
//
// J_{ij} = (F_i(x + h*e_j/2) - F_i(x - h*e_j/2)) / h
//
// Column-by-column computation with the same adaptive step as fd_gradient.
// The VectorFunction interface is: void f(const Vector<S,N>& x, VectorX<S>& out).
// This matches the constraints(x, c) and residuals(x, r) patterns from the
// concept hierarchy, so fd_jacobian works for constraint and residual Jacobians.
//
// N is the input (column) dimension. The output (row) dimension M is runtime
// since residual and constraint counts may not be compile-time constants.
//
// Reference: K&W Section 2.3.1 (applied to vector-valued functions).

template <int N = dynamic_dimension, typename VectorFunction, typename Scalar = double>
void fd_jacobian(const VectorFunction& f,
                 const Eigen::Vector<Scalar, N>& x,
                 Eigen::MatrixX<Scalar>& J,
                 int num_outputs)
{
    const int n = x.size();
    J.resize(num_outputs, n);
    const Scalar cbrt_eps = std::cbrt(std::numeric_limits<Scalar>::epsilon());
    Eigen::Vector<Scalar, N> x_perturbed = x;
    Eigen::VectorX<Scalar> f_plus(num_outputs);
    Eigen::VectorX<Scalar> f_minus(num_outputs);

    for(int j = 0; j < n; ++j)
    {
        const Scalar h = cbrt_eps * std::max(Scalar(1), std::abs(x(j)));
        const Scalar half_h = h / Scalar(2);
        x_perturbed(j) = x(j) + half_h;
        f(x_perturbed, f_plus);
        x_perturbed(j) = x(j) - half_h;
        f(x_perturbed, f_minus);
        J.col(j) = (f_plus - f_minus) / h;
        x_perturbed(j) = x(j);
    }
}

}

#endif
