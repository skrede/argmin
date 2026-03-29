#ifndef HPP_GUARD_NABLAPP_FORMULATION_CONCEPTS_H
#define HPP_GUARD_NABLAPP_FORMULATION_CONCEPTS_H

#include <Eigen/Core>

#include <concepts>

namespace nablapp
{

// Concept hierarchy for optimization problem formulations.
//
// Maps the textbook taxonomy (K&W Ch. 1, N&W Ch. 1) to C++23 concepts:
//   objective          -- f(x) : R^n -> R
//   differentiable     -- objective + gradient
//   second_order       -- differentiable + Hessian
//   bound_constrained  -- objective + box bounds
//   constrained        -- objective + equality/inequality constraints
//   least_squares      -- objective + residuals + Jacobian
//
// All concepts are parameterised on Scalar (defaulting to double) per D-07.
// Problem types satisfy concepts by duck-typing -- no base class required.

template <typename P, typename S = double>
concept objective = requires(const P& p, const Eigen::VectorX<S>& x)
{
    { p.value(x) } -> std::convertible_to<S>;
    { p.dimension() } -> std::convertible_to<int>;
};

template <typename P, typename S = double>
concept differentiable = objective<P, S> &&
    requires(const P& p, const Eigen::VectorX<S>& x, Eigen::VectorX<S>& g)
{
    { p.gradient(x, g) };
};

template <typename P, typename S = double>
concept second_order = differentiable<P, S> &&
    requires(const P& p, const Eigen::VectorX<S>& x, Eigen::MatrixX<S>& H)
{
    { p.hessian(x, H) };
};

template <typename P, typename S = double>
concept bound_constrained = objective<P, S> &&
    requires(const P& p)
{
    { p.lower_bounds() } -> std::convertible_to<Eigen::VectorX<S>>;
    { p.upper_bounds() } -> std::convertible_to<Eigen::VectorX<S>>;
};

template <typename P, typename S = double>
concept constrained = objective<P, S> &&
    requires(const P& p, const Eigen::VectorX<S>& x, Eigen::VectorX<S>& c, Eigen::MatrixX<S>& J)
{
    { p.constraints(x, c) };
    { p.constraint_jacobian(x, J) };
    { p.num_equality() } -> std::convertible_to<int>;
    { p.num_inequality() } -> std::convertible_to<int>;
};

template <typename P, typename S = double>
concept least_squares = objective<P, S> &&
    requires(const P& p, const Eigen::VectorX<S>& x, Eigen::VectorX<S>& r, Eigen::MatrixX<S>& J)
{
    { p.residuals(x, r) };
    { p.jacobian(x, J) };
    { p.num_residuals() } -> std::convertible_to<int>;
};

}

#endif
