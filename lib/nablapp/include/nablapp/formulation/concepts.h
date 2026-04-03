#ifndef HPP_GUARD_NABLAPP_FORMULATION_CONCEPTS_H
#define HPP_GUARD_NABLAPP_FORMULATION_CONCEPTS_H

#include "nablapp/types.h"
#include "nablapp/result/status.h"
#include "nablapp/result/step_result.h"
#include "nablapp/result/solve_result.h"

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

// Dimension extraction machinery. Every problem type MUST declare
// static constexpr int problem_dimension. There is no fallback -- if a
// type lacks this member, problem_dimension_v<T> fails to compile.

template <typename P>
concept has_problem_dimension = requires {
    { P::problem_dimension } -> std::convertible_to<int>;
};

template <has_problem_dimension P>
inline constexpr int problem_dimension_v = P::problem_dimension;

template <typename P, typename S = double>
concept objective = has_problem_dimension<P> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x)
{
    { p.value(x) } -> std::convertible_to<S>;
    { p.dimension() } -> std::convertible_to<int>;
};

template <typename P, typename S = double>
concept differentiable = objective<P, S> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x,
             Eigen::Vector<S, problem_dimension_v<P>>& g)
{
    { p.gradient(x, g) };
};

template <typename P, typename S = double>
concept second_order = differentiable<P, S> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x, Eigen::MatrixX<S>& H)
{
    { p.hessian(x, H) };
};

template <typename P, typename S = double>
concept bound_constrained = objective<P, S> &&
    requires(const P& p)
{
    { p.lower_bounds() } -> std::convertible_to<Eigen::Vector<S, problem_dimension_v<P>>>;
    { p.upper_bounds() } -> std::convertible_to<Eigen::Vector<S, problem_dimension_v<P>>>;
};

template <typename P, typename S = double>
concept constrained_values = objective<P, S> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x, Eigen::VectorX<S>& c)
{
    { p.constraints(x, c) };
    { p.num_equality() } -> std::convertible_to<int>;
    { p.num_inequality() } -> std::convertible_to<int>;
};

template <typename P, typename S = double>
concept constrained = constrained_values<P, S> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x, Eigen::MatrixX<S>& J)
{
    { p.constraint_jacobian(x, J) };
};

template <typename P, typename S = double>
concept least_squares = objective<P, S> &&
    requires(const P& p, const Eigen::Vector<S, problem_dimension_v<P>>& x,
             Eigen::VectorX<S>& r, Eigen::MatrixX<S>& J)
{
    { p.residuals(x, r) };
    { p.jacobian(x, J) };
    { p.num_residuals() } -> std::convertible_to<int>;
};

// NLP solver concept for downstream consumer injection (e.g., ctrlpp).
//
// Checks that a solver provides the full iterative execution model
// plus constraint violation diagnostics. Designed to be satisfied by
// basic_solver<Policy> for any constrained policy.
//
// Reference: D-07, D-08 from Phase 4 CONTEXT.
template <typename Solver, typename S = double>
concept nlp_solver = requires(Solver& solver, const Solver& csolver,
                              const Eigen::VectorX<S>& x0)
{
    typename Solver::scalar_type;
    typename Solver::state_type;
    { solver.step() } -> std::convertible_to<step_result<S>>;
    { solver.solve() } -> std::convertible_to<solve_result<S>>;
    { solver.step_n(int{}) } -> std::convertible_to<solve_result<S>>;
    { csolver.state() };
    { csolver.status() } -> std::convertible_to<solver_status>;
    { solver.reset(x0) };
    { solver.reset_clear(x0) };
    { solver.abort() };
    { csolver.constraint_violation() } -> std::convertible_to<S>;
};

}

#endif
