#ifndef HPP_GUARD_ARGMIN_FORMULATION_CONCEPTS_H
#define HPP_GUARD_ARGMIN_FORMULATION_CONCEPTS_H

#include "argmin/types.h"
#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <cstddef>
#include <concepts>

namespace argmin
{

// Concept hierarchy for optimization problem formulations.
//
// Maps the textbook taxonomy (K&W Ch. 1, N&W Ch. 1) to C++20 concepts:
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

// Constraint count extraction. Constrained problem types MAY declare
// static constexpr int constraint_count (total equality + inequality).
// When present, enables fixed-size constraint vectors and Jacobians
// throughout the solver pipeline. When absent, constraint dimensions
// remain dynamic. Reference: D-03.

template <typename P>
concept has_constraint_count = requires {
    { P::constraint_count } -> std::convertible_to<int>;
};

template <has_constraint_count P>
inline constexpr int constraint_count_v = P::constraint_count;

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

// Harness contract for solver states.
//
// basic_solver dereferences exactly one member on every driven state: x, the
// current iterate. step() reads state.x for the x_norm fill, and both reset
// paths hand a fresh x0 to the policy which writes it into state.x. This is the
// minimal, mandatory surface every policy state must expose.
template <typename State>
concept solver_state = requires(const State& s)
{
    { s.x };
};

// Opt-in refinement for constrained policy states.
//
// The best-seen seeding and the solver's constraint_violation() reporting probe
// c_eq / c_ineq behind if-constexpr guards so unconstrained policies (lbfgsb,
// cmaes, ...) are never forced to carry dead constraint members. This concept
// names that opt-in surface for the constrained paths without folding it into
// the mandatory solver_state contract.
template <typename State>
concept constrained_policy_state = solver_state<State> &&
    requires(const State& s)
{
    { s.c_eq };
    { s.c_ineq };
};

// Harness contract for solver policies (duck-typed, no base class).
//
// basic_solver drives a policy through init/step/reset/reset_clear and reads
// state.x. solver_policy pins exactly the harness-visible surface so an
// ill-formed policy fails at the construction site with a one-line diagnostic
// instead of a deep template-instantiation error.
//
// init(problem, x0, opts) is deliberately NOT constrained structurally: it is
// templated on the problem and convergence types and its shape is an
// implementation detail; freezing it would block workspace-layout evolution.
//
// Move-safety is a documented invariant, not a compile-time check: a policy
// with self-referential state (e.g. a subproblem caching addresses of sibling
// state fields) must box those self-references so they survive basic_solver's
// noexcept move.
template <typename Policy, typename State, typename S = double>
concept solver_policy = solver_state<State> &&
    requires(Policy& policy, State& state, const Eigen::VectorX<S>& x0)
{
    typename Policy::scalar_type;
    { policy.step(state) } -> std::convertible_to<step_result<S>>;
    { policy.reset(state, x0) };
    { policy.reset_clear(state, x0) };
};

// Harness contract for solver-group schedules (duck-typed, no base class).
//
// basic_solver_group unconditionally drives a schedule through select() and
// notify() each step and reset() when the group is (re)initialized. schedule
// pins that mandatory surface. set_num_solvers() is intentionally left out of
// the contract: only stateful schedules that cache the solver count need it, so
// the group probes it behind if-constexpr and stateless schedules that read the
// count directly from select(n) need not provide it.
template <typename Schedule, typename S = double>
concept schedule = requires(Schedule& sched, std::size_t n,
                            const step_result<S>& result)
{
    { sched.select(n) } -> std::convertible_to<std::size_t>;
    { sched.reset() };
    { sched.notify(result) };
};

}

#endif
