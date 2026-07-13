#ifndef HPP_GUARD_ARGMIN_SOLVER_STEP_BUDGET_SOLVER_H
#define HPP_GUARD_ARGMIN_SOLVER_STEP_BUDGET_SOLVER_H

#include "argmin/types.h"
#include "argmin/detail/solve_loop.h"
#include "argmin/detail/solver_core.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <utility>
#include <cstdint>
#include <type_traits>

namespace argmin
{

// Real-time step-budget driver: budget is max_iterations only.
//
// step_budget_solver is a thin facade over detail::solver_core composed with
// the ONE detail::run_solve_loop. The core owns the wrapped Policy, its
// resolved state, the problem binding, the counters, the stored convergence
// criteria, the abort flag, and the by-step convergence consultation. This
// driver supplies the iterative execution model on top: step(), solve(),
// step_n(budget). solve/step_n delegate their loop body to the single
// detail::run_solve_loop helper, passing an always-false budget predicate --
// there is no wall clock on this path. That confinement is structural: this
// header includes NO <chrono>, and neither does the solve loop it composes,
// so a translation unit budgeting purely by iterations never pulls the clock
// in. Wall-clock budgeting lives on the sibling time_budget_solver /
// step_and_time_budget_solver drivers.
//
// step_budget_solver<Policy, N, Problem> has three template parameters: the
// policy, the compile-time dimension N, and the Problem type. N and Problem
// are deduced from the problem via CTAD deduction guides -- users never write
// N or Problem explicitly.
//
// For policies with non-template state_type (pre-conversion), Problem
// defaults to void and is ignored. For policies with template state_type<P>,
// Problem is used to instantiate the state.
//
// The policy contract is enforced by the solver_policy concept
// (argmin/formulation/concepts.h), static_assert-ed inside solver_core where
// state_type resolves.
//
// Every ctor and both reset entry points take x0 as a deduced
// Eigen::MatrixBase<Derived>: a fixed-N caller vector binds without a heap
// materialization, and a static_assert names the expected vector shape.
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class step_budget_solver
{
public:
    using core_type = detail::solver_core<Policy, N, Problem, Convergence>;
    using policy_type = Policy;
    // scalar_type is derived directly from Policy (not core_type) so the x0
    // constructor parameter mentions only Policy. Routing it through
    // core_type would drag N/Problem into the (non-deduced) x0 diagnostic and
    // perturb CTAD guide selection.
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename core_type::state_type;

    // Diagnostic for the deduced x0 expression: a fixed-N caller must pass an
    // N-row column vector. A mismatch fails here with a one-line message
    // instead of deep inside Eigen's assignment. dynamic_dimension (N == -1)
    // accepts any column count.
    template <typename Derived>
    static constexpr bool x0_shape_ok =
        (Derived::ColsAtCompileTime == 1) &&
        (N == dynamic_dimension || Derived::RowsAtCompileTime == Eigen::Dynamic ||
         Derived::RowsAtCompileTime == N);

    template <typename Derived>
    static Eigen::Vector<scalar_type, N> to_x0(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        return Eigen::Vector<scalar_type, N>(x0);
    }

    template <typename P, typename Derived>
    step_budget_solver(Policy policy, const P& problem,
                       const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts = {})
        : core_{std::move(policy), problem, to_x0(x0), opts}
    {}

    // CTAD converting constructor: accepts an un-rebound policy and rebinds
    // it to the problem's compile-time dimension, preserving the caller's
    // configured options.
    template <typename OrigPolicy, typename P, typename Derived>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    step_budget_solver(OrigPolicy&& orig, const P& problem,
                       const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts = {})
        : core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts}
    {}

    template <typename P, typename Derived>
    step_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts = {})
        : core_{problem, to_x0(x0), opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_budget_solver(Policy policy, const P& problem,
                       const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts,
                       const PolicyOpts& policy_opts)
        : core_{std::move(policy), problem, to_x0(x0), opts, policy_opts}
    {}

    // CTAD converting constructor with policy options.
    template <typename OrigPolicy, typename P, typename Derived, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_budget_solver(OrigPolicy&& orig, const P& problem,
                       const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts,
                       const PolicyOpts& policy_opts)
        : core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts, policy_opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts,
                       const PolicyOpts& policy_opts)
        : core_{problem, to_x0(x0), opts, policy_opts}
    {}

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P, typename Derived>
        requires (!has_options_type<Policy>)
    step_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                       const solver_options<Convergence>& opts,
                       const solver_options<>& policy_opts)
        : core_{problem, to_x0(x0), opts, policy_opts}
    {}

    step_budget_solver(const step_budget_solver&) = delete;
    step_budget_solver& operator=(const step_budget_solver&) = delete;
    step_budget_solver(step_budget_solver&&) noexcept = default;
    step_budget_solver& operator=(step_budget_solver&&) noexcept = default;

    // Single iteration with per-step convergence consultation. See
    // solver_core::step().
    step_result<scalar_type> step() { return core_.step(); }

    // Solve with stored convergence, budgeted by the stored max_iterations.
    // Re-entrancy matches solver_core: solve() and step_n(budget) are
    // continuations, not restarts -- call reset()/reset_clear() first to
    // restart from a clean counter and a new starting point.
    solve_result<scalar_type, N> solve()
    {
        return step_n(core_.max_iterations(), make_default_opts());
    }

    // Solve with an explicit convergence policy via solver_options.
    template <typename ExplicitConvergence>
    solve_result<scalar_type, N> solve(const solver_options<ExplicitConvergence>& opts)
    {
        return step_n(opts.max_iterations, opts);
    }

    // Step with budget, stored convergence.
    solve_result<scalar_type, N> step_n(std::uint32_t budget)
    {
        return step_n(budget, make_default_opts());
    }

    // Step with budget and an explicit convergence policy via solver_options.
    //
    // Delegates the loop body to detail::run_solve_loop with an always-false
    // budget predicate: this driver budgets by iterations alone, so no wall
    // clock is ever consulted.
    template <typename OptsConvergence>
    solve_result<scalar_type, N> step_n(std::uint32_t budget,
                                        const solver_options<OptsConvergence>& opts)
    {
        auto never_exhausted = [](std::uint32_t) -> bool { return false; };
        return detail::run_solve_loop(core_, budget, opts, never_exhausted);
    }

    const state_type& state() const { return core_.state(); }

    // Read the stored convergence policy (populated with per-criterion
    // last_check_results by the most recent step_n() call).
    const auto& convergence() const { return core_.convergence(); }

    solver_status status() const { return core_.status(); }

    // The wrapped policy instance (options included).
    const Policy& policy() const { return core_.policy(); }

    // The gradient-like norm reported by the most recent step, or a NaN
    // sentinel when no step has run yet.
    scalar_type gradient_norm() const { return core_.gradient_norm(); }

    scalar_type constraint_violation() const { return core_.constraint_violation(); }

    void abort() { core_.abort(); }

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs).
    template <typename Derived>
    void reset(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        core_.reset(x0);
    }

    // Reset to a new starting point, clearing all algorithm state.
    template <typename Derived>
    void reset_clear(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        core_.reset_clear(x0);
    }

private:
    // Rebuild the local solver_options for the defaulted solve() /
    // step_n(budget) paths from the constructor-configured stored state.
    solver_options<Convergence> make_default_opts() const
    {
        solver_options<Convergence> opts;
        opts.max_iterations = core_.max_iterations();
        opts.constraint_tolerance = core_.constraint_tolerance();
        opts.feasibility_tolerance = core_.feasibility_tolerance();
        opts.convergence = core_.convergence();
        return opts;
    }

    core_type core_;
};

// CTAD deduction guides.
//
// These guides extract N from the problem type via problem_dimension_v<Problem>
// and rebind the policy to that dimension via Policy::template rebind<N>.
// Problem is deduced from the constructor argument and stored as the third
// template parameter for policies with template state_type<P>.
//
// The x0 parameter is a generic Eigen::MatrixBase<Derived> so a fixed-N caller
// vector binds directly; Convergence is deduced from the opts argument's
// solver_options<Convergence> type where present, else keeps the class default.

// Policy + Problem + x0 + opts
template <typename Policy, typename Problem, typename Derived, typename Convergence>
step_budget_solver(Policy, const Problem&,
                   const Eigen::MatrixBase<Derived>&,
                   const solver_options<Convergence>&)
    -> step_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                          problem_dimension_v<Problem>,
                          Problem,
                          Convergence>;

// Policy + Problem + x0 (no opts): Convergence keeps the class default.
template <typename Policy, typename Problem, typename Derived>
step_budget_solver(Policy, const Problem&, const Eigen::MatrixBase<Derived>&)
    -> step_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                          problem_dimension_v<Problem>,
                          Problem>;

// Policy + Problem + x0 + opts + policy_opts
template <typename Policy, typename Problem, typename Derived, typename Convergence,
          typename PolicyOpts>
    requires has_options_type<typename Policy::template rebind<problem_dimension_v<Problem>>>
          && std::same_as<std::remove_cvref_t<PolicyOpts>,
                 typename Policy::template rebind<problem_dimension_v<Problem>>::options_type>
step_budget_solver(Policy, const Problem&,
                   const Eigen::MatrixBase<Derived>&,
                   const solver_options<Convergence>&, const PolicyOpts&)
    -> step_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                          problem_dimension_v<Problem>,
                          Problem,
                          Convergence>;

}

#endif
