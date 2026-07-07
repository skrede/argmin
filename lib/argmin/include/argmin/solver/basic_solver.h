#ifndef HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H

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

#include <chrono>
#include <utility>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace argmin
{

// Solver wrapper with convergence loop.
//
// basic_solver is a thin facade over detail::solver_core: the core owns the
// wrapped Policy, its resolved state, the problem binding, the counters, the
// stored convergence criteria, the abort flag, and the by-step convergence
// consultation. basic_solver supplies the iterative execution model on top:
// step(), solve(), step_n(budget). solve/step_n delegate their loop body to
// the single detail::run_solve_loop helper, passing a budget predicate that
// wraps the wall-time deadline (max_time_ stays a basic_solver member and the
// steady_clock reads live only in that predicate). Convergence checking is
// performed by the core -- the policy does NOT know about convergence.
//
// basic_solver<Policy, N, Problem> has three template parameters:
// the policy, the compile-time dimension N, and the Problem type.
// N and Problem are deduced from the problem via CTAD deduction guides.
// Users never write N or Problem explicitly -- CTAD handles both.
//
// For policies with non-template state_type (pre-conversion), Problem
// defaults to void and is ignored. For policies with template
// state_type<P>, Problem is used to instantiate the state.
//
// The policy contract is enforced by the solver_policy concept
// (argmin/formulation/concepts.h), static_assert-ed inside solver_core where
// state_type resolves. It pins the harness-visible surface (scalar_type;
// step/reset/reset_clear; state.x); init() is intentionally left unconstrained
// since it is templated on Problem/Convergence and its shape is an
// implementation detail.
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class basic_solver
{
public:
    using core_type = detail::solver_core<Policy, N, Problem, Convergence>;
    using policy_type = Policy;
    // scalar_type is derived directly from Policy (not core_type) so the x0
    // constructor parameter Eigen::VectorX<scalar_type> mentions only Policy.
    // Routing it through core_type would drag N/Problem into that parameter's
    // (non-deduced) type and perturb CTAD guide selection, letting the
    // implicit constructor guide default Problem to void instead of the
    // explicit guide deducing it from the problem argument.
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename core_type::state_type;

    template <typename P>
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , core_{std::move(policy), problem, x0, opts}
    {}

    // CTAD converting constructor: accepts an un-rebound policy and rebinds
    // it to the problem's compile-time dimension, preserving the caller's
    // configured options. The requires-clause strips cvref from OrigPolicy so
    // both lvalue and rvalue source policies are accepted (an lvalue deduces
    // OrigPolicy as a reference type, which has no nested rebind alias).
    template <typename OrigPolicy, typename P>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    basic_solver(OrigPolicy&& orig, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , core_{std::forward<OrigPolicy>(orig), problem, x0, opts}
    {}

    template <typename P>
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , core_{problem, x0, opts}
    {}

    template <typename P, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , core_{std::move(policy), problem, x0, opts, policy_opts}
    {}

    // CTAD converting constructor with policy options. The explicit
    // policy_opts (already rebound-typed) is installed by init(); the
    // requires-clause strips cvref from OrigPolicy so an lvalue source policy
    // is accepted.
    template <typename OrigPolicy, typename P, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(OrigPolicy&& orig, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , core_{std::forward<OrigPolicy>(orig), problem, x0, opts, policy_opts}
    {}

    template <typename P, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , core_{problem, x0, opts, policy_opts}
    {}

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P>
        requires (!has_options_type<Policy>)
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const solver_options<>& policy_opts)
        : max_time_{opts.max_time}
        , core_{problem, x0, opts, policy_opts}
    {}

    basic_solver(const basic_solver&) = delete;
    basic_solver& operator=(const basic_solver&) = delete;
    basic_solver(basic_solver&&) noexcept = default;
    basic_solver& operator=(basic_solver&&) noexcept = default;

    // Single iteration with per-step convergence consultation. See
    // solver_core::step().
    step_result<scalar_type> step() { return core_.step(); }

    // Solve with default convergence (uses stored max_iterations).
    //
    // Builds solver_options<Convergence> (the class's own stored convergence
    // type, not solver_options<>) so a non-default Convergence is used
    // without coercion, and copies the constructor-configured
    // constraint_tolerance / feasibility_tolerance into the local opts so the
    // best-seen feasibility comparator honors them instead of reverting to
    // solver_options's 1e-6 default.
    //
    // Re-entrancy: solve() and step_n(budget) are continuations, not
    // restarts. Each call executes up to its budget of ADDITIONAL iterations
    // from the current iterate, and the iteration counter, evaluation
    // counter, and any windowed convergence state accumulate across calls. A
    // second solve() therefore resumes where the first stopped (a warm
    // restart), which is the semantics a tick-loop caller wants. To restart
    // from a clean counter and a new starting point, call reset() (preserves
    // algorithm state) or reset_clear() (drops it) first.
    solve_result<scalar_type, N> solve()
    {
        return step_n(core_.max_iterations(), make_default_opts());
    }

    // Solve with explicit convergence policy via solver_options.
    template <typename ExplicitConvergence>
    solve_result<scalar_type, N> solve(const solver_options<ExplicitConvergence>& opts)
    {
        return step_n(opts.max_iterations, opts);
    }

    // Step with budget, default convergence. See solve() above for the
    // rationale on building solver_options<Convergence> and copying the
    // stored tolerances.
    solve_result<scalar_type, N> step_n(std::uint32_t budget)
    {
        return step_n(budget, make_default_opts());
    }

    // Step with budget and explicit convergence policy via solver_options.
    //
    // Delegates the loop body to detail::run_solve_loop, supplying a budget
    // predicate that wraps the wall-time deadline. The steady_clock reads
    // live only inside this predicate (chrono confinement to the dedicated
    // time drivers is a later step); the predicate measures against the
    // loop's own start stamp t0, so the deadline uses the identical origin
    // the reported wall_time is measured from.
    template <typename OptsConvergence>
    solve_result<scalar_type, N> step_n(std::uint32_t budget,
                                        const solver_options<OptsConvergence>& opts)
    {
        auto budget_exhausted =
            [&opts](std::chrono::steady_clock::time_point t0) -> bool
        {
            if(opts.max_time)
            {
                auto elapsed = std::chrono::steady_clock::now() - t0;
                return elapsed >= *opts.max_time;
            }
            return false;
        };
        return detail::run_solve_loop(core_, budget, opts, budget_exhausted);
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
    void reset(const Eigen::VectorX<scalar_type>& x0) { core_.reset(x0); }

    // Reset to a new starting point, clearing all algorithm state.
    void reset_clear(const Eigen::VectorX<scalar_type>& x0) { core_.reset_clear(x0); }

private:
    // Rebuild the local solver_options for the defaulted solve() /
    // step_n(budget) paths from the constructor-configured stored state.
    // max_time_ is the basic_solver-owned wall-time budget; the remaining
    // fields are read back from the core.
    solver_options<Convergence> make_default_opts() const
    {
        solver_options<Convergence> opts;
        opts.max_iterations = core_.max_iterations();
        opts.max_time = max_time_;
        opts.constraint_tolerance = core_.constraint_tolerance();
        opts.feasibility_tolerance = core_.feasibility_tolerance();
        opts.convergence = core_.convergence();
        return opts;
    }

    // The wall-time budget stays a basic_solver member (not the core): the
    // steady_clock deadline is the driver's concern, wired through the budget
    // predicate in step_n. Confining the chrono machinery to a dedicated time
    // driver is a later step.
    std::optional<std::chrono::nanoseconds> max_time_{};
    core_type core_;
};

// CTAD deduction guides.
//
// These guides extract N from the problem type via problem_dimension_v<Problem>
// and rebind the policy to that dimension via Policy::template rebind<N>.
// Problem is deduced from the constructor argument and stored as the third
// template parameter for policies with template state_type<P>.
//
// The x0 parameter uses Eigen::VectorX<typename Policy::scalar_type> explicitly
// (not a generic X) to match the constructor signature precisely. GCC 15 CTAD
// resolution requires this to correctly deduce Problem when state_type is a
// template -- a generic X parameter causes the guide to be ignored in favour
// of the implicit deduction guide which defaults Problem to void.

// Policy + Problem + x0 + opts
//
// Convergence is deduced here from the opts argument's solver_options<
// Convergence> type and flows into the class's own Convergence template
// parameter, so a non-default Convergence (e.g. a custom relative-tolerance
// policy) is stored without coercion to default_convergence.
template <typename Policy, typename Problem, typename Convergence>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&,
             const solver_options<Convergence>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem,
                    Convergence>;

// Policy + Problem + x0 (no opts): Convergence is not deducible here, so it
// keeps the class's own default_convergence default.
template <typename Policy, typename Problem>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem>;

// Policy + Problem + x0 + opts + policy_opts
template <typename Policy, typename Problem, typename Convergence,
          typename PolicyOpts>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&,
             const solver_options<Convergence>&, const PolicyOpts&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem,
                    Convergence>;

}

#endif
