#ifndef HPP_GUARD_ARGMIN_SOLVER_STEP_AND_TIME_BUDGET_SOLVER_H
#define HPP_GUARD_ARGMIN_SOLVER_STEP_AND_TIME_BUDGET_SOLVER_H

#include "argmin/types.h"
#include "argmin/detail/solve_loop.h"
#include "argmin/detail/solver_core.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/timed_solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"
#include "argmin/solver/time_budget_options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <chrono>
#include <utility>
#include <cstdint>
#include <type_traits>

namespace argmin
{

// Combined step + wall-clock budget driver -- the like-for-like successor of
// the original step_budget_solver semantics, where both an iteration cap and a
// wall-clock deadline are first-class stopping conditions.
//
// Mechanically it is the same thin facade over detail::solver_core +
// detail::run_solve_loop as time_budget_solver: solve()/step_n() run the loop
// up to max_iterations while the deadline predicate (polled every
// time_poll_stride iterations) stops it earlier once max_time elapses.
// Whichever bound is reached first -- iteration cap, wall-clock deadline, or
// the convergence policy -- terminates the run. The distinction from
// time_budget_solver is one of contract, not implementation: here the
// iteration cap is an equal, advertised budget rather than a safety backstop.
//
// The result is timed_solve_result (chrono-free diagnostics plus wall_time);
// because it publicly derives from solve_result, this driver satisfies
// nlp_solver as the other drivers do. Every chrono symbol is confined to this
// time path.
//
// Every ctor and both reset entry points take x0 as a deduced
// Eigen::MatrixBase<Derived> so a fixed-N caller vector binds without a heap
// materialization.
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class step_and_time_budget_solver
{
public:
    using core_type = detail::solver_core<Policy, N, Problem, Convergence>;
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename core_type::state_type;
    using result_type = timed_solve_result<scalar_type, N>;

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
    step_and_time_budget_solver(Policy policy, const P& problem,
                                const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{std::move(policy), problem, to_x0(x0), opts.core}
    {}

    // CTAD converting constructor: rebinds an un-rebound policy to the
    // problem's compile-time dimension, preserving the caller's options.
    template <typename OrigPolicy, typename P, typename Derived>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    step_and_time_budget_solver(OrigPolicy&& orig, const P& problem,
                                const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts.core}
    {}

    template <typename P, typename Derived>
    step_and_time_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts = {})
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{problem, to_x0(x0), opts.core}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_and_time_budget_solver(Policy policy, const P& problem,
                                const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts,
                                const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{std::move(policy), problem, to_x0(x0), opts.core, policy_opts}
    {}

    template <typename OrigPolicy, typename P, typename Derived, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_and_time_budget_solver(OrigPolicy&& orig, const P& problem,
                                const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts,
                                const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts.core, policy_opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    step_and_time_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts,
                                const PolicyOpts& policy_opts)
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{problem, to_x0(x0), opts.core, policy_opts}
    {}

    // Overload for policies without options_type: the fifth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P, typename Derived>
        requires (!has_options_type<Policy>)
    step_and_time_budget_solver(const P& problem, const Eigen::MatrixBase<Derived>& x0,
                                const time_budget_options<Convergence>& opts,
                                const solver_options<>& policy_opts)
        : max_time_{opts.max_time}
        , poll_stride_{opts.time_poll_stride}
        , core_{problem, to_x0(x0), opts.core, policy_opts}
    {}

    step_and_time_budget_solver(const step_and_time_budget_solver&) = delete;
    step_and_time_budget_solver& operator=(const step_and_time_budget_solver&) = delete;
    step_and_time_budget_solver(step_and_time_budget_solver&&) noexcept = default;
    step_and_time_budget_solver& operator=(step_and_time_budget_solver&&) noexcept = default;

    step_result<scalar_type> step() { return core_.step(); }

    // Solve with both budgets: max_iterations and the wall-clock deadline.
    result_type solve()
    {
        return run(core_.max_iterations(), make_default_opts(),
                   max_time_, poll_stride_);
    }

    template <typename ExplicitConvergence>
    result_type solve(const time_budget_options<ExplicitConvergence>& opts)
    {
        return run(opts.core.max_iterations, opts.core,
                   opts.max_time, opts.time_poll_stride);
    }

    result_type step_n(std::uint32_t budget)
    {
        return run(budget, make_default_opts(), max_time_, poll_stride_);
    }

    template <typename ExplicitConvergence>
    result_type step_n(std::uint32_t budget,
                       const time_budget_options<ExplicitConvergence>& opts)
    {
        return run(budget, opts.core, opts.max_time, opts.time_poll_stride);
    }

    const state_type& state() const { return core_.state(); }
    const auto& convergence() const { return core_.convergence(); }
    solver_status status() const { return core_.status(); }
    const Policy& policy() const { return core_.policy(); }
    scalar_type gradient_norm() const { return core_.gradient_norm(); }
    scalar_type constraint_violation() const { return core_.constraint_violation(); }
    void abort() { core_.abort(); }

    template <typename Derived>
    void reset(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        core_.reset(x0);
    }

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
    solver_options<Convergence> make_default_opts() const
    {
        solver_options<Convergence> opts;
        opts.max_iterations = core_.max_iterations();
        opts.constraint_tolerance = core_.constraint_tolerance();
        opts.feasibility_tolerance = core_.feasibility_tolerance();
        opts.convergence = core_.convergence();
        return opts;
    }

    // Drive the shared loop under both the iteration budget and the wall-clock
    // deadline. The deadline origin t0 and the reported wall_time share the
    // identical stamp captured here. The deadline is polled every `stride`
    // iterations (stride 0 == every iteration).
    template <typename OptsConvergence>
    result_type run(std::uint32_t budget,
                    const solver_options<OptsConvergence>& core_opts,
                    std::chrono::nanoseconds max_t, std::uint32_t stride)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + max_t;
        auto exhausted = [deadline, stride](std::uint32_t i) -> bool
        {
            if(stride == 0 || (i % stride) == 0)
                return std::chrono::steady_clock::now() >= deadline;
            return false;
        };
        auto base = detail::run_solve_loop(core_, budget, core_opts, exhausted);
        const auto t1 = std::chrono::steady_clock::now();

        result_type r;
        static_cast<solve_result<scalar_type, N>&>(r) = base;
        r.wall_time = t1 - t0;
        return r;
    }

    std::chrono::nanoseconds max_time_{std::chrono::nanoseconds::zero()};
    std::uint32_t poll_stride_{20};
    core_type core_;
};

// CTAD deduction guides mirroring the other drivers'.

template <typename Policy, typename Problem, typename Derived, typename Convergence>
step_and_time_budget_solver(Policy, const Problem&,
                            const Eigen::MatrixBase<Derived>&,
                            const time_budget_options<Convergence>&)
    -> step_and_time_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                                   problem_dimension_v<Problem>,
                                   Problem,
                                   Convergence>;

template <typename Policy, typename Problem, typename Derived>
step_and_time_budget_solver(Policy, const Problem&, const Eigen::MatrixBase<Derived>&)
    -> step_and_time_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                                   problem_dimension_v<Problem>,
                                   Problem>;

template <typename Policy, typename Problem, typename Derived, typename Convergence,
          typename PolicyOpts>
step_and_time_budget_solver(Policy, const Problem&,
                            const Eigen::MatrixBase<Derived>&,
                            const time_budget_options<Convergence>&, const PolicyOpts&)
    -> step_and_time_budget_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                                   problem_dimension_v<Problem>,
                                   Problem,
                                   Convergence>;

}

#endif
