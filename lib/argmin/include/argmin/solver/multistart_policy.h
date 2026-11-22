#ifndef HPP_GUARD_ARGMIN_SOLVER_MULTISTART_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_MULTISTART_POLICY_H

// Multi-start decorator: runs the inner policy from 1 + max_restarts starts and returns the best run. A run ends on
// any terminal verdict of its own (the inner policy's status, else the decorator's copy of the convergence settings)
// or when it stops improving; the next run starts from a Halton point in the box, or from the best run's point when
// there is no box. Only the last run's end ends the solve, with the best run's result and status.
// Reference: Halton, J. H. (1960), quasi-random sequence seeding.
//            K&W Section 8.1 (multi-start global optimization).

#include "argmin/detail/halton.h"
#include "argmin/detail/evaluation_count.h"
#include "argmin/detail/decorator_verdict.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/solver/restarting_policy.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <utility>

namespace argmin
{

template <typename Inner>
struct multistart_policy
{
    using scalar_type = typename Inner::scalar_type;

    template <int M>
    using rebind = multistart_policy<typename Inner::template rebind<M>>;

    static constexpr std::uint16_t default_max_restarts = 10;

    struct options_type
    {
        typename Inner::options_type inner{};
        std::uint16_t max_restarts{default_max_restarts};
        std::optional<std::uint32_t> stall_budget_per_restart{};  // truly optional: unset = auto 10 + 30 n
        // No stall_window or feasibility_gate: each run is judged by the decorator's own convergence copy under the
        // inner policy's options, and the solve's criteria never see a run's steps.
    };

    // Reuses restarting_policy's inner-state resolution (state_type<P> when the inner policy templates on the problem,
    // plain state_type otherwise) and derives x / lower / upper from the inner state's own vector type so fixed-N
    // inner policies keep fully-fixed state inside this decorator rather than collapsing to a dynamic vector.
    template <typename P = void>
    struct state_type : detail::wrapped_evaluations<detail::resolve_inner_state_t<Inner, P>>
    {
        detail::resolve_inner_state_t<Inner, P> inner;
        decltype(std::declval<detail::resolve_inner_state_t<Inner, P>>().x) x{};
        decltype(x) lower{};
        decltype(x) upper{};
        scalar_type objective_value{};
        detail::member_judge<scalar_type> judge{};
        detail::best_member<decltype(x), scalar_type> best{};
        bool restart_pending{false};
        std::uint16_t restart_count{0};
        int dimension{0};

        // Mirrors the inner policy's constraint residuals so step_budget_solver::constraint_violation() reads the real
        // inner violation for constrained inner policies instead of silently reading zero.
        Eigen::VectorX<scalar_type> c_eq{};
        Eigen::VectorX<scalar_type> c_ineq{};
    };

    Inner inner_policy_{};
    options_type options{};

    std::uint16_t effective_max_restarts() const
    {
        return options.max_restarts;
    }

    std::uint32_t effective_stall_budget(int n) const
    {
        return options.stall_budget_per_restart.value_or(
            static_cast<std::uint32_t>(10 + 30 * n));
    }

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem, const auto& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        inner_policy_.options = policy_opts.inner;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem, const auto& x0,
                             const solver_options<Convergence>& opts)
    {
        state_type<Problem> s;
        s.inner = inner_policy_.init(problem, x0, opts);
        s.dimension = problem.dimension();
        s.judge = detail::member_judge<scalar_type>{opts, inner_policy_, effective_stall_budget(s.dimension)};

        if constexpr(detail::boxed_problem<Problem, scalar_type>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }

        sync_from_inner(s);
        return s;
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        if(s.restart_pending)
            start_next(s);
        const step_result<scalar_type> result = inner_policy_.step(s.inner);
        sync_from_inner(s);
        return detail::member_step<P>(s, result, [&](solver_status) { return s.restart_count >= effective_max_restarts(); });
    }

    template <typename P>
    solver_status reset(state_type<P>& s, const auto& x0)
    {
        return settle(s, inner_policy_.reset(s.inner, x0));
    }

    template <typename P>
    solver_status reset_clear(state_type<P>& s, const auto& x0)
    {
        return settle(s, inner_policy_.reset_clear(s.inner, x0));
    }

private:
    // The box is a property of the problem type: a fixed-N state holds unset lower and upper vectors of nonzero size.
    template <typename P>
    void start_next(state_type<P>& s)
    {
        if constexpr(detail::boxed_problem<P, scalar_type>)
            inner_policy_.reset_clear(s.inner, detail::halton_to_bounds<scalar_type>(s.restart_count, s.dimension, s.lower, s.upper));
        else
            inner_policy_.reset_clear(s.inner, s.best.x());
        ++s.restart_count;
        s.restart_pending = false;
        s.judge.restart();
        sync_from_inner(s);
    }

    template <typename P>
    static solver_status settle(state_type<P>& s, solver_status status)
    {
        s.judge.restart();
        s.best.clear();
        s.restart_pending = false;
        s.restart_count = 0;
        sync_from_inner(s);
        return status;
    }

    template <typename P>
    static void sync_from_inner(state_type<P>& s)
    {
        s.x = s.inner.x;
        s.objective_value = s.inner.objective_value;
        detail::mirror_evaluations(s, s.inner);
        if constexpr(requires { s.inner.c_eq; s.inner.c_ineq; })
        {
            s.c_eq = s.inner.c_eq;
            s.c_ineq = s.inner.c_ineq;
        }
    }
};

}

#endif
