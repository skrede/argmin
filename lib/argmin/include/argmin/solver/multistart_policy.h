#ifndef HPP_GUARD_ARGMIN_SOLVER_MULTISTART_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_MULTISTART_POLICY_H

// Multi-start restart decorator for derivative-free solver policies.
//
// Transparent GoF decorator that wraps any policy and adds Halton-sequence
// multi-start restarts. When the inner solver stagnates, the decorator
// generates a new starting point via Halton quasi-random sequence mapped
// to the problem bounds, then restarts the inner solver from that point.
//
// Unlike CMA-ES IPOP (which doubles population size), this generates
// DIVERSE starting points across the feasible region. Suitable for any
// derivative-free policy (BOBYQA, CMA-ES).
//
// Reference: Halton, J. H. (1960), quasi-random sequence seeding.
//            K&W Section 8.1 (multi-start global optimization).

#include "argmin/detail/halton.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/solver/restarting_policy.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cstdint>
#include <limits>
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

    struct options_type
    {
        typename Inner::options_type inner{};
        std::optional<std::uint16_t> max_restarts{};
        std::optional<std::uint32_t> stall_budget_per_restart{};
        // stall_window / feasibility_gate intentionally not declared on the decorator.
        // multistart_policy emits synthetic (obj_change=1.0, step_size=1.0) step_results on
        // restart-consuming steps; exposing an external stall_window would double-fire against
        // the inner policy's own stall logic, which is already forwarded via options.inner.
        // feasibility_gate is inner-policy territory for the same reason. Silent no-op made
        // explicit here — see forward_policy_hints in basic_solver.h for the plumbing that
        // this SKIP deactivates.
    };

    // Reuses restarting_policy's inner-state resolution (state_type<P> when the
    // inner policy templates on the problem, plain state_type otherwise) instead
    // of duplicating it, and derives x / lower / upper / best_ever_x from the
    // inner state's own vector type so fixed-N inner policies keep fully-fixed
    // state inside this decorator rather than collapsing to Eigen::VectorXd.
    template <typename P = void>
    struct state_type
    {
        detail::resolve_inner_state_t<Inner, P> inner;
        decltype(std::declval<detail::resolve_inner_state_t<Inner, P>>().x) x{};
        decltype(x) lower{};
        decltype(x) upper{};
        double objective_value{};
        std::uint32_t stagnation_count{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
        decltype(x) best_ever_x{};
        bool restart_pending{false};
        std::uint16_t restart_count{0};
        int dimension{0};

        // Mirrors the inner policy's constraint residuals so
        // basic_solver::constraint_violation() reads the real inner violation
        // for constrained inner policies instead of silently reading zero.
        Eigen::VectorXd c_eq{};
        Eigen::VectorXd c_ineq{};
    };

    Inner inner_policy_{};
    options_type options{};

    static constexpr std::uint16_t default_max_restarts = 10;

    std::uint16_t effective_max_restarts() const
    {
        return options.max_restarts.value_or(default_max_restarts);
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

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem, const auto& x0,
                             const solver_options<Convergence>& opts)
    {
        state_type<Problem> s;
        s.inner = inner_policy_.init(problem, x0, opts);
        s.best_ever_value = s.inner.objective_value;
        s.best_ever_x = s.inner.x;
        s.dimension = problem.dimension();

        if constexpr(bound_constrained<Problem>)
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
        {
            s.restart_pending = false;

            if(s.restart_count >= effective_max_restarts())
            {
                // All restarts exhausted -- restore best-ever solution and
                // report an honest terminal status. The restart budget
                // running out proves nothing about optimality, so this must
                // not be reported as solver_status::converged.
                s.x = s.best_ever_x;
                s.objective_value = s.best_ever_value;
                s.inner.x = s.best_ever_x;
                s.inner.objective_value = s.best_ever_value;
                return step_result<scalar_type>{
                    .objective_value = s.best_ever_value,
                    .gradient_norm = 0.0,
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                    .policy_status = solver_status::budget_exhausted,
                };
            }

            // Generate Halton-seeded starting point within bounds. Widened to
            // Eigen::VectorXd on both ternary branches: halton_to_bounds always
            // returns a dynamic vector, while s.best_ever_x may now be a
            // fixed-N vector for fixed-N inner policies, and the conditional
            // operator requires a single common type.
            Eigen::VectorXd new_x0 = (s.lower.size() > 0 && s.upper.size() > 0)
                ? detail::halton_to_bounds(s.restart_count, s.dimension, s.lower, s.upper)
                : Eigen::VectorXd(s.best_ever_x);

            inner_policy_.reset_clear(s.inner, new_x0);
            ++s.restart_count;
            s.stagnation_count = 0;

            sync_from_inner(s);

            return step_result<scalar_type>{
                .objective_value = s.inner.objective_value,
                .gradient_norm = 1.0,
                .step_size = 1.0,
                .objective_change = 1.0,
                .improved = false,
            };
        }

        // Forward to inner policy.
        auto result = inner_policy_.step(s.inner);

        // Track best-ever across all restarts.
        if(result.objective_value < s.best_ever_value)
        {
            s.best_ever_value = result.objective_value;
            s.best_ever_x = s.inner.x;
            s.stagnation_count = 0;
        }
        else
        {
            ++s.stagnation_count;
        }

        // Stall detection: trigger restart if no improvement for stall_budget steps.
        if(s.stagnation_count >= effective_stall_budget(s.dimension))
            s.restart_pending = true;

        sync_from_inner(s);

        return result;
    }

    template <typename P>
    void reset(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset(s.inner, x0);
        s.stagnation_count = 0;
        s.restart_pending = false;
        sync_from_inner(s);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset_clear(s.inner, x0);
        s.stagnation_count = 0;
        s.restart_pending = false;
        s.restart_count = 0;
        s.best_ever_value = s.inner.objective_value;
        s.best_ever_x = s.inner.x;
        sync_from_inner(s);
    }

private:
    template <typename P>
    static void sync_from_inner(state_type<P>& s)
    {
        s.x = s.inner.x;
        s.objective_value = s.inner.objective_value;
        if constexpr(requires { s.inner.c_eq; s.inner.c_ineq; })
        {
            s.c_eq = s.inner.c_eq;
            s.c_ineq = s.inner.c_ineq;
        }
    }
};

}

#endif
