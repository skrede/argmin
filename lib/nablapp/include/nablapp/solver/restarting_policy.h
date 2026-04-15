#ifndef HPP_GUARD_NABLAPP_SOLVER_RESTARTING_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_RESTARTING_POLICY_H

// IPOP restart decorator for population-based solver policies.
//
// Transparent GoF decorator that wraps any policy and adds restart-on-
// stagnation with increasing population size (IPOP). The decorator
// forwards init/step/reset/reset_clear to the inner policy and monitors
// for stagnation. When triggered, restart consumes one step() call.
//
// rebind<M> forwards through the decorator so basic_solver CTAD works
// transparently.
//
// Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy
//            with Increasing Population Size", CEC 2005.

#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace nablapp
{

namespace detail
{
// Resolve state_type: use Inner::state_type<P> when available, else Inner::state_type.
// Mirrors basic_solver's resolve_state_t for use in policy decorators.
template <typename InnerPolicy, typename P>
consteval auto resolve_inner_state_tag()
{
    if constexpr(requires { typename InnerPolicy::template state_type<P>; })
        return std::type_identity<typename InnerPolicy::template state_type<P>>{};
    else
        return std::type_identity<typename InnerPolicy::state_type>{};
}

template <typename InnerPolicy, typename P>
using resolve_inner_state_t =
    typename decltype(resolve_inner_state_tag<InnerPolicy, P>())::type;
}

template <typename Inner>
struct restarting_policy
{
    using scalar_type = typename Inner::scalar_type;

    template <int M>
    using rebind = restarting_policy<typename Inner::template rebind<M>>;

    struct options_type
    {
        typename Inner::options_type inner{};
        std::optional<std::uint32_t> stagnation_limit{};
        std::optional<double> population_multiplier{};
        // stall_window / feasibility_gate intentionally not declared on the decorator.
        // restarting_policy emits synthetic (obj_change=1.0, step_size=1.0) step_results on
        // restart and maintains its own internal stall logic (see the restart-trigger block
        // below). Exposing an external stall_window would double-fire against this internal
        // logic; feasibility_gate is inner-policy territory. Silent no-op made explicit here —
        // see forward_policy_hints in basic_solver.h.
    };

    template <typename P = void>
    struct state_type
    {
        detail::resolve_inner_state_t<Inner, P> inner;
        decltype(std::declval<detail::resolve_inner_state_t<Inner, P>>().x) x{};
        const P* problem{nullptr};
        decltype(x) x0_saved{};
        std::uint32_t stagnation_count{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
        bool restart_pending{false};
        std::uint32_t restart_count{0};
        std::uint32_t dimension{0};
        std::uint32_t initial_lambda{0};

        Eigen::VectorXd c_eq{};
        Eigen::VectorXd c_ineq{};
    };

    Inner inner_policy_{};
    options_type options{};

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                    const auto& x0,
                    const solver_options<Convergence>& opts)
    {
        inner_policy_.options = options.inner;

        state_type<Problem> s;
        s.problem = &problem;
        s.x0_saved = x0;
        s.inner = inner_policy_.init(problem, x0, opts);
        s.best_ever_value = s.inner.objective_value;
        s.stagnation_count = 0;
        s.restart_count = 0;
        s.restart_pending = false;
        s.dimension = static_cast<std::uint32_t>(x0.size());

        // Extract initial lambda from inner state
        if constexpr(requires { s.inner.lambda; })
            s.initial_lambda = static_cast<std::uint32_t>(s.inner.lambda);
        else if constexpr(requires { s.inner.params.lambda; })
            s.initial_lambda = static_cast<std::uint32_t>(s.inner.params.lambda);
        else
            s.initial_lambda = static_cast<std::uint32_t>(
                4 + 3 * std::log(static_cast<double>(s.dimension)));

        sync_from_inner(s);
        return s;
    }

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const auto& x0,
                    const solver_options<Convergence>& opts,
                    const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        if(s.restart_pending)
        {
            reinit(s);
            s.restart_pending = false;
            s.stagnation_count = 0;

            return step_result<scalar_type>{
                .objective_value = s.inner.objective_value,
                .gradient_norm = 1.0,
                .step_size = 1.0,
                .objective_change = 1.0,
                .improved = false,
                .x_norm = s.inner.x.norm(),
            };
        }

        auto result = inner_policy_.step(s.inner);

        // Stall detection
        if(result.objective_value < s.best_ever_value)
        {
            s.best_ever_value = result.objective_value;
            s.stagnation_count = 0;
        }
        else
        {
            ++s.stagnation_count;
        }

        std::uint32_t limit = options.stagnation_limit.value_or(
            static_cast<std::uint32_t>(
                10 + std::ceil(30.0 * s.dimension / s.initial_lambda)));

        if(s.stagnation_count >= limit)
            s.restart_pending = true;

        sync_from_inner(s);
        return result;
    }

    template <typename P>
    void reset(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset(s.inner, x0);
        s.stagnation_count = 0;
        s.best_ever_value = s.inner.objective_value;
        s.restart_pending = false;
        sync_from_inner(s);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset_clear(s.inner, x0);
        s.stagnation_count = 0;
        s.restart_count = 0;
        s.best_ever_value = s.inner.objective_value;
        s.restart_pending = false;
        sync_from_inner(s);
    }

private:

    template <typename P>
    void reinit(state_type<P>& s)
    {
        ++s.restart_count;

        // IPOP: population = initial_lambda * 2^restart_count
        double multiplier = options.population_multiplier.value_or(2.0);
        auto new_pop = static_cast<std::uint32_t>(
            s.initial_lambda * std::pow(multiplier, s.restart_count));

        // Update inner population size via the appropriate field
        if constexpr(requires { inner_policy_.options.population_size; })
            inner_policy_.options.population_size = new_pop;
        else if constexpr(requires { inner_policy_.options.lambda; })
            inner_policy_.options.lambda = new_pop;

        inner_policy_.reset_clear(s.inner, s.x0_saved);
        sync_from_inner(s);
    }

    template <typename P>
    static void sync_from_inner(state_type<P>& s)
    {
        s.x = s.inner.x;
        if constexpr(requires { s.inner.c_eq; s.inner.c_ineq; })
        {
            s.c_eq = s.inner.c_eq;
            s.c_ineq = s.inner.c_ineq;
        }
    }
};

}

#endif
